//! LLM post-processing backend for Gemma-3n (text-only).
//!
//! Supports `onnx-community/gemma-3n-E2B-it-ONNX` (and similar Gemma-3n
//! exports).  Only the text decoder is used — audio_encoder and
//! vision_encoder are intentionally ignored.
//!
//! # Schema (from graph inspection)
//!
//! ## embed_tokens
//!   input:  `input_ids`        i64  [batch, seq]
//!   output: `inputs_embeds`    f32  [batch, seq, 2048]
//!           `per_layer_inputs` f32  [batch, seq, 30, 256]
//!
//! ## decoder_model_merged
//!   inputs:
//!     `inputs_embeds`          f32  [batch, seq, 2048]
//!     `position_ids`           i64  [batch, seq]           ← rank-2
//!     `per_layer_inputs`       f32  [batch, seq, 30, 256]
//!     `past_key_values.N.key`  f32  [batch, 2, past, 256]  (N = 0..29)
//!     `past_key_values.N.value`f32  [batch, 2, past, 256]
//!   outputs:
//!     `logits`                 f32  [batch, seq, 262144]
//!     `present.N.key`          f32  [batch, 2, total, 256]
//!     `present.N.value`        f32  [batch, 2, total, 256]
//!
//! # Chat template
//!   `<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
//!
//! # Stop tokens
//!   1  → `<eos>`
//!   106 → `<end_of_turn>`

use anyhow::{anyhow, Result};
use ndarray::Array3;
use ort::session::Session;
use std::path::Path;

const NUM_LAYERS: usize = 30;
const ATTN_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
/// Size of the per-layer embedding produced by embed_tokens: [seq, 30, 256]
const PER_LAYER_N: usize = 30;
const PER_LAYER_DIM: usize = 256;

// ──────────────────────────────────────────────────────────────────────────
// KV cache state
// ──────────────────────────────────────────────────────────────────────────

struct DecodeState {
    /// key cache per layer: flattened [1, ATTN_HEADS, past_seq, HEAD_DIM]
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    total_seq_len: usize,
}

impl DecodeState {
    fn new() -> Self {
        Self {
            keys: vec![vec![]; NUM_LAYERS],
            values: vec![vec![]; NUM_LAYERS],
            total_seq_len: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Backend
// ──────────────────────────────────────────────────────────────────────────

pub struct GemmaBackend {
    embed_session: Session,
    decoder_session: Session,
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: u32,
    end_of_turn_id: u32,
}

impl GemmaBackend {
    pub fn new<P: AsRef<Path>>(model_path: P, intra_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        // Prefer the most compressed quantisation available
        let decoder_path = [
            "decoder_model_merged_q4.onnx",
            "decoder_model_merged_q4f16.onnx",
            "decoder_model_merged_fp16.onnx",
            "decoder_model_merged.onnx",
        ]
        .iter()
        .map(|f| path.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("No decoder ONNX file found in {:?}", path))?;

        let embed_path = [
            "embed_tokens_quantized.onnx",
            "embed_tokens_int8.onnx",
            "embed_tokens_uint8.onnx",
            "embed_tokens_fp16.onnx",
            "embed_tokens.onnx",
        ]
        .iter()
        .map(|f| path.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("No embed_tokens ONNX file found in {:?}", path))?;

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("tokenizer.json not found in {:?}", path));
        }

        log::info!(
            "Gemma model files: decoder={}, embed={}",
            decoder_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            embed_path.file_name().unwrap_or_default().to_string_lossy(),
        );

        log::info!("Loading Gemma tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        log::info!("Loading Gemma embed_tokens session");
        let embed_session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&embed_path)?;

        log::info!("Loading Gemma decoder session");
        let decoder_session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&decoder_path)?;

        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab.get("<eos>").copied().unwrap_or(1);
        let end_of_turn_id = vocab.get("<end_of_turn>").copied().unwrap_or(106);

        log::info!(
            "Gemma ready — {} layers, eos={} end_of_turn={}",
            NUM_LAYERS,
            eos_token_id,
            end_of_turn_id,
        );

        Ok(Self {
            embed_session,
            decoder_session,
            tokenizer,
            eos_token_id,
            end_of_turn_id,
        })
    }

    /// Process `text` through the LLM using `prompt_template` (`{text}` placeholder).
    pub fn process(&mut self, text: &str, prompt_template: &str) -> Result<String> {
        let prompt = prompt_template.replace("{text}", text);
        // Gemma-3n instruct chat template
        let formatted = format!(
            "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
            prompt
        );

        log::debug!(
            "Gemma prompt ({} chars): {}",
            formatted.len(),
            &formatted[..formatted.len().min(200)]
        );

        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        log::debug!("Gemma input token count: {}", prompt_ids.len());

        let output_ids = self.greedy_decode(&prompt_ids, 1024)?;
        log::debug!("Gemma generated {} new tokens", output_ids.len());

        let output_text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow!("Decode failed: {}", e))?;

        let result = output_text.trim().to_string();
        log::debug!("Gemma raw output: {:?}", result);
        Ok(result)
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /// Run embed_tokens.
    /// Returns (inputs_embeds [1, seq, 2048], per_layer_inputs [1, seq, 30, 256]).
    fn embed(&mut self, token_ids: &[u32]) -> Result<(Array3<f32>, Vec<f32>, usize)> {
        let seq = token_ids.len();
        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        let input_val: ort::value::Value =
            ort::value::Value::from_array(([1usize, seq], ids_i64))?.into();

        let outputs = self
            .embed_session
            .run(vec![("input_ids".to_string(), input_val)])
            .map_err(|e| anyhow!("embed_tokens failed: {}", e))?;

        // Output 0: inputs_embeds [1, seq, hidden]
        let (shape0, data0) = outputs[0].try_extract_tensor::<f32>()?;
        let d = shape0.as_ref();
        if d.len() != 3 {
            return Err(anyhow!("embed_tokens unexpected rank {}", d.len()));
        }
        let embeds = Array3::from_shape_vec(
            (d[0] as usize, d[1] as usize, d[2] as usize),
            data0.to_vec(),
        )?;

        // Output 1: per_layer_inputs [1, seq, n_layers, layer_dim]
        let (_, data1) = outputs[1].try_extract_tensor::<f32>()?;
        let per_layer: Vec<f32> = data1.to_vec();

        Ok((embeds, per_layer, seq))
    }

    fn greedy_decode(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut state = DecodeState::new();
        let mut generated: Vec<u32> = Vec::new();

        // Prefill
        let (embeds, per_layer, seq_len) = self.embed(prompt_ids)?;
        let next_token = self.decoder_step(&embeds, &per_layer, seq_len, &mut state)?;
        state.total_seq_len += seq_len;

        if self.is_stop_token(next_token) {
            return Ok(generated);
        }
        generated.push(next_token);

        // Autoregressive
        for _ in 1..max_new_tokens {
            let last = *generated.last().unwrap();
            let (step_embeds, step_per_layer, _) = self.embed(&[last])?;
            let next_token = self.decoder_step(&step_embeds, &step_per_layer, 1, &mut state)?;
            state.total_seq_len += 1;
            if self.is_stop_token(next_token) {
                break;
            }
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == self.eos_token_id || token == self.end_of_turn_id
    }

    fn decoder_step(
        &mut self,
        input_embeds: &Array3<f32>,
        per_layer_data: &[f32],
        cur_seq: usize,
        state: &mut DecodeState,
    ) -> Result<u32> {
        let batch = 1usize;
        let hidden = input_embeds.shape()[2];
        let embeds_data: Vec<f32> = input_embeds.iter().copied().collect();

        // position_ids: i64 [1, cur_seq]  — absolute positions
        let start = state.total_seq_len as i64;
        let position_ids: Vec<i64> = (start..start + cur_seq as i64).collect();

        let mut inputs: Vec<(String, ort::value::Value)> = Vec::new();

        inputs.push((
            "inputs_embeds".to_string(),
            ort::value::Value::from_array(([batch, cur_seq, hidden], embeds_data))?.into(),
        ));
        inputs.push((
            "position_ids".to_string(),
            ort::value::Value::from_array(([batch, cur_seq], position_ids))?.into(),
        ));
        inputs.push((
            "per_layer_inputs".to_string(),
            ort::value::Value::from_array((
                [batch, cur_seq, PER_LAYER_N, PER_LAYER_DIM],
                per_layer_data.to_vec(),
            ))?
            .into(),
        ));

        // KV cache inputs
        for layer in 0..NUM_LAYERS {
            let past_seq = if state.keys[layer].is_empty() {
                0
            } else {
                state.keys[layer].len() / (batch * ATTN_HEADS * HEAD_DIM)
            };
            // ort doesn't support 0-length dims — use 1 with zeros when empty
            let p = if past_seq == 0 { 1 } else { past_seq };

            let key_data = if state.keys[layer].is_empty() {
                vec![0.0f32; batch * ATTN_HEADS * p * HEAD_DIM]
            } else {
                state.keys[layer].clone()
            };
            let val_data = if state.values[layer].is_empty() {
                vec![0.0f32; batch * ATTN_HEADS * p * HEAD_DIM]
            } else {
                state.values[layer].clone()
            };

            inputs.push((
                format!("past_key_values.{}.key", layer),
                ort::value::Value::from_array(([batch, ATTN_HEADS, p, HEAD_DIM], key_data))?.into(),
            ));
            inputs.push((
                format!("past_key_values.{}.value", layer),
                ort::value::Value::from_array(([batch, ATTN_HEADS, p, HEAD_DIM], val_data))?.into(),
            ));
        }

        let outputs = self
            .decoder_session
            .run(inputs)
            .map_err(|e| anyhow!("Gemma decoder step failed: {}", e))?;

        // Extract next token from logits [1, seq, vocab]
        let (logits_shape, logits_data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims = logits_shape.as_ref();
        if dims.len() < 2 {
            return Err(anyhow!("Unexpected logits rank {}", dims.len()));
        }
        let vocab_size = *dims.last().unwrap() as usize;
        let seq_out = if dims.len() == 3 { dims[1] as usize } else { 1 };
        let last_offset = (seq_out - 1) * vocab_size;
        let last_logits = &logits_data[last_offset..last_offset + vocab_size];

        let next_token = last_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(self.eos_token_id);

        // Update KV cache from present.N.key / present.N.value outputs
        // Output layout: [logits, present.0.key, present.0.value, present.1.key, ...]
        let mut out_idx = 1usize;
        for layer in 0..NUM_LAYERS {
            if out_idx >= outputs.len() {
                break;
            }
            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                state.keys[layer] = data.to_vec();
            }
            out_idx += 1;
            if out_idx < outputs.len() {
                if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                    state.values[layer] = data.to_vec();
                }
                out_idx += 1;
            }
        }

        Ok(next_token)
    }
}
