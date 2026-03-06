//! LLM post-processing backend for Vox.
//!
//! Supports the Qwen3.5 ONNX family (0.8B, 2B, 4B, …).  Architecture
//! constants (layer count, cache tensor shapes, which layers use standard
//! attention vs. Gated DeltaNet) are derived at load time by inspecting the
//! ONNX session inputs, so the same code works across model sizes.
//!
//! # Hybrid architecture recap
//!
//! Each Qwen3.5 model mixes:
//!   - **Gated DeltaNet** layers (linear attention) — state kept as
//!     `past_conv.N`      f32 [batch, conv_dim, 4]
//!     `past_recurrent.N` f32 [batch, rec_heads, rec_dim, rec_dim]
//!   - **Gated Attention** layers (standard softmax) — state kept as
//!     `past_key_values.N.key/value` f32 [batch, attn_heads, past_seq, head_dim]
//!
//! Additional inputs every step:
//!   `inputs_embeds`  f32 [1, cur_seq, hidden]
//!   `attention_mask` i64 [1, total_seq]
//!   `position_ids`   i64 [3, 1, cur_seq]   ← rank-3, RoPE phases

use anyhow::{anyhow, Result};
use half::f16;
use ndarray::Array3;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use std::path::Path;

// ──────────────────────────────────────────────────────────────────────────
// Architecture description (derived at load time)
// ──────────────────────────────────────────────────────────────────────────

enum KvDtype {
    F32,
    F16,
}

/// All architecture parameters derived from ONNX session inputs.
struct Arch {
    num_layers: usize,
    /// Sorted set of layer indices that use standard attention (KV cache).
    attn_layers: Vec<usize>,
    conv_dim0: usize,     // e.g. 6144 (0.8B), 8192 (4B)
    conv_dim1: usize,     // always 4
    rec_heads: usize,     // e.g. 16 (0.8B), 32 (4B)
    rec_dim: usize,       // always 128
    attn_heads: usize,    // e.g. 2 (0.8B), 4 (4B)
    attn_head_dim: usize, // always 256
    kv_dtype: KvDtype,
}

impl Arch {
    /// Inspect session inputs and derive architecture parameters.
    fn from_session(session: &Session) -> Result<Self> {
        let mut attn_layers: Vec<usize> = Vec::new();
        let mut conv_layers: Vec<usize> = Vec::new();
        let mut conv_dim0 = 0usize;
        let mut conv_dim1 = 4usize;
        let mut rec_heads = 0usize;
        let mut rec_dim = 128usize;
        let mut attn_heads = 0usize;
        let mut attn_head_dim = 256usize;

        let mut kv_dtype = KvDtype::F32;

        for inlet in session.inputs() {
            let name = inlet.name();
            if name.starts_with("past_key_values.") && name.ends_with(".key") {
                let layer: usize = name
                    .split('.')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| anyhow!("Cannot parse layer from '{}'", name))?;
                attn_layers.push(layer);

                // Read attn_heads, attn_head_dim and kv_dtype from shape on first encounter
                if attn_heads == 0 {
                    if let ValueType::Tensor { shape, ty, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // shape: [batch, heads, past_seq, head_dim]
                        if dims.len() == 4 && dims[1] > 0 && dims[3] > 0 {
                            attn_heads = dims[1] as usize;
                            attn_head_dim = dims[3] as usize;
                        }
                        if matches!(ty, TensorElementType::Float16) {
                            kv_dtype = KvDtype::F16;
                        }
                    }
                }
            } else if name.starts_with("past_conv.") {
                let layer: usize = name["past_conv.".len()..]
                    .parse()
                    .map_err(|_| anyhow!("Cannot parse layer from '{}'", name))?;
                conv_layers.push(layer);

                // Read conv shape on first encounter
                if conv_dim0 == 0 {
                    if let ValueType::Tensor { shape, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // shape: [batch, conv_dim0, conv_dim1]
                        if dims.len() == 3 && dims[1] > 0 && dims[2] > 0 {
                            conv_dim0 = dims[1] as usize;
                            conv_dim1 = dims[2] as usize;
                        }
                    }
                }
            } else if name.starts_with("past_recurrent.") {
                // Read recurrent shape on first encounter
                if rec_heads == 0 {
                    if let ValueType::Tensor { shape, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // shape: [batch, rec_heads, rec_dim, rec_dim]
                        if dims.len() == 4 && dims[1] > 0 && dims[2] > 0 {
                            rec_heads = dims[1] as usize;
                            rec_dim = dims[2] as usize;
                        }
                    }
                }
            }
        }

        attn_layers.sort_unstable();
        conv_layers.sort_unstable();

        let num_layers = attn_layers.len() + conv_layers.len();
        if num_layers == 0 {
            return Err(anyhow!(
                "No past_key_values or past_conv inputs found — is this a Qwen3.5 ONNX model?"
            ));
        }

        // Fallbacks for fully-dynamic shapes (all dims == -1 / 0 in exported ONNX)
        if attn_heads == 0 {
            attn_heads = 2;
        }
        if conv_dim0 == 0 {
            conv_dim0 = 6144;
        }
        if rec_heads == 0 {
            rec_heads = 16;
        }

        log::info!(
            "LLM arch: {} layers ({} attn, {} deltanet), \
             conv=[{}, {}], rec=[{}, {}, {}], attn=[heads={}, dim={}], kv_dtype={}",
            num_layers,
            attn_layers.len(),
            conv_layers.len(),
            conv_dim0,
            conv_dim1,
            rec_heads,
            rec_dim,
            rec_dim,
            attn_heads,
            attn_head_dim,
            match kv_dtype {
                KvDtype::F32 => "f32",
                KvDtype::F16 => "f16",
            },
        );

        Ok(Self {
            num_layers,
            attn_layers,
            conv_dim0,
            conv_dim1,
            rec_heads,
            rec_dim,
            attn_heads,
            attn_head_dim,
            kv_dtype,
        })
    }

    fn is_attn_layer(&self, layer: usize) -> bool {
        self.attn_layers.binary_search(&layer).is_ok()
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Decode state
// ──────────────────────────────────────────────────────────────────────────

enum AttnCache {
    F32(Vec<Vec<f32>>, Vec<Vec<f32>>),
    F16(Vec<Vec<f16>>, Vec<Vec<f16>>),
}

enum RecurrentCache {
    F32 {
        conv: Vec<Vec<f32>>,
        recurrent: Vec<Vec<f32>>,
    },
    F16 {
        conv: Vec<Vec<f16>>,
        recurrent: Vec<Vec<f16>>,
    },
}

struct DecodeState {
    recurrent: RecurrentCache,
    attn: AttnCache,
    total_seq_len: usize,
}

impl DecodeState {
    fn new(arch: &Arch) -> Self {
        let (attn, recurrent) = match arch.kv_dtype {
            KvDtype::F32 => {
                let zero_conv = vec![0.0f32; arch.conv_dim0 * arch.conv_dim1];
                let zero_rec = vec![0.0f32; arch.rec_heads * arch.rec_dim * arch.rec_dim];
                (
                    AttnCache::F32(vec![vec![]; arch.num_layers], vec![vec![]; arch.num_layers]),
                    RecurrentCache::F32 {
                        conv: vec![zero_conv; arch.num_layers],
                        recurrent: vec![zero_rec; arch.num_layers],
                    },
                )
            }
            KvDtype::F16 => {
                let zero_conv = vec![f16::ZERO; arch.conv_dim0 * arch.conv_dim1];
                let zero_rec = vec![f16::ZERO; arch.rec_heads * arch.rec_dim * arch.rec_dim];
                (
                    AttnCache::F16(vec![vec![]; arch.num_layers], vec![vec![]; arch.num_layers]),
                    RecurrentCache::F16 {
                        conv: vec![zero_conv; arch.num_layers],
                        recurrent: vec![zero_rec; arch.num_layers],
                    },
                )
            }
        };
        Self {
            recurrent,
            attn,
            total_seq_len: 0,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Backend
// ──────────────────────────────────────────────────────────────────────────

pub struct LlmBackend {
    embed_session: Session,
    decoder_session: Session,
    tokenizer: tokenizers::Tokenizer,
    arch: Arch,
    eos_token_id: u32,
    im_end_token_id: u32,
}

impl LlmBackend {
    pub fn new<P: AsRef<Path>>(model_path: P, intra_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        // Prefer the most compressed quantisation available, falling back to
        // less compressed variants.  This covers both 0.8B (which ships _q4
        // files) and 4B (same) without hardcoding the suffix.
        let decoder_path = [
            "decoder_model_merged_q4.onnx",
            "decoder_model_merged_quantized.onnx",
            "decoder_model_merged_fp16.onnx",
            "decoder_model_merged.onnx",
        ]
        .iter()
        .map(|f| path.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("No decoder ONNX file found in {:?}", path))?;

        let embed_path = [
            "embed_tokens_q4.onnx",
            "embed_tokens_quantized.onnx",
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
            "LLM model files: decoder={}, embed={}",
            decoder_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy(),
            embed_path.file_name().unwrap_or_default().to_string_lossy(),
        );

        log::info!("Loading LLM tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        log::info!("Loading LLM embed_tokens session");
        let embed_session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&embed_path)?;

        log::info!("Loading LLM decoder session");
        let decoder_session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&decoder_path)?;

        let arch = Arch::from_session(&decoder_session)?;

        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab.get("<|endoftext|>").copied().unwrap_or(151645);
        let im_end_token_id = vocab.get("<|im_end|>").copied().unwrap_or(151643);

        log::info!(
            "LLM ready — eos={} im_end={}",
            eos_token_id,
            im_end_token_id
        );

        Ok(Self {
            embed_session,
            decoder_session,
            tokenizer,
            arch,
            eos_token_id,
            im_end_token_id,
        })
    }

    /// Process `text` through the LLM using `prompt_template` (`{text}` placeholder).
    pub fn process(&mut self, text: &str, prompt_template: &str) -> Result<String> {
        let prompt = prompt_template.replace("{text}", text);
        // Pre-filled empty <think></think> = non-thinking mode per the chat template.
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        log::debug!("LLM prompt ({} chars): {:?}", formatted.len(), formatted);
        log::debug!(
            "LLM prompt (first 200 chars): {}",
            &formatted[..formatted.len().min(200)]
        );

        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        log::debug!("LLM input token count: {}", prompt_ids.len());

        let output_ids = self.greedy_decode(&prompt_ids, 1024)?;
        log::debug!("LLM generated {} new tokens", output_ids.len());

        let output_text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow!("Decode failed: {}", e))?;

        let result = output_text.trim().to_string();
        log::debug!("LLM raw output: {:?}", result);
        Ok(result)
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn embed(&mut self, token_ids: &[u32]) -> Result<Array3<f32>> {
        let seq = token_ids.len();
        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();
        let input_val: ort::value::Value =
            ort::value::Value::from_array(([1usize, seq], ids_i64))?.into();
        let outputs = self
            .embed_session
            .run(vec![("input_ids".to_string(), input_val)])
            .map_err(|e| anyhow!("embed_tokens failed: {}", e))?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let d = shape.as_ref();
        if d.len() != 3 {
            return Err(anyhow!("embed_tokens unexpected rank {}", d.len()));
        }
        Ok(Array3::from_shape_vec(
            (d[0] as usize, d[1] as usize, d[2] as usize),
            data.to_vec(),
        )?)
    }

    fn greedy_decode(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut state = DecodeState::new(&self.arch);
        let mut generated: Vec<u32> = Vec::new();

        // Prefill: process the full prompt in one forward pass
        let prompt_embeds = self.embed(prompt_ids)?;
        let next_token = self.decoder_step(&prompt_embeds, prompt_ids.len(), &mut state)?;
        state.total_seq_len += prompt_ids.len();
        log::debug!("LLM first token: id={}", next_token);

        if self.is_stop_token(next_token) {
            return Ok(generated);
        }
        generated.push(next_token);

        // Autoregressive: one token at a time
        for step in 1..max_new_tokens {
            let last = *generated.last().unwrap();
            let step_embeds = self.embed(&[last])?;
            let next_token = self.decoder_step(&step_embeds, 1, &mut state)?;
            state.total_seq_len += 1;
            log::debug!("LLM token[{}]: id={}", step, next_token);
            if self.is_stop_token(next_token) {
                break;
            }
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn is_stop_token(&self, token: u32) -> bool {
        token == self.eos_token_id || token == self.im_end_token_id
    }

    fn decoder_step(
        &mut self,
        input_embeds: &Array3<f32>,
        cur_seq: usize,
        state: &mut DecodeState,
    ) -> Result<u32> {
        let batch = 1usize;
        let total_seq = state.total_seq_len + cur_seq;
        let hidden_dim = input_embeds.shape()[2];
        let arch = &self.arch; // borrow before mutable self below

        let embeds_data: Vec<f32> = input_embeds.iter().copied().collect();
        let attn_mask: Vec<i64> = vec![1i64; total_seq];

        let start_pos = state.total_seq_len as i64;
        let pos_row0: Vec<i64> = (start_pos..start_pos + cur_seq as i64).collect();
        let pos_zeros: Vec<i64> = vec![0i64; cur_seq];
        let mut position_ids_data = Vec::with_capacity(3 * cur_seq);
        position_ids_data.extend_from_slice(&pos_row0);
        position_ids_data.extend_from_slice(&pos_zeros);
        position_ids_data.extend_from_slice(&pos_zeros);

        let attn_heads = arch.attn_heads;
        let attn_head_dim = arch.attn_head_dim;
        let conv_dim0 = arch.conv_dim0;
        let conv_dim1 = arch.conv_dim1;
        let rec_heads = arch.rec_heads;
        let rec_dim = arch.rec_dim;
        let num_layers = arch.num_layers;
        let attn_layers = arch.attn_layers.clone();

        let mut inputs: Vec<(String, ort::value::Value)> = Vec::new();

        inputs.push((
            "inputs_embeds".to_string(),
            ort::value::Value::from_array(([batch, cur_seq, hidden_dim], embeds_data))?.into(),
        ));
        inputs.push((
            "attention_mask".to_string(),
            ort::value::Value::from_array(([batch, total_seq], attn_mask))?.into(),
        ));
        inputs.push((
            "position_ids".to_string(),
            ort::value::Value::from_array(([3usize, batch, cur_seq], position_ids_data))?.into(),
        ));

        for layer in 0..num_layers {
            if attn_layers.binary_search(&layer).is_ok() {
                match &state.attn {
                    AttnCache::F32(ref keys, ref vals) => {
                        let past_seq = if keys[layer].is_empty() {
                            0
                        } else {
                            keys[layer].len() / (batch * attn_heads * attn_head_dim)
                        };
                        let p = if past_seq == 0 { 1 } else { past_seq };
                        let key_data = if keys[layer].is_empty() {
                            vec![0.0f32; batch * attn_heads * p * attn_head_dim]
                        } else {
                            keys[layer].clone()
                        };
                        let val_data = if vals[layer].is_empty() {
                            vec![0.0f32; batch * attn_heads * p * attn_head_dim]
                        } else {
                            vals[layer].clone()
                        };
                        inputs.push((
                            format!("past_key_values.{}.key", layer),
                            ort::value::Value::from_array((
                                [batch, attn_heads, p, attn_head_dim],
                                key_data,
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_key_values.{}.value", layer),
                            ort::value::Value::from_array((
                                [batch, attn_heads, p, attn_head_dim],
                                val_data,
                            ))?
                            .into(),
                        ));
                    }
                    AttnCache::F16(ref keys, ref vals) => {
                        let past_seq = if keys[layer].is_empty() {
                            0
                        } else {
                            keys[layer].len() / (batch * attn_heads * attn_head_dim)
                        };
                        let p = if past_seq == 0 { 1 } else { past_seq };
                        let key_data = if keys[layer].is_empty() {
                            vec![f16::ZERO; batch * attn_heads * p * attn_head_dim]
                        } else {
                            keys[layer].clone()
                        };
                        let val_data = if vals[layer].is_empty() {
                            vec![f16::ZERO; batch * attn_heads * p * attn_head_dim]
                        } else {
                            vals[layer].clone()
                        };
                        inputs.push((
                            format!("past_key_values.{}.key", layer),
                            ort::value::Value::from_array((
                                [batch, attn_heads, p, attn_head_dim],
                                key_data,
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_key_values.{}.value", layer),
                            ort::value::Value::from_array((
                                [batch, attn_heads, p, attn_head_dim],
                                val_data,
                            ))?
                            .into(),
                        ));
                    }
                }
            } else {
                match &state.recurrent {
                    RecurrentCache::F32 { conv, recurrent } => {
                        inputs.push((
                            format!("past_conv.{}", layer),
                            ort::value::Value::from_array((
                                [batch, conv_dim0, conv_dim1],
                                conv[layer].clone(),
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_recurrent.{}", layer),
                            ort::value::Value::from_array((
                                [batch, rec_heads, rec_dim, rec_dim],
                                recurrent[layer].clone(),
                            ))?
                            .into(),
                        ));
                    }
                    RecurrentCache::F16 { conv, recurrent } => {
                        inputs.push((
                            format!("past_conv.{}", layer),
                            ort::value::Value::from_array((
                                [batch, conv_dim0, conv_dim1],
                                conv[layer].clone(),
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_recurrent.{}", layer),
                            ort::value::Value::from_array((
                                [batch, rec_heads, rec_dim, rec_dim],
                                recurrent[layer].clone(),
                            ))?
                            .into(),
                        ));
                    }
                }
            }
        }

        let outputs = self
            .decoder_session
            .run(inputs)
            .map_err(|e| anyhow!("Decoder step failed: {}", e))?;

        // Extract next token from logits (output 0) — may be f32 or f16
        let next_token = {
            let (logits_shape, logits_f32, logits_f16) =
                if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
                    (shape, Some(data.to_vec()), None)
                } else {
                    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
                    (
                        shape,
                        None,
                        Some(data.iter().map(|x| x.to_f32()).collect::<Vec<f32>>()),
                    )
                };
            let dims = logits_shape.as_ref();
            if dims.len() < 2 {
                return Err(anyhow!("Unexpected logits rank {}", dims.len()));
            }
            let vocab_size = *dims.last().unwrap() as usize;
            let seq_out = if dims.len() == 3 { dims[1] as usize } else { 1 };
            let last_offset = (seq_out - 1) * vocab_size;
            let last_logits_slice = logits_f32.as_deref().or(logits_f16.as_deref()).unwrap();
            let last_logits = &last_logits_slice[last_offset..last_offset + vocab_size];
            last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eos_token_id)
        };

        // Update state from present_* outputs
        // Layout: logits, then per-layer in order:
        //   attn layer  → present.N.key, present.N.value
        //   deltanet    → present_conv.N, present_recurrent.N
        let mut out_idx = 1usize;
        for layer in 0..num_layers {
            if out_idx >= outputs.len() {
                break;
            }
            if attn_layers.binary_search(&layer).is_ok() {
                match &mut state.attn {
                    AttnCache::F32(ref mut keys, ref mut vals) => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                            keys[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                                vals[layer] = data.to_vec();
                            }
                            out_idx += 1;
                        }
                    }
                    AttnCache::F16(ref mut keys, ref mut vals) => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                            keys[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                                vals[layer] = data.to_vec();
                            }
                            out_idx += 1;
                        }
                    }
                }
            } else {
                match &mut state.recurrent {
                    RecurrentCache::F32 {
                        ref mut conv,
                        ref mut recurrent,
                    } => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                            conv[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                                recurrent[layer] = data.to_vec();
                            }
                            out_idx += 1;
                        }
                    }
                    RecurrentCache::F16 {
                        ref mut conv,
                        ref mut recurrent,
                    } => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                            conv[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                                recurrent[layer] = data.to_vec();
                            }
                            out_idx += 1;
                        }
                    }
                }
            }
        }

        Ok(next_token)
    }
}
