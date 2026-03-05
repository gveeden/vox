//! LLM backend for the Qwen3 ONNX family (onnx-community/Qwen3-*-ONNX).
//!
//! Qwen3 uses a **standard transformer** architecture — no DeltaNet / hybrid
//! layers.  The ONNX export is a single merged file that accepts `input_ids`
//! directly (embedding is baked in), unlike the Qwen3.5 export which has a
//! separate `embed_tokens` session.
//!
//! # ONNX interface
//!
//! Inputs every step:
//!   `input_ids`       i64 [1, cur_seq]
//!   `attention_mask`  i64 [1, total_seq]
//!   `position_ids`    i64 [1, cur_seq]   ← rank-2 (not rank-3 like Qwen3.5)
//!   `past_key_values.N.key`   f32 [1, kv_heads, past_seq, head_dim]  for N in 0..num_layers
//!   `past_key_values.N.value` f32 [1, kv_heads, past_seq, head_dim]
//!
//! Output 0: logits  f32 [1, cur_seq, vocab_size]
//! Outputs 1..: present_key_values.N.key / .value  (same shape as above + cur_seq)
//!
//! Architecture constants (kv_heads, head_dim, num_layers) are derived at
//! load time by inspecting ONNX session inputs, so the same code handles
//! 0.6B, 1.7B, 4B, etc. without hardcoding.

use anyhow::{anyhow, Result};
use half::f16;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Architecture (derived from session inputs at load time)
// ─────────────────────────────────────────────────────────────────────────────

struct Arch {
    num_layers: usize,
    kv_heads: usize,
    head_dim: usize,
    kv_dtype: KvDtype,
}

enum KvDtype {
    F32,
    F16,
}

impl Arch {
    fn from_session(session: &Session) -> Result<Self> {
        let mut num_layers = 0usize;
        let mut kv_heads = 0usize;
        let mut head_dim = 0usize;
        let mut kv_dtype = KvDtype::F32;

        for inlet in session.inputs() {
            let name = inlet.name();
            if name.starts_with("past_key_values.") && name.ends_with(".key") {
                let layer: usize = name
                    .split('.')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| anyhow!("Cannot parse layer index from '{}'", name))?;
                if layer + 1 > num_layers {
                    num_layers = layer + 1;
                }

                if kv_heads == 0 {
                    if let ValueType::Tensor { shape, ty, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        if dims.len() == 4 {
                            let h = dims[1].unsigned_abs() as usize;
                            let d = dims[3].unsigned_abs() as usize;
                            if h > 0 {
                                kv_heads = h;
                            }
                            if d > 0 {
                                head_dim = d;
                            }
                        }
                        if matches!(ty, TensorElementType::Float16) {
                            kv_dtype = KvDtype::F16;
                        }
                    }
                }
            }
        }

        if num_layers == 0 {
            return Err(anyhow!(
                "No past_key_values inputs found — is this a Qwen3 ONNX model?"
            ));
        }

        if kv_heads == 0 {
            kv_heads = 8;
        }
        if head_dim == 0 {
            head_dim = 128;
        }

        log::info!(
            "Qwen3 arch: {} layers, {} kv_heads, head_dim={}, kv_dtype={}",
            num_layers,
            kv_heads,
            head_dim,
            match kv_dtype {
                KvDtype::F32 => "f32",
                KvDtype::F16 => "f16",
            },
        );

        Ok(Self {
            num_layers,
            kv_heads,
            head_dim,
            kv_dtype,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KV-cache state
// ─────────────────────────────────────────────────────────────────────────────

enum KvData {
    F32(Vec<Vec<f32>>, Vec<Vec<f32>>),
    F16(Vec<Vec<f16>>, Vec<Vec<f16>>),
}

struct KvState {
    data: KvData,
    total_seq_len: usize,
}

impl KvState {
    fn new(num_layers: usize, dtype: &KvDtype) -> Self {
        match dtype {
            KvDtype::F32 => Self {
                data: KvData::F32(vec![vec![]; num_layers], vec![vec![]; num_layers]),
                total_seq_len: 0,
            },
            KvDtype::F16 => Self {
                data: KvData::F16(vec![vec![]; num_layers], vec![vec![]; num_layers]),
                total_seq_len: 0,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

pub struct Qwen3Backend {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    arch: Arch,
    eos_token_id: u32,
    im_end_token_id: u32,
}

impl Qwen3Backend {
    pub fn new<P: AsRef<Path>>(model_path: P, intra_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        // Prefer the most compressed available quantisation
        let model_file = [
            "model_q4.onnx",
            "model_q4f16.onnx",
            "model_int8.onnx",
            "model_quantized.onnx",
            "model_uint8.onnx",
            "model_bnb4.onnx",
            "model_fp16.onnx",
            "model.onnx",
        ]
        .iter()
        .map(|f| path.join(f))
        .find(|p| p.exists())
        .ok_or_else(|| anyhow!("No model ONNX file found in {:?}", path))?;

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("tokenizer.json not found in {:?}", path));
        }

        log::info!(
            "Qwen3 model file: {}",
            model_file.file_name().unwrap_or_default().to_string_lossy()
        );

        log::info!("Loading Qwen3 tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        log::info!("Loading Qwen3 session");
        let session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&model_file)?;

        let arch = Arch::from_session(&session)?;

        let vocab = tokenizer.get_vocab(true);
        // <|im_end|> = 151645, <|endoftext|> = 151643 — both are stop tokens
        let im_end_token_id = vocab.get("<|im_end|>").copied().unwrap_or(151645);
        let eos_token_id = vocab.get("<|endoftext|>").copied().unwrap_or(151643);

        log::info!(
            "Qwen3 ready — eos={} im_end={}",
            eos_token_id,
            im_end_token_id
        );

        Ok(Self {
            session,
            tokenizer,
            arch,
            eos_token_id,
            im_end_token_id,
        })
    }

    /// Apply the LLM to `text` using `prompt_template` (with `{text}` placeholder).
    pub fn process(&mut self, text: &str, prompt_template: &str) -> Result<String> {
        let prompt = prompt_template.replace("{text}", text);
        // Pre-fill empty <think></think> block to suppress chain-of-thought
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt
        );

        log::debug!(
            "Qwen3 prompt ({} chars): {}",
            formatted.len(),
            &formatted[..formatted.len().min(200)]
        );

        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        log::debug!("Qwen3 input tokens: {}", prompt_ids.len());

        let output_ids = self.greedy_decode(&prompt_ids, 1024)?;
        log::debug!("Qwen3 generated {} tokens", output_ids.len());

        let output_text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow!("Decode failed: {}", e))?;

        Ok(output_text.trim().to_string())
    }

    // ── Private helpers ───────────────────────────────────────────────────

    fn is_stop_token(&self, token: u32) -> bool {
        token == self.eos_token_id || token == self.im_end_token_id
    }

    fn greedy_decode(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut state = KvState::new(self.arch.num_layers, &self.arch.kv_dtype);
        let mut generated: Vec<u32> = Vec::new();

        // Prefill
        let next = self.forward_step(prompt_ids, &mut state)?;
        state.total_seq_len += prompt_ids.len();

        if self.is_stop_token(next) {
            return Ok(generated);
        }
        generated.push(next);

        // Autoregressive decode
        for step in 1..max_new_tokens {
            let last = *generated.last().unwrap();
            let next = self.forward_step(&[last], &mut state)?;
            state.total_seq_len += 1;
            log::debug!("Qwen3 token[{}]: {}", step, next);
            if self.is_stop_token(next) {
                break;
            }
            generated.push(next);
        }

        Ok(generated)
    }

    fn forward_step(&mut self, token_ids: &[u32], state: &mut KvState) -> Result<u32> {
        let batch = 1usize;
        let cur_seq = token_ids.len();
        let past_seq = state.total_seq_len;
        let total_seq = past_seq + cur_seq;

        let kv_heads = self.arch.kv_heads;
        let head_dim = self.arch.head_dim;
        let num_layers = self.arch.num_layers;

        // input_ids  i64 [1, cur_seq]
        let ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();

        // attention_mask  i64 [1, total_seq]  (all ones)
        let attn_mask: Vec<i64> = vec![1i64; total_seq];

        // position_ids  i64 [1, cur_seq]  — rank-2 for Qwen3 (unlike rank-3 in Qwen3.5)
        let start = past_seq as i64;
        let position_ids: Vec<i64> = (start..start + cur_seq as i64).collect();

        let mut inputs: Vec<(String, ort::value::Value)> = Vec::new();

        inputs.push((
            "input_ids".to_string(),
            ort::value::Value::from_array(([batch, cur_seq], ids_i64))?.into(),
        ));
        inputs.push((
            "attention_mask".to_string(),
            ort::value::Value::from_array(([batch, total_seq], attn_mask))?.into(),
        ));
        inputs.push((
            "position_ids".to_string(),
            ort::value::Value::from_array(([batch, cur_seq], position_ids))?.into(),
        ));

        for layer in 0..num_layers {
            let p = if past_seq == 0 { 1 } else { past_seq };
            let size = batch * kv_heads * p * head_dim;

            match &mut state.data {
                KvData::F32(ref mut key_cache, ref mut val_cache) => {
                    let key_data = if key_cache[layer].is_empty() {
                        vec![0.0f32; size]
                    } else {
                        key_cache[layer].clone()
                    };
                    let val_data = if val_cache[layer].is_empty() {
                        vec![0.0f32; size]
                    } else {
                        val_cache[layer].clone()
                    };

                    inputs.push((
                        format!("past_key_values.{}.key", layer),
                        ort::value::Value::from_array(([batch, kv_heads, p, head_dim], key_data))?
                            .into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{}.value", layer),
                        ort::value::Value::from_array(([batch, kv_heads, p, head_dim], val_data))?
                            .into(),
                    ));
                }
                KvData::F16(ref mut key_cache, ref mut val_cache) => {
                    let key_data = if key_cache[layer].is_empty() {
                        vec![f16::ZERO; size]
                    } else {
                        key_cache[layer].clone()
                    };
                    let val_data = if val_cache[layer].is_empty() {
                        vec![f16::ZERO; size]
                    } else {
                        val_cache[layer].clone()
                    };

                    inputs.push((
                        format!("past_key_values.{}.key", layer),
                        ort::value::Value::from_array(([batch, kv_heads, p, head_dim], key_data))?
                            .into(),
                    ));
                    inputs.push((
                        format!("past_key_values.{}.value", layer),
                        ort::value::Value::from_array(([batch, kv_heads, p, head_dim], val_data))?
                            .into(),
                    ));
                }
            }
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow!("Qwen3 forward step failed: {}", e))?;

        let (logits_shape, logits_data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims = logits_shape.as_ref();
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

        let mut out_idx = 1usize;
        for layer in 0..num_layers {
            if out_idx >= outputs.len() {
                break;
            }
            match &mut state.data {
                KvData::F32(ref mut key_cache, ref mut _val_cache) => {
                    if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                        key_cache[layer] = data.to_vec();
                    }
                }
                KvData::F16(ref mut key_cache, ref mut _val_cache) => {
                    if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                        key_cache[layer] = data.to_vec();
                    }
                }
            }
            out_idx += 1;
            if out_idx >= outputs.len() {
                break;
            }
            match &mut state.data {
                KvData::F32(ref mut _key_cache, ref mut val_cache) => {
                    if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                        val_cache[layer] = data.to_vec();
                    }
                }
                KvData::F16(ref mut _key_cache, ref mut val_cache) => {
                    if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                        val_cache[layer] = data.to_vec();
                    }
                }
            }
            out_idx += 1;
        }

        Ok(next_token)
    }
}
