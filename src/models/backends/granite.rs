//! Granite 4.0 Hybrid ONNX backend for Vox.
//!
//! Supports the onnx-community/granite-4.0-h-1b-ONNX model family.
//! Architecture: hybrid Transformer (attention) + Mamba (SSM) layers.
//!
//! Cache layout per layer:
//!   Attention: `past_key_values.{i}.key/value`  [batch, kv_heads, past_seq, head_dim]
//!   Mamba:     `past_conv.{i}`                   [batch, conv_d_inner, d_conv]
//!              `past_ssm.{i}`                    [batch, mamba_n_heads, mamba_d_head, mamba_d_state]

use anyhow::{anyhow, Result};
use half::f16;
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::ValueType;
use std::path::Path;

// ─────────────────────────────────────────────────────────────────────────────
// Architecture description (derived at load time from session inputs)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum KvDtype {
    F32,
    F16,
}

struct Arch {
    num_layers: usize,
    /// Sorted layer indices that use attention KV cache.
    attn_layers: Vec<usize>,
    // Attention dims
    attn_heads: usize,
    attn_head_dim: usize,
    // Mamba conv dims: [batch, conv_d_inner, d_conv]
    conv_d_inner: usize,
    d_conv: usize,
    // Mamba SSM dims: [batch, mamba_n_heads, mamba_d_head, mamba_d_state]
    mamba_n_heads: usize,
    mamba_d_head: usize,
    mamba_d_state: usize,
    /// Dtype for attention KV cache.
    kv_dtype: KvDtype,
    /// Dtype for Mamba conv/ssm states.
    mamba_dtype: KvDtype,
    has_position_ids: bool,
    has_num_logits_to_keep: bool,
}

impl Arch {
    fn from_session(session: &Session) -> Result<Self> {
        let mut attn_layers: Vec<usize> = Vec::new();
        let mut mamba_layers: Vec<usize> = Vec::new();
        let mut attn_heads = 0usize;
        let mut attn_head_dim = 0usize;
        let mut conv_d_inner = 0usize;
        let mut d_conv = 4usize;
        let mut mamba_n_heads = 0usize;
        let mut mamba_d_head = 0usize;
        let mut mamba_d_state = 0usize;
        let mut kv_dtype = KvDtype::F32;
        let mut mamba_dtype = KvDtype::F32;
        let mut has_position_ids = false;
        let mut has_num_logits_to_keep = false;

        for inlet in session.inputs() {
            let name = inlet.name();
            if name == "position_ids" {
                has_position_ids = true;
            } else if name == "num_logits_to_keep" {
                has_num_logits_to_keep = true;
            } else if name.starts_with("past_key_values.") && name.ends_with(".key") {
                let layer: usize = name
                    .split('.')
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| anyhow!("Cannot parse layer from '{}'", name))?;
                attn_layers.push(layer);
                if attn_heads == 0 {
                    if let ValueType::Tensor { shape, ty, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // [batch, kv_heads, past_seq, head_dim]
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
                mamba_layers.push(layer);
                if conv_d_inner == 0 {
                    if let ValueType::Tensor { shape, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // [batch, conv_d_inner, d_conv]
                        if dims.len() == 3 && dims[1] > 0 && dims[2] > 0 {
                            conv_d_inner = dims[1] as usize;
                            d_conv = dims[2] as usize;
                        }
                    }
                }
            } else if name.starts_with("past_ssm.") {
                if mamba_n_heads == 0 {
                    if let ValueType::Tensor { shape, ty, .. } = inlet.dtype() {
                        let dims = shape.as_ref();
                        // [batch, mamba_n_heads, mamba_d_head, mamba_d_state]
                        if dims.len() == 4 && dims[1] > 0 && dims[2] > 0 && dims[3] > 0 {
                            mamba_n_heads = dims[1] as usize;
                            mamba_d_head = dims[2] as usize;
                            mamba_d_state = dims[3] as usize;
                        }
                        if matches!(ty, TensorElementType::Float16) {
                            mamba_dtype = KvDtype::F16;
                        }
                    }
                }
            }
        }

        attn_layers.sort_unstable();
        attn_layers.dedup();
        mamba_layers.sort_unstable();
        mamba_layers.dedup();

        let num_layers = attn_layers.len() + mamba_layers.len();
        if num_layers == 0 {
            return Err(anyhow!(
                "No cache inputs found — is this a Granite 4.0 ONNX model?"
            ));
        }

        // Fallbacks for fully-dynamic shapes (dims == -1 in exported ONNX)
        if attn_heads == 0 {
            attn_heads = 4;
        }
        if attn_head_dim == 0 {
            attn_head_dim = 64;
        }
        if conv_d_inner == 0 {
            conv_d_inner = 2048;
        }
        if mamba_n_heads == 0 {
            mamba_n_heads = 4;
        }
        if mamba_d_head == 0 {
            mamba_d_head = 64;
        }
        if mamba_d_state == 0 {
            mamba_d_state = 128;
        }

        log::info!(
            "Granite arch: {} layers ({} attn, {} mamba), \
             attn=[heads={}, dim={}], conv=[{}, {}], ssm=[{}, {}, {}], \
             kv_dtype={}, mamba_dtype={}",
            num_layers,
            attn_layers.len(),
            mamba_layers.len(),
            attn_heads,
            attn_head_dim,
            conv_d_inner,
            d_conv,
            mamba_n_heads,
            mamba_d_head,
            mamba_d_state,
            match kv_dtype { KvDtype::F32 => "f32", KvDtype::F16 => "f16" },
            match mamba_dtype { KvDtype::F32 => "f32", KvDtype::F16 => "f16" },
        );

        Ok(Self {
            num_layers,
            attn_layers,
            attn_heads,
            attn_head_dim,
            conv_d_inner,
            d_conv,
            mamba_n_heads,
            mamba_d_head,
            mamba_d_state,
            kv_dtype,
            mamba_dtype,
            has_position_ids,
            has_num_logits_to_keep,
        })
    }

    fn is_attn_layer(&self, layer: usize) -> bool {
        self.attn_layers.binary_search(&layer).is_ok()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decode state
// ─────────────────────────────────────────────────────────────────────────────

enum AttnCache {
    F32(Vec<Vec<f32>>, Vec<Vec<f32>>),
    F16(Vec<Vec<f16>>, Vec<Vec<f16>>),
}

enum MambaCache {
    F32 {
        conv: Vec<Vec<f32>>,
        ssm: Vec<Vec<f32>>,
    },
    F16 {
        conv: Vec<Vec<f16>>,
        ssm: Vec<Vec<f16>>,
    },
}

struct DecodeState {
    attn: AttnCache,
    mamba: MambaCache,
    total_seq_len: usize,
}

impl DecodeState {
    fn new(arch: &Arch) -> Self {
        let n = arch.num_layers;

        let attn = match arch.kv_dtype {
            KvDtype::F32 => AttnCache::F32(vec![vec![]; n], vec![vec![]; n]),
            KvDtype::F16 => AttnCache::F16(vec![vec![]; n], vec![vec![]; n]),
        };

        let mamba = match arch.mamba_dtype {
            KvDtype::F32 => MambaCache::F32 {
                conv: vec![
                    vec![0.0f32; arch.conv_d_inner * arch.d_conv];
                    n
                ],
                ssm: vec![
                    vec![0.0f32; arch.mamba_n_heads * arch.mamba_d_head * arch.mamba_d_state];
                    n
                ],
            },
            KvDtype::F16 => MambaCache::F16 {
                conv: vec![
                    vec![f16::ZERO; arch.conv_d_inner * arch.d_conv];
                    n
                ],
                ssm: vec![
                    vec![f16::ZERO; arch.mamba_n_heads * arch.mamba_d_head * arch.mamba_d_state];
                    n
                ],
            },
        };

        Self {
            attn,
            mamba,
            total_seq_len: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

pub struct GraniteBackend {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
    arch: Arch,
    eos_token_id: u32,
}

impl GraniteBackend {
    pub fn new<P: AsRef<Path>>(model_path: P, intra_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        let onnx_path = ["model.onnx", "model_q4f16.onnx", "model_fp16.onnx", "model_q4.onnx"]
            .iter()
            .map(|f| path.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| anyhow!("No ONNX model file found in {:?}", path))?;

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("tokenizer.json not found in {:?}", path));
        }

        log::info!(
            "Loading Granite model: {}",
            onnx_path.file_name().unwrap_or_default().to_string_lossy()
        );

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        let session = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&onnx_path)?;

        let arch = Arch::from_session(&session)?;

        let vocab = tokenizer.get_vocab(true);
        let eos_token_id = vocab
            .get("<|end_of_text|>")
            .copied()
            .unwrap_or(49153);

        log::info!("Granite ready — eos={}", eos_token_id);

        Ok(Self {
            session,
            tokenizer,
            arch,
            eos_token_id,
        })
    }

    /// Generate a response for the given plain-text prompt.
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        // Apply Granite chat template:
        //   <|start_of_role|>user<|end_of_role|>{prompt}<|end_of_text|>
        //   <|start_of_role|>assistant<|end_of_role|>
        let formatted = format!(
            "<|start_of_role|>user<|end_of_role|>{}<|end_of_text|>\
             <|start_of_role|>assistant<|end_of_role|>",
            prompt
        );

        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        log::debug!("Granite prompt tokens: {}", prompt_ids.len());

        let output_ids = self.greedy_decode(&prompt_ids, 1024)?;

        let output_text = self
            .tokenizer
            .decode(&output_ids, true)
            .map_err(|e| anyhow!("Decode failed: {}", e))?;

        Ok(output_text.trim().to_string())
    }

    fn greedy_decode(&mut self, prompt_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut state = DecodeState::new(&self.arch);
        let mut generated: Vec<u32> = Vec::new();

        // Prefill: process entire prompt in one pass
        let next_token = self.decoder_step(prompt_ids, &mut state)?;
        state.total_seq_len += prompt_ids.len();

        if next_token == self.eos_token_id {
            return Ok(generated);
        }
        generated.push(next_token);

        // Autoregressive decode
        for step in 1..max_new_tokens {
            let last = *generated.last().unwrap();
            let next_token = self.decoder_step(&[last], &mut state)?;
            state.total_seq_len += 1;
            log::debug!("Granite token[{}]: {}", step, next_token);
            if next_token == self.eos_token_id {
                break;
            }
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn decoder_step(&mut self, input_ids: &[u32], state: &mut DecodeState) -> Result<u32> {
        let batch = 1usize;
        let cur_seq = input_ids.len();
        let total_seq = state.total_seq_len + cur_seq;

        let ids_i64: Vec<i64> = input_ids.iter().map(|&t| t as i64).collect();
        let attn_mask: Vec<i64> = vec![1i64; total_seq];

        // Capture arch values before the mutable self borrow below
        let attn_heads = self.arch.attn_heads;
        let attn_head_dim = self.arch.attn_head_dim;
        let conv_d_inner = self.arch.conv_d_inner;
        let d_conv = self.arch.d_conv;
        let mamba_n_heads = self.arch.mamba_n_heads;
        let mamba_d_head = self.arch.mamba_d_head;
        let mamba_d_state = self.arch.mamba_d_state;
        let num_layers = self.arch.num_layers;
        let attn_layers = self.arch.attn_layers.clone();
        let has_position_ids = self.arch.has_position_ids;
        let has_num_logits_to_keep = self.arch.has_num_logits_to_keep;

        let mut inputs: Vec<(String, ort::value::Value)> = Vec::new();

        inputs.push((
            "input_ids".to_string(),
            ort::value::Value::from_array(([batch, cur_seq], ids_i64))?.into(),
        ));
        inputs.push((
            "attention_mask".to_string(),
            ort::value::Value::from_array(([batch, total_seq], attn_mask))?.into(),
        ));

        if has_position_ids {
            let start = state.total_seq_len as i64;
            let pos: Vec<i64> = (start..start + cur_seq as i64).collect();
            inputs.push((
                "position_ids".to_string(),
                ort::value::Value::from_array(([batch, cur_seq], pos))?.into(),
            ));
        }

        if has_num_logits_to_keep {
            inputs.push((
                "num_logits_to_keep".to_string(),
                ort::value::Value::from_array(([] as [usize; 0], vec![1i64]))?.into(),
            ));
        }

        for layer in 0..num_layers {
            if attn_layers.binary_search(&layer).is_ok() {
                match &state.attn {
                    AttnCache::F32(ref keys, ref vals) => {
                        let past_seq = if keys[layer].is_empty() {
                            0
                        } else {
                            keys[layer].len() / (batch * attn_heads * attn_head_dim)
                        };
                        let p = past_seq.max(1);
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
                        let p = past_seq.max(1);
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
                // Mamba layer
                match &state.mamba {
                    MambaCache::F32 { conv, ssm } => {
                        inputs.push((
                            format!("past_conv.{}", layer),
                            ort::value::Value::from_array((
                                [batch, conv_d_inner, d_conv],
                                conv[layer].clone(),
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_ssm.{}", layer),
                            ort::value::Value::from_array((
                                [batch, mamba_n_heads, mamba_d_head, mamba_d_state],
                                ssm[layer].clone(),
                            ))?
                            .into(),
                        ));
                    }
                    MambaCache::F16 { conv, ssm } => {
                        inputs.push((
                            format!("past_conv.{}", layer),
                            ort::value::Value::from_array((
                                [batch, conv_d_inner, d_conv],
                                conv[layer].clone(),
                            ))?
                            .into(),
                        ));
                        inputs.push((
                            format!("past_ssm.{}", layer),
                            ort::value::Value::from_array((
                                [batch, mamba_n_heads, mamba_d_head, mamba_d_state],
                                ssm[layer].clone(),
                            ))?
                            .into(),
                        ));
                    }
                }
            }
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow!("Granite decoder step failed: {}", e))?;

        // Extract next token from logits (output 0)
        let next_token = {
            let (logits_f32, logits_f16, vocab_size, seq_out) =
                if let Ok((shape, data)) = outputs[0].try_extract_tensor::<f32>() {
                    let dims = shape.as_ref();
                    let vs = *dims.last().unwrap() as usize;
                    let so = if dims.len() == 3 { dims[1] as usize } else { 1 };
                    (Some(data.to_vec()), None, vs, so)
                } else {
                    let (shape, data) = outputs[0].try_extract_tensor::<f16>()?;
                    let dims = shape.as_ref();
                    let vs = *dims.last().unwrap() as usize;
                    let so = if dims.len() == 3 { dims[1] as usize } else { 1 };
                    (
                        None,
                        Some(data.iter().map(|x| x.to_f32()).collect::<Vec<f32>>()),
                        vs,
                        so,
                    )
                };
            let last_offset = (seq_out - 1) * vocab_size;
            let logits = logits_f32.as_deref().or(logits_f16.as_deref()).unwrap();
            let last_logits = &logits[last_offset..last_offset + vocab_size];
            last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(self.eos_token_id)
        };

        // Update caches from present_* outputs (output 1 onwards, in layer order)
        // Per layer: attn → 2 outputs (key, value); mamba → 2 outputs (conv, ssm)
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
                match &mut state.mamba {
                    MambaCache::F32 {
                        ref mut conv,
                        ref mut ssm,
                    } => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                            conv[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f32>() {
                                ssm[layer] = data.to_vec();
                            }
                            out_idx += 1;
                        }
                    }
                    MambaCache::F16 {
                        ref mut conv,
                        ref mut ssm,
                    } => {
                        if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                            conv[layer] = data.to_vec();
                        }
                        out_idx += 1;
                        if out_idx < outputs.len() {
                            if let Ok((_, data)) = outputs[out_idx].try_extract_tensor::<f16>() {
                                ssm[layer] = data.to_vec();
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
