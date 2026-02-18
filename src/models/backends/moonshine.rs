use anyhow::{anyhow, Result};
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use tokenizers::Tokenizer;

use super::audio::resample_audio;
use super::TranscriptionBackend;

pub struct MoonshineBackend {
    encoder: Session,
    decoder: Session,
    tokenizer: Tokenizer,
    model_name: String,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
}

impl MoonshineBackend {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();

        // Determine encoder path
        let encoder_path = if path.join("encoder.onnx").exists() {
            path.join("encoder.onnx")
        } else if path.join("encoder_model.onnx").exists() {
            path.join("encoder_model.onnx")
        } else {
            return Err(anyhow!("Moonshine encoder model not found"));
        };

        // Determine decoder path
        let decoder_path = if path.join("decoder.onnx").exists() {
            path.join("decoder.onnx")
        } else if path.join("decoder_model_merged.onnx").exists() {
            path.join("decoder_model_merged.onnx")
        } else {
            return Err(anyhow!("Moonshine decoder model not found"));
        };

        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(anyhow!("Moonshine tokenizer.json not found"));
        }

        // Load ONNX models
        let encoder = Session::builder()?
            .commit_from_file(&encoder_path)
            .map_err(|e| anyhow!("Failed to load Moonshine encoder: {}", e))?;

        let decoder = Session::builder()?
            .commit_from_file(&decoder_path)
            .map_err(|e| anyhow!("Failed to load Moonshine decoder: {}", e))?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load Moonshine tokenizer: {}", e))?;

        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("moonshine")
            .to_string();

        // Detect model dimensions from model name
        let (num_layers, num_heads, head_dim) = detect_moonshine_dimensions(&model_name)?;

        log::info!(
            "Moonshine model: {} layers, {} heads, {} head_dim",
            num_layers,
            num_heads,
            head_dim
        );

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            model_name,
            num_layers,
            num_heads,
            head_dim,
        })
    }
}

impl TranscriptionBackend for MoonshineBackend {
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        // Resample to 16kHz if needed
        let audio_16k = if sample_rate != 16000 {
            resample_audio(audio, sample_rate, 16000)
        } else {
            audio.to_vec()
        };

        // Moonshine expects audio normalized to [-1.0, 1.0] (no mel spectrogram!)
        // Just need to ensure the audio is properly normalized
        let max_val = audio_16k.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let normalized_audio: Vec<f32> = if max_val > 0.0 {
            audio_16k.iter().map(|x| x / max_val).collect()
        } else {
            audio_16k
        };

        // Run encoder
        let audio_len = normalized_audio.len();
        let input_values = Value::from_array(([1, audio_len], normalized_audio))?;

        let encoder_outputs = self.encoder.run(ort::inputs![input_values])?;
        let encoder_hidden_states = &encoder_outputs[0];

        // Extract encoder output shape
        let (encoder_shape, _encoder_data) = encoder_hidden_states.try_extract_tensor::<f32>()?;
        let seq_len = encoder_shape[1] as usize;
        let _hidden_size = encoder_shape[2] as usize;

        // Greedy decoding
        // Moonshine uses standard BPE tokens: <s> for start, </s> for end
        let mut input_ids = vec![self
            .tokenizer
            .token_to_id("<s>")
            .ok_or_else(|| anyhow!("BOS token <s> not found"))?];

        let eot_token = self
            .tokenizer
            .token_to_id("</s>")
            .ok_or_else(|| anyhow!("EOS token </s> not found"))?;

        let max_length = 256;

        // Use model-specific parameters
        let num_layers = self.num_layers;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        for step in 0..max_length {
            // Pass ALL tokens generated so far (required when use_cache_branch=false)
            let input_ids_i64: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
            let input_seq_len = input_ids_i64.len();

            // Dummy past_seq_len for zero-filled KV tensors
            let past_seq_len = 1;
            let kv_size = num_heads * past_seq_len * head_dim;

            // use_cache_branch: both models expect bool
            let use_cache_val: Value = Value::from_array(([1], vec![false]))?.into();

            // Run decoder with inputs based on num_layers
            #[allow(noop_method_call)]
            let decoder_outputs = match num_layers {
                6 => self.decoder.run(ort::inputs![
                    Value::from_array(([1, input_seq_len], input_ids_i64))?,
                    encoder_hidden_states.clone(),
                    // Layer 0
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 1
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 2
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 3
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 4
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 5
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    use_cache_val
                ])?,
                8 => self.decoder.run(ort::inputs![
                    Value::from_array(([1, input_seq_len], input_ids_i64))?,
                    encoder_hidden_states.clone(),
                    // Layer 0
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 1
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 2
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 3
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 4
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 5
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 6
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    // Layer 7
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, past_seq_len, head_dim],
                        vec![0.0f32; kv_size]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    Value::from_array((
                        [1, num_heads, seq_len, head_dim],
                        vec![0.0f32; num_heads * seq_len * head_dim]
                    ))?,
                    use_cache_val
                ])?,
                _ => {
                    return Err(anyhow!(
                        "Unsupported num_layers: {}, expected 6 or 8",
                        num_layers
                    ))
                }
            };

            let logits = &decoder_outputs[0];

            // Get next token (greedy)
            let (logits_shape, logits_data) = logits.try_extract_tensor::<f32>()?;
            let vocab_size = logits_shape[2] as usize;

            // Get logits for the last token (last position in sequence)
            let last_token_offset = (logits_shape[1] as usize - 1) * vocab_size;
            let last_token_logits =
                &logits_data[last_token_offset..(last_token_offset + vocab_size)];

            let next_token_id = last_token_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap();

            log::debug!("Step {}: Generated token ID {}", step, next_token_id);

            if next_token_id == eot_token {
                log::debug!("Reached EOS token");
                break;
            }

            input_ids.push(next_token_id);
        }

        log::debug!("Generated {} tokens total", input_ids.len());

        // Decode tokens
        let text = self
            .tokenizer
            .decode(&input_ids, true)
            .map_err(|e| anyhow!("Tokenizer decode failed: {}", e))?;

        Ok(text.trim().to_string())
    }

    fn backend_type(&self) -> &'static str {
        "moonshine"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Detect Moonshine model dimensions from model name
fn detect_moonshine_dimensions(model_name: &str) -> Result<(usize, usize, usize)> {
    // Moonshine model specifications:
    // Tiny: 6 layers, 8 heads, 36 head_dim (hidden_size 288)
    // Base: 8 layers, 8 heads, 52 head_dim (hidden_size 416)

    let model_lower = model_name.to_lowercase();

    if model_lower.contains("base") {
        log::info!("Detected Moonshine Base model");
        Ok((8, 8, 52))
    } else if model_lower.contains("tiny") {
        log::info!("Detected Moonshine Tiny model");
        Ok((6, 8, 36))
    } else {
        // Default to Tiny for unknown variants
        log::warn!(
            "Unknown Moonshine variant '{}', defaulting to Tiny dimensions",
            model_name
        );
        Ok((6, 8, 36))
    }
}
