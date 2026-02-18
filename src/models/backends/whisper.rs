use anyhow::{anyhow, Result};
use ndarray::{Array2, Array3, Axis};
use ort::session::Session;
use ort::tensor::TensorElementType;
use ort::value::{Value, ValueType};
use std::path::Path;

use super::{audio, TranscriptionBackend};

/// Classification of a single decoder input slot, derived from the ONNX model schema.
#[derive(Debug, Clone)]
enum DecoderInputKind {
    InputIds,
    EncoderHidden,
    DecoderKV,
    EncoderKV,
    /// Rank-1 int64 arange [0, 1, ..., seq_len-1] giving absolute token positions.
    /// Present in newer Optimum exports (e.g. whisper-small) but absent in older ones.
    CachePosition,
    UseCacheBranch,
}

/// Pre-computed schema for the decoder's inputs, derived once at load time by
/// inspecting `session.inputs()`.  This drives the dynamic `Vec<(String,Value)>`
/// build in `decode_step()` without any hard-coded layer counts or tuple macros.
#[derive(Debug)]
struct DecoderInputSchema {
    /// Ordered list of (input_name, kind) pairs matching session.inputs() order.
    slots: Vec<(String, DecoderInputKind)>,
    /// Number of decoder transformer layers (= number of DecoderKV slots).
    num_layers: usize,
    /// Number of attention heads per layer.
    num_heads: usize,
    /// Per-head key/value dimension.
    head_dim: usize,
}

/// Whisper backend for transcription
pub struct WhisperBackend {
    encoder: Session,
    pub(crate) decoder: Session,
    tokenizer: WhisperTokenizer,
    model_name: String,
    // Whisper-specific configuration
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    // Derived once from the decoder ONNX schema
    decoder_schema: DecoderInputSchema,
    encoder_seq_len: usize,
}

/// Whisper tokenizer using HuggingFace tokenizer
struct WhisperTokenizer {
    hf_tokenizer: tokenizers::Tokenizer,
    special_tokens: WhisperSpecialTokens,
}

struct WhisperSpecialTokens {
    sot: i64, // Start of transcript
    eot: i64, // End of transcript
    transcribe: i64,
    translate: i64,
    no_speech: i64,
    notimestamp: i64,
}

impl WhisperBackend {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();

        // Find encoder and decoder files
        let encoder_path = if path.join("encoder.onnx").exists() {
            path.join("encoder.onnx")
        } else if path.join("encoder_model.onnx").exists() {
            path.join("encoder_model.onnx")
        } else {
            return Err(anyhow!("Encoder model not found in {:?}", path));
        };

        let decoder_path = if path.join("decoder.onnx").exists() {
            path.join("decoder.onnx")
        } else if path.join("decoder_model_merged.onnx").exists() {
            path.join("decoder_model_merged.onnx")
        } else if path.join("decoder_model.onnx").exists() {
            path.join("decoder_model.onnx")
        } else {
            return Err(anyhow!("Decoder model not found in {:?}", path));
        };

        // Load tokenizer from tokenizer.json (HuggingFace format)
        let tokenizer_path = path.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Err(anyhow!("tokenizer.json not found in {:?}", path));
        }

        let tokenizer = WhisperTokenizer::from_hf_tokenizer(&tokenizer_path)?;

        // Load encoder session
        let encoder = Session::builder()?.commit_from_file(&encoder_path)?;

        // Load decoder session
        let decoder = Session::builder()?.commit_from_file(&decoder_path)?;

        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("whisper")
            .to_string();

        // Auto-detect decoder input schema from the ONNX session metadata.
        let decoder_schema = DecoderInputSchema::from_session(&decoder, &model_name)?;

        // n_mels is an audio pre-processing constant that isn't reliably encoded
        // in the ONNX shape (all dims are dynamic in Optimum exports), so we
        // keep a lightweight name-based fallback just for this one value.
        let n_mels = detect_n_mels(&model_name);

        // All standard Whisper variants use 1500 encoder frames.
        let encoder_seq_len = 1500;

        log::info!(
            "Whisper model '{}' - {} layers, {} heads, {} head_dim, {} mels, {} encoder_len",
            model_name,
            decoder_schema.num_layers,
            decoder_schema.num_heads,
            decoder_schema.head_dim,
            n_mels,
            encoder_seq_len
        );

        Ok(Self {
            encoder,
            decoder,
            tokenizer,
            model_name,
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels,
            decoder_schema,
            encoder_seq_len,
        })
    }

    /// Compute log-mel spectrogram for Whisper input
    fn compute_features(&self, audio: &[f32]) -> Result<Array3<f32>> {
        // Whisper expects 30 seconds of audio at 16kHz = 480000 samples
        const MAX_SAMPLES: usize = 480000; // 30s * 16000

        // Pad or truncate audio to 30 seconds
        let audio = if audio.len() > MAX_SAMPLES {
            audio[..MAX_SAMPLES].to_vec()
        } else {
            let mut padded = audio.to_vec();
            padded.resize(MAX_SAMPLES, 0.0);
            padded
        };

        // Compute mel spectrogram
        let mel_spec = audio::compute_mel_spectrogram(
            &audio,
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            0.0,
            8000.0,
        )?;

        // Log mel spec shape for debugging
        log::debug!(
            "Mel spectrogram shape: {:?} (expected: [80, 3000])",
            mel_spec.shape()
        );

        // Convert to [1, n_mels, n_frames] format
        let mel_spec = mel_spec.insert_axis(Axis(0));

        // Apply Whisper's normalization formula
        // Reference: transformers/models/whisper/feature_extraction_whisper.py

        // 1. Clip dynamic range to 8 decades (max - 8.0)
        let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mel_spec = mel_spec.mapv(|x| x.max(max_val - 8.0));

        // 2. Normalize: (log_spec + 4.0) / 4.0
        // This transforms from ~[-8, 0] range to ~[0, 2] range
        let mel_spec = mel_spec.mapv(|x| (x + 4.0) / 4.0);

        Ok(mel_spec)
    }

    /// Run encoder to get audio features
    fn encode(&mut self, mel: &Array3<f32>) -> Result<Array3<f32>> {
        // Create input value from ndarray (convert to tuple form for ort)
        let shape = mel.shape();
        let dims = [shape[0], shape[1], shape[2]];
        let data = mel.iter().copied().collect::<Vec<f32>>();
        let input_value = Value::from_array((dims.as_ref(), data))?;

        log::debug!("Encoder input shape: {:?}", dims);

        // Run encoder
        let outputs = self.encoder.run(ort::inputs![input_value])?;

        // Extract first output (encoder hidden states)
        let output_tensor = &outputs[0];
        let (shape, data) = output_tensor.try_extract_tensor::<f32>()?;

        // Convert to ndarray - shape is [batch, time, features]
        let dims = shape.as_ref();

        log::debug!("Encoder output shape from ONNX: {:?}", dims);

        if dims.len() != 3 {
            return Err(anyhow!(
                "Encoder output has unexpected rank: {} (expected 3), shape: {:?}",
                dims.len(),
                dims
            ));
        }

        let hidden_states = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )?;

        Ok(hidden_states)
    }

    /// Run decoder to get next token logits
    fn decode_step(
        &mut self,
        input_ids: &Array2<i64>,
        encoder_hidden: &Array3<f32>,
    ) -> Result<Array3<f32>> {
        let batch_size = 1usize;
        let num_heads = self.decoder_schema.num_heads;
        let head_dim = self.decoder_schema.head_dim;
        let encoder_seq_len = self.encoder_seq_len;

        // Prepare input_ids value
        let input_ids_shape = input_ids.shape();
        let input_ids_dims = [input_ids_shape[0], input_ids_shape[1]];
        let input_ids_data = input_ids.iter().copied().collect::<Vec<i64>>();

        // Prepare encoder_hidden_states value
        let encoder_shape = encoder_hidden.shape();
        let encoder_dims = [encoder_shape[0], encoder_shape[1], encoder_shape[2]];
        let encoder_data = encoder_hidden.iter().copied().collect::<Vec<f32>>();

        // HACK: Use seq_len=1 with zeros instead of seq_len=0 because ort doesn't
        // support 0-sized dimensions.
        let past_seq_len = 1usize;
        let decoder_kv_size = batch_size * num_heads * past_seq_len * head_dim;
        let encoder_kv_size = batch_size * num_heads * encoder_seq_len * head_dim;

        log::debug!(
            "Decoder inputs: input_ids {:?}, encoder {:?}, layers={}, heads={}, past_seq_len={}",
            input_ids_dims,
            encoder_dims,
            self.decoder_schema.num_layers,
            num_heads,
            past_seq_len,
        );

        // Build named inputs dynamically from the pre-computed schema, which was
        // derived from session.inputs() at load time.  This ensures we always
        // match the exact order and types the ONNX model expects, regardless of
        // model size or export tool.
        let mut inputs: Vec<(String, Value)> = Vec::with_capacity(self.decoder_schema.slots.len());
        for (name, kind) in &self.decoder_schema.slots {
            let value: Value = match kind {
                DecoderInputKind::InputIds => {
                    Value::from_array((input_ids_dims.as_ref(), input_ids_data.clone()))?.into()
                }
                DecoderInputKind::EncoderHidden => {
                    Value::from_array((encoder_dims.as_ref(), encoder_data.clone()))?.into()
                }
                DecoderInputKind::DecoderKV => Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into(),
                DecoderInputKind::EncoderKV => Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into(),
                DecoderInputKind::CachePosition => {
                    // Rank-1 int64 arange [0, 1, ..., seq_len-1] giving the absolute
                    // position of each token in the current decoder input sequence.
                    let seq_len = input_ids_dims[1];
                    let positions: Vec<i64> = (0..seq_len as i64).collect();
                    Value::from_array(([seq_len], positions))?.into()
                }
                DecoderInputKind::UseCacheBranch => {
                    Value::from_array(([1usize], vec![false]))?.into()
                }
            };
            inputs.push((name.clone(), value));
        }

        let outputs = self.decoder.run(inputs).map_err(|e| {
            log::error!("Decoder inference failed: {}", e);
            anyhow!("Decoder failed: {}", e)
        })?;

        // Get logits - shape is [batch, seq_len, vocab_size]
        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        let dims = shape.as_ref();

        log::debug!("Decoder output shape: {:?}, data len: {}", dims, data.len());

        if dims.len() < 3 {
            return Err(anyhow!(
                "Decoder output has unexpected rank: {} (expected 3)",
                dims.len()
            ));
        }

        let logits = Array3::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data.to_vec(),
        )?;

        Ok(logits)
    }

    /// Greedy decoding
    fn greedy_decode(
        &mut self,
        encoder_hidden: &Array3<f32>,
        max_length: usize,
    ) -> Result<Vec<i64>> {
        // Start with SOT token
        let mut tokens = vec![self.tokenizer.special_tokens.sot];

        // Add language token (English) and task token
        tokens.push(50259); // <|en|>
        tokens.push(self.tokenizer.special_tokens.transcribe);

        log::debug!("Starting greedy decode, initial tokens: {:?}", tokens);

        for step in 0..max_length {
            // Prepare input: [batch=1, seq_len]
            let input_ids = Array2::from_shape_vec((1, tokens.len()), tokens.clone())?;

            log::debug!("Step {}: input_ids shape {:?}", step, input_ids.shape());

            // Run decoder
            let logits = self.decode_step(&input_ids, encoder_hidden)?;

            log::debug!("Step {}: logits shape {:?}", step, logits.shape());

            // Get last token logits - shape [batch, vocab_size]
            // logits shape: [1, seq_len, vocab_size]
            let seq_len = logits.shape()[1];

            if seq_len == 0 {
                return Err(anyhow!("Decoder returned empty sequence"));
            }

            let last_logits_slice = logits.slice(ndarray::s![0, seq_len - 1, ..]);
            let last_logits: Vec<f32> = last_logits_slice.iter().copied().collect();

            // Greedy: pick highest probability
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(self.tokenizer.special_tokens.eot);

            tokens.push(next_token);

            log::debug!(
                "Step {}: next_token={}, tokens so far: {:?}",
                step,
                next_token,
                &tokens[..tokens.len().min(10)]
            );

            if next_token == self.tokenizer.special_tokens.eot {
                log::debug!("Got EOT token at step {}", step);
                break;
            }

            // Also stop on no_speech token if it's early
            if next_token == self.tokenizer.special_tokens.no_speech && tokens.len() < 10 {
                log::debug!("Got no_speech token at step {}", step);
                break;
            }
        }
        Ok(tokens)
    }
}

impl TranscriptionBackend for WhisperBackend {
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        // Resample to 16kHz if needed
        let mut audio = if sample_rate != self.sample_rate {
            audio::resample_audio(audio, sample_rate, self.sample_rate)
        } else {
            audio.to_vec()
        };

        // Normalize audio to have max absolute value of 1.0 (Whisper expects normalized audio)
        let max_abs = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > 0.0 {
            let scale = 1.0 / max_abs;
            for sample in &mut audio {
                *sample *= scale;
            }
        }

        // Check audio stats
        let audio_min = audio.iter().copied().fold(f32::INFINITY, f32::min);
        let audio_max = audio.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let audio_mean = audio.iter().sum::<f32>() / audio.len() as f32;
        let audio_rms = (audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32).sqrt();
        log::info!(
            "Audio stats: len={}, min={:.3}, max={:.3}, mean={:.3}, rms={:.3}",
            audio.len(),
            audio_min,
            audio_max,
            audio_mean,
            audio_rms
        );

        // Compute mel spectrogram
        let mel = self.compute_features(&audio)?;

        // Check mel spectrogram stats
        let mel_min = mel.iter().copied().fold(f32::INFINITY, f32::min);
        let mel_max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mel_mean = mel.iter().sum::<f32>() / mel.len() as f32;
        log::info!(
            "Mel spectrogram stats: shape={:?}, min={:.3}, max={:.3}, mean={:.3}",
            mel.shape(),
            mel_min,
            mel_max,
            mel_mean
        );

        // Encode
        let encoder_hidden = self.encode(&mel)?;

        // Check encoder output stats
        let enc_min = encoder_hidden.iter().copied().fold(f32::INFINITY, f32::min);
        let enc_max = encoder_hidden
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let enc_mean = encoder_hidden.iter().sum::<f32>() / encoder_hidden.len() as f32;
        log::info!(
            "Encoder output stats: shape={:?}, min={:.3}, max={:.3}, mean={:.3}",
            encoder_hidden.shape(),
            enc_min,
            enc_max,
            enc_mean
        );

        // Decode
        let tokens = self.greedy_decode(&encoder_hidden, 448)?;

        log::info!(
            "Generated {} tokens: {:?}",
            tokens.len(),
            &tokens[..tokens.len().min(20)]
        );

        // Convert tokens to text
        let text = self.tokenizer.decode(&tokens, true);

        log::info!("Decoded text: '{}'", text);

        Ok(text)
    }

    fn backend_type(&self) -> &'static str {
        "whisper"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl WhisperTokenizer {
    /// Load from HuggingFace tokenizer.json format
    fn from_hf_tokenizer<P: AsRef<Path>>(path: P) -> Result<Self> {
        let hf_tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("Failed to load HF tokenizer: {}", e))?;

        // Extract vocab to get special token IDs
        let vocab = hf_tokenizer.get_vocab(true);

        // Get special token IDs
        let sot = *vocab.get("<|startoftranscript|>").unwrap_or(&50258) as i64;
        let eot = *vocab.get("<|endoftext|>").unwrap_or(&50256) as i64;
        let transcribe = *vocab.get("<|transcribe|>").unwrap_or(&50359) as i64;
        let translate = *vocab.get("<|translate|>").unwrap_or(&50358) as i64;
        let no_speech = *vocab.get("<|nospeech|>").unwrap_or(&50362) as i64;
        let notimestamp = *vocab.get("<|notimestamps|>").unwrap_or(&50363) as i64;

        let special_tokens = WhisperSpecialTokens {
            sot,
            eot,
            transcribe,
            translate,
            no_speech,
            notimestamp,
        };

        Ok(Self {
            hf_tokenizer,
            special_tokens,
        })
    }

    fn decode(&self, tokens: &[i64], skip_special: bool) -> String {
        // Convert i64 tokens to u32 for HF tokenizer
        let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();

        // Use HF tokenizer's decode method which handles BPE correctly
        match self.hf_tokenizer.decode(&token_ids, skip_special) {
            Ok(text) => text,
            Err(e) => {
                log::error!("Tokenizer decode error: {}", e);
                String::new()
            }
        }
    }

    fn is_special_token(&self, token: i64) -> bool {
        // Special tokens are generally >= 50256 in Whisper
        token >= 50256 && token < 50365
    }
}

impl DecoderInputSchema {
    /// Build the schema by inspecting `session.inputs()` from the loaded ONNX
    /// decoder model.  Input names follow the HuggingFace Optimum convention:
    ///
    ///   input_ids                        → i64  [batch, seq]
    ///   encoder_hidden_states            → f32  [batch, enc_seq, d_model]
    ///   past_key_values.N.decoder.key    → f32  [batch, heads, past_seq, head_dim]
    ///   past_key_values.N.decoder.value  → f32  [batch, heads, past_seq, head_dim]
    ///   past_key_values.N.encoder.key    → f32  [batch, heads, enc_seq, head_dim]
    ///   past_key_values.N.encoder.value  → f32  [batch, heads, enc_seq, head_dim]
    ///   use_cache_branch                 → bool [1]
    ///
    /// `num_heads` and `head_dim` are read from the concrete shape of the first
    /// decoder KV tensor when available (non-negative dims), with a name-based
    /// fallback for models exported with fully dynamic axes.
    fn from_session(session: &Session, model_name: &str) -> Result<Self> {
        let mut slots = Vec::new();
        let mut num_layers = 0usize;
        let mut num_heads_from_shape: Option<usize> = None;
        let mut head_dim_from_shape: Option<usize> = None;

        for outlet in session.inputs() {
            let name = outlet.name();
            let kind = if name == "input_ids" {
                DecoderInputKind::InputIds
            } else if name == "encoder_hidden_states" {
                DecoderInputKind::EncoderHidden
            } else if name.contains("decoder.key") || name.contains("decoder.value") {
                // Try to read num_heads / head_dim from the shape of the first
                // decoder KV tensor (shape: [batch, heads, seq, head_dim]).
                if num_heads_from_shape.is_none() {
                    if let ValueType::Tensor { shape, .. } = outlet.dtype() {
                        let dims = shape.as_ref();
                        if dims.len() == 4 && dims[1] > 0 && dims[3] > 0 {
                            num_heads_from_shape = Some(dims[1] as usize);
                            head_dim_from_shape = Some(dims[3] as usize);
                        }
                    }
                }
                if name.contains("decoder.key") {
                    num_layers += 1;
                }
                DecoderInputKind::DecoderKV
            } else if name.contains("encoder.key") || name.contains("encoder.value") {
                DecoderInputKind::EncoderKV
            } else if name == "cache_position" {
                DecoderInputKind::CachePosition
            } else if name == "use_cache_branch" {
                DecoderInputKind::UseCacheBranch
            } else if matches!(outlet.dtype().tensor_type(), Some(TensorElementType::Bool)) {
                // Unknown name but bool type — treat as use_cache_branch
                log::warn!(
                    "Unknown bool input '{}', treating as use_cache_branch",
                    name
                );
                DecoderInputKind::UseCacheBranch
            } else if matches!(outlet.dtype().tensor_type(), Some(TensorElementType::Int64)) {
                log::warn!("Unknown i64 input '{}', treating as input_ids", name);
                DecoderInputKind::InputIds
            } else {
                // f32 — count as a KV tensor; assume decoder KV if no encoder hint
                log::warn!("Unknown f32 input '{}', treating as DecoderKV", name);
                DecoderInputKind::DecoderKV
            };
            slots.push((name.to_string(), kind));
        }

        if num_layers == 0 {
            return Err(anyhow!(
                "Could not detect any decoder KV layers from session inputs. \
                 Check that the ONNX model uses HuggingFace Optimum naming."
            ));
        }

        // Fall back to name-based lookup for num_heads / head_dim when the
        // ONNX model was exported with fully dynamic axes (all dims = -1).
        let (num_heads, head_dim) = match (num_heads_from_shape, head_dim_from_shape) {
            (Some(h), Some(d)) => (h, d),
            _ => {
                log::warn!(
                    "KV tensor shape dims are dynamic; falling back to name-based \
                     head/dim lookup for model '{}'",
                    model_name
                );
                detect_heads_and_dim(model_name, num_layers)
            }
        };

        log::info!(
            "Decoder schema: {} layers, {} heads, {} head_dim, {} input slots",
            num_layers,
            num_heads,
            head_dim,
            slots.len()
        );

        Ok(Self {
            slots,
            num_layers,
            num_heads,
            head_dim,
        })
    }
}

/// Return (num_heads, head_dim) from the model name as a fallback when the
/// ONNX shape dimensions are all dynamic.  All standard Whisper variants use
/// head_dim=64; num_heads varies by size.
fn detect_heads_and_dim(model_name: &str, num_layers: usize) -> (usize, usize) {
    let model_lower = model_name.to_lowercase();
    let num_heads = if model_lower.contains("large") && !model_lower.contains("turbo") {
        // Large v1/v2/v3: 32 layers, 20 heads
        20
    } else if model_lower.contains("large") {
        // Large-v3-turbo: 24 layers, 16 heads
        16
    } else if model_lower.contains("medium") {
        16
    } else if model_lower.contains("small") {
        12
    } else if model_lower.contains("base") {
        8
    } else {
        // Tiny or unknown — also derive from layer count as a secondary hint
        match num_layers {
            32 => 20,
            24 => 16,
            12 => 12,
            6 => 8,
            _ => 6, // tiny / default
        }
    };
    (num_heads, 64)
}

/// Detect n_mels (mel-spectrogram frequency bins) from the model name.
/// This is the only dimension that cannot be reliably read from the ONNX
/// session schema because Optimum exports it as a fully dynamic axis.
fn detect_n_mels(model_name: &str) -> usize {
    let model_lower = model_name.to_lowercase();
    if model_lower.contains("large") {
        128
    } else {
        80
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_n_mels() {
        assert_eq!(detect_n_mels("whisper-large-v3"), 128);
        assert_eq!(detect_n_mels("whisper-large-v3-turbo"), 128);
        assert_eq!(detect_n_mels("whisper-medium"), 80);
        assert_eq!(detect_n_mels("whisper-small"), 80);
        assert_eq!(detect_n_mels("whisper-base"), 80);
        assert_eq!(detect_n_mels("whisper-tiny"), 80);
    }

    #[test]
    fn test_detect_heads_and_dim() {
        // head_dim is always 64 for all standard Whisper models
        assert_eq!(detect_heads_and_dim("whisper-tiny", 4), (6, 64));
        assert_eq!(detect_heads_and_dim("whisper-base", 6), (8, 64));
        assert_eq!(detect_heads_and_dim("whisper-small", 12), (12, 64));
        assert_eq!(detect_heads_and_dim("whisper-medium", 24), (16, 64));
        assert_eq!(detect_heads_and_dim("whisper-large-v3-turbo", 24), (16, 64));
        assert_eq!(detect_heads_and_dim("whisper-large-v3", 32), (20, 64));
    }
}
