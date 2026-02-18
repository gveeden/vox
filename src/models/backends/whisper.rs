use anyhow::{anyhow, Result};
use ndarray::{arr1, Array1, Array2, Array3, Array4, ArrayView3, Axis};
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::path::Path;

use super::{audio, whisper_decoder_layers, TranscriptionBackend};

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
    // Model dimensions (vary by model size)
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
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

        // Detect model dimensions based on model name
        let (num_layers, num_heads, head_dim, n_mels) = detect_whisper_dimensions(&model_name)?;

        // Encoder sequence length varies by model size
        let encoder_seq_len = if n_mels == 128 {
            1500 // Large v3 uses different encoder output
        } else {
            1500 // Standard for other models
        };

        log::info!(
            "Whisper model '{}' - {} layers, {} heads, {} head_dim, {} mels, {} encoder_len",
            model_name,
            num_layers,
            num_heads,
            head_dim,
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
            num_layers,
            num_heads,
            head_dim,
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
        let batch_size = 1;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let num_layers = self.num_layers;
        let encoder_seq_len = 1500;

        // Prepare input_ids (input 0)
        let input_ids_shape = input_ids.shape();
        let input_ids_dims = [input_ids_shape[0], input_ids_shape[1]];
        let input_ids_data = input_ids.iter().copied().collect::<Vec<i64>>();
        let input_ids_value: Value =
            Value::from_array((input_ids_dims.as_ref(), input_ids_data))?.into();

        // Prepare encoder_hidden_states (input 1)
        let encoder_shape = encoder_hidden.shape();
        let encoder_dims = [encoder_shape[0], encoder_shape[1], encoder_shape[2]];
        let encoder_data = encoder_hidden.iter().copied().collect::<Vec<f32>>();
        let encoder_value: Value = Value::from_array((encoder_dims.as_ref(), encoder_data))?.into();

        // Prepare past key-values for first decode step (inputs 2 to 2+4*num_layers-1)
        // Each layer has 4 tensors: decoder.key, decoder.value, encoder.key, encoder.value
        // HACK: Use seq_len=1 with zeros instead of seq_len=0 because ort doesn't support
        // 0-sized dimensions in tuple format.
        let past_seq_len = 1;

        let decoder_kv_size = batch_size * num_heads * past_seq_len * head_dim;
        let encoder_kv_size = batch_size * num_heads * encoder_seq_len * head_dim;

        // use_cache_branch (last input) - set to false to disable caching
        let use_cache: Value = Value::from_array(([1], vec![false]))?.into();

        log::debug!(
            "Decoder inputs: input_ids {:?}, encoder {:?}, layers={}, heads={}, past_seq_len={}",
            input_ids_dims,
            encoder_dims,
            num_layers,
            num_heads,
            past_seq_len,
        );

        // Run decoder with the correct number of inputs based on model size
        // We need to create KV tensors inline since Value doesn't implement Clone
        let outputs = match num_layers {
            4 => {
                // Tiny: 4 layers = 16 KV tensors
                let kv0: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv1: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv2: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv3: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv4: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv5: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv6: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv7: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv8: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv9: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv10: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv11: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv12: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv13: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv14: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv15: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                self.decoder.run(ort::inputs![
                    input_ids_value,
                    encoder_value,
                    kv0,
                    kv1,
                    kv2,
                    kv3,
                    kv4,
                    kv5,
                    kv6,
                    kv7,
                    kv8,
                    kv9,
                    kv10,
                    kv11,
                    kv12,
                    kv13,
                    kv14,
                    kv15,
                    use_cache
                ]).map_err(|e| anyhow!("Decoder failed: {}", e))
            }
            6 => {
                // Base: 6 layers = 24 KV tensors
                let kv0: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv1: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv2: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv3: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv4: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv5: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv6: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv7: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv8: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv9: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv10: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv11: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv12: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv13: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv14: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv15: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv16: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv17: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv18: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv19: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv20: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv21: Value = Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into();
                let kv22: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                let kv23: Value = Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into();
                self.decoder.run(ort::inputs![
                    input_ids_value,
                    encoder_value,
                    kv0,
                    kv1,
                    kv2,
                    kv3,
                    kv4,
                    kv5,
                    kv6,
                    kv7,
                    kv8,
                    kv9,
                    kv10,
                    kv11,
                    kv12,
                    kv13,
                    kv14,
                    kv15,
                    kv16,
                    kv17,
                    kv18,
                    kv19,
                    kv20,
                    kv21,
                    kv22,
                    kv23,
                    use_cache
                ]).map_err(|e| anyhow!("Decoder failed: {}", e))
            }
            12 => {
                // Small: 12 layers = 48 KV tensors
                self.run_decoder_with_layers_12(
                    input_ids_value, encoder_value, use_cache,
                    batch_size, num_heads, past_seq_len, head_dim, encoder_seq_len,
                    decoder_kv_size, encoder_kv_size
                )
            }
            24 => {
                // Medium: 24 layers = 96 KV tensors
                self.run_decoder_with_layers_24(
                    input_ids_value, encoder_value, use_cache,
                    batch_size, num_heads, past_seq_len, head_dim, encoder_seq_len,
                    decoder_kv_size, encoder_kv_size
                )
            }
            32 => {
                // Large: 32 layers = 128 KV tensors
                self.run_decoder_with_layers_32(
                    input_ids_value, encoder_value, use_cache,
                    batch_size, num_heads, past_seq_len, head_dim, encoder_seq_len,
                    decoder_kv_size, encoder_kv_size
                )
            }
            _ => {
                return Err(anyhow!(
                    "Unsupported number of layers: {}. Currently supported: 4 (tiny), 6 (base), 12 (small), 24 (medium), 32 (large)",
                    num_layers
                ))
            }
        };

        let outputs = match outputs {
            Ok(out) => out,
            Err(e) => {
                log::error!("Decoder inference failed: {}", e);
                return Err(anyhow!("Decoder failed: {}", e));
            }
        };

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

/// Detect Whisper model dimensions from model name
/// Returns (num_layers, num_heads, head_dim, n_mels)
fn detect_whisper_dimensions(model_name: &str) -> Result<(usize, usize, usize, usize)> {
    let model_lower = model_name.to_lowercase();

    // Whisper model specifications
    // Note: These are decoder layer counts for KV cache
    if model_lower.contains("large") {
        // Large: 32 layers, 20 heads, 64 head_dim, 128 mels (Large v2/v3)
        log::info!("Detected Whisper Large model");
        Ok((32, 20, 64, 128))
    } else if model_lower.contains("medium") {
        // Medium: 24 layers, 16 heads, 64 head_dim, 80 mels
        log::info!("Detected Whisper Medium model");
        Ok((24, 16, 64, 80))
    } else if model_lower.contains("small") {
        // Small: 12 layers, 12 heads, 64 head_dim, 80 mels
        log::info!("Detected Whisper Small model");
        Ok((12, 12, 64, 80))
    } else if model_lower.contains("base") {
        // Base: 6 layers, 8 heads, 64 head_dim, 80 mels
        log::info!("Detected Whisper Base model");
        Ok((6, 8, 64, 80))
    } else if model_lower.contains("tiny") {
        // Tiny: 4 layers, 6 heads, 64 head_dim, 80 mels
        log::info!("Detected Whisper Tiny model");
        Ok((4, 6, 64, 80))
    } else {
        // Default to Tiny for unknown variants
        log::warn!(
            "Unknown Whisper variant '{}', defaulting to Tiny dimensions",
            model_name
        );
        Ok((4, 6, 64, 80))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_tokenizer_decode() {
        let encoder: HashMap<String, i64> = [
            ("hello".to_string(), 1),
            ("world".to_string(), 2),
            ("Ġtest".to_string(), 3),
            ("<|endoftext|>".to_string(), 50256),
        ]
        .into_iter()
        .collect();

        let decoder: HashMap<i64, String> = [
            (1, "hello".to_string()),
            (2, "world".to_string()),
            (3, "Ġtest".to_string()),
            (50256, "<|endoftext|>".to_string()),
        ]
        .into_iter()
        .collect();

        let special_tokens = WhisperSpecialTokens {
            sot: 50258,
            eot: 50256,
            transcribe: 50359,
            translate: 50358,
            no_speech: 50362,
            notimestamp: 50363,
        };

        let tokenizer = WhisperTokenizer {
            encoder,
            decoder,
            special_tokens,
        };

        let tokens = vec![1, 2, 3, 50256];
        let text = tokenizer.decode(&tokens, true);
        assert_eq!(text, "helloworld test");
    }
}
