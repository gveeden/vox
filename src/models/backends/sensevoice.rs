use anyhow::{anyhow, Result};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::audio::{compute_fbank_features, resample_audio};
use super::TranscriptionBackend;

pub struct SenseVoiceBackend {
    model: Session,
    mean: Vec<f32>,
    istd: Vec<f32>, // inverse std (1/std) for faster computation
    token_map: HashMap<i32, String>,
    model_name: String,
}

impl SenseVoiceBackend {
    pub fn new<P: AsRef<Path>>(model_path: P, intra_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        // Check for required files
        let model_file = path.join("model.onnx");
        if !model_file.exists() {
            return Err(anyhow!("SenseVoice model.onnx not found"));
        }

        let mvn_file = path.join("am.mvn");
        if !mvn_file.exists() {
            return Err(anyhow!("SenseVoice am.mvn not found"));
        }

        // Load ONNX model
        let model = Session::builder()?
            .with_intra_threads(intra_threads)?
            .commit_from_file(&model_file)
            .map_err(|e| anyhow!("Failed to load SenseVoice model: {}", e))?;

        // Parse am.mvn file for mean and variance
        let (mean, istd) = parse_am_mvn(&mvn_file)?;

        // Load vocabulary from vocab.json (generated from tokenizer.model)
        let vocab_file = path.join("vocab.json");
        let token_map = if vocab_file.exists() {
            load_vocabulary(&vocab_file)?
        } else {
            log::warn!("vocab.json not found, using basic token map");
            build_basic_token_map()
        };

        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("sensevoice")
            .to_string();

        Ok(Self {
            model,
            mean,
            istd,
            token_map,
            model_name,
        })
    }
}

impl TranscriptionBackend for SenseVoiceBackend {
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        // Resample to 16kHz if needed
        let audio_16k = if sample_rate != 16000 {
            resample_audio(audio, sample_rate, 16000)
        } else {
            audio.to_vec()
        };

        // Compute fbank features: 80-dimensional filter banks
        // SenseVoice uses 560-dim features (80 fbank * 7 frames context)
        let n_fft = 512;
        let hop_length = 160; // 10ms hop at 16kHz
        let n_mels = 80;

        let fbank = compute_fbank_features(&audio_16k, 16000, n_fft, hop_length, n_mels)?;

        // Apply mean-variance normalization
        let mut normalized = fbank.clone();
        let (n_frames, n_features) = (normalized.shape()[0], normalized.shape()[1]);

        for i in 0..n_frames {
            for j in 0..n_features {
                let idx = j % self.mean.len();
                normalized[[i, j]] = (normalized[[i, j]] - self.mean[idx]) * self.istd[idx];
            }
        }

        // Context expansion: SenseVoice expects 560-dim features (7 frames of 80-dim fbank)
        // This means we stack [t-3, t-2, t-1, t, t+1, t+2, t+3] context
        let context_size = 7;
        let expanded_dim = n_mels * context_size;
        let mut expanded_features = Vec::new();

        for t in 0..n_frames {
            let mut frame_context = vec![0.0f32; expanded_dim];

            for c in 0..context_size {
                let offset = c as i32 - (context_size as i32 / 2);
                let src_frame = ((t as i32 + offset).max(0).min(n_frames as i32 - 1)) as usize;

                for f in 0..n_mels {
                    frame_context[c * n_mels + f] = normalized[[src_frame, f]];
                }
            }

            expanded_features.push(frame_context);
        }

        // Convert to model input format: [batch=1, seq_len, 560]
        let seq_len = expanded_features.len();
        let flat_features: Vec<f32> = expanded_features.into_iter().flatten().collect();

        let speech_input = Value::from_array(([1, seq_len, expanded_dim], flat_features))?;
        let speech_lengths = Value::from_array(([1], vec![seq_len as i64]))?;

        // Run model
        let outputs = self.model.run(ort::inputs![speech_input, speech_lengths])?;
        let encoder_out = &outputs[0];

        // Extract output logits
        let (shape, data) = encoder_out.try_extract_tensor::<f32>()?;
        let vocab_size = shape[2] as usize;
        let out_seq_len = shape[1] as usize;

        // Simple greedy CTC decoding - find most likely token at each step
        let mut tokens = Vec::new();
        let mut prev_token = 0i32;

        for t in 0..out_seq_len {
            let offset = t * vocab_size;
            let frame_logits = &data[offset..(offset + vocab_size)];

            let token_id = frame_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap_or(0);

            // CTC: skip blanks (0) and repeated tokens
            if token_id != 0 && token_id != prev_token {
                tokens.push(token_id);
            }
            prev_token = token_id;
        }

        // Decode tokens using vocabulary mapping
        let text = if tokens.is_empty() {
            String::from("")
        } else {
            tokens
                .iter()
                .filter_map(|&id| {
                    self.token_map.get(&id).and_then(|s| {
                        // Filter out special tokens (those starting with <| and ending with |>)
                        if s.starts_with("<|") && s.ends_with("|>") {
                            None
                        } else {
                            Some(s.as_str())
                        }
                    })
                })
                .collect::<Vec<&str>>()
                .join("")
        };

        // Replace SentencePiece underscore with space and clean up
        let cleaned = text.replace('▁', " ");

        // If we couldn't decode any tokens, show the raw token IDs for debugging
        let result = if cleaned.trim().is_empty() && !tokens.is_empty() {
            format!(
                "[tokens: {}]",
                tokens
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            cleaned
        };

        Ok(result.trim().to_string())
    }

    fn backend_type(&self) -> &'static str {
        "sensevoice"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}

/// Parse am.mvn file to extract mean and standard deviation
fn parse_am_mvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut mean_values = Vec::new();
    let mut std_values = Vec::new();
    let mut section = 0; // 0 = looking for AddShift, 1 = looking for Rescale

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();

        // Look for <LearnRateCoef> lines which contain the actual values
        if trimmed.starts_with("<LearnRateCoef>") && trimmed.contains('[') {
            // Extract the part between [ and ]
            if let Some(start) = trimmed.find('[') {
                if let Some(end) = trimmed.rfind(']') {
                    let values_str = &trimmed[start + 1..end];
                    let numbers: Vec<f32> = values_str
                        .split_whitespace()
                        .filter_map(|s| s.parse::<f32>().ok())
                        .collect();

                    if !numbers.is_empty() {
                        if section == 0 {
                            mean_values = numbers;
                            section = 1; // Now look for std values
                        } else if section == 1 {
                            std_values = numbers;
                            break; // We have both
                        }
                    }
                }
            }
        }
    }

    if mean_values.is_empty() {
        return Err(anyhow!("Failed to parse mean values from am.mvn"));
    }

    // If we didn't find std values, use 1.0 (no scaling)
    if std_values.is_empty() {
        std_values = vec![1.0f32; mean_values.len()];
    }

    // Compute inverse std for faster computation (multiply instead of divide)
    let istd: Vec<f32> = std_values
        .iter()
        .map(|&s| if s != 0.0 { 1.0 / s } else { 1.0 })
        .collect();

    Ok((mean_values, istd))
}

/// Load vocabulary from vocab.json file
fn load_vocabulary(vocab_path: &Path) -> Result<HashMap<i32, String>> {
    let file = File::open(vocab_path)?;
    let vocab: HashMap<String, String> =
        serde_json::from_reader(file).map_err(|e| anyhow!("Failed to parse vocab.json: {}", e))?;

    // Convert String keys to i32
    let mut token_map = HashMap::new();
    for (k, v) in vocab {
        if let Ok(id) = k.parse::<i32>() {
            token_map.insert(id, v);
        }
    }

    log::info!("Loaded {} tokens from vocabulary", token_map.len());
    Ok(token_map)
}

/// Build a basic token-to-character mapping (fallback)
fn build_basic_token_map() -> HashMap<i32, String> {
    let mut map = HashMap::new();

    // Common English lowercase letters (approximate BPE tokens)
    // These are guesses based on typical SentencePiece models
    map.insert(5, " ".to_string());
    map.insert(6, "t".to_string());
    map.insert(7, "h".to_string());
    map.insert(8, "e".to_string());
    map.insert(9, " ".to_string());
    map.insert(10, "i".to_string());
    map.insert(11, "s".to_string());
    map.insert(12, " ".to_string());
    map.insert(13, "a".to_string());
    map.insert(14, " ".to_string());
    map.insert(15, "t".to_string());
    map.insert(16, "e".to_string());
    map.insert(17, "s".to_string());
    map.insert(18, "t".to_string());

    // Add more common characters
    for (i, c) in "abcdefghijklmnopqrstuvwxyz".chars().enumerate() {
        map.insert(100 + i as i32, c.to_string());
    }

    // Add digits
    for (i, c) in "0123456789".chars().enumerate() {
        map.insert(200 + i as i32, c.to_string());
    }

    // Common punctuation
    map.insert(300, ".".to_string());
    map.insert(301, ",".to_string());
    map.insert(302, "!".to_string());
    map.insert(303, "?".to_string());
    map.insert(304, " ".to_string());

    map
}
