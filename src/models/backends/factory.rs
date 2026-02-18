use anyhow::{anyhow, Result};
use std::path::Path;

use super::TranscriptionBackend;

/// Create a transcription backend based on model type
///
/// # Arguments
/// * `model_path` - Path to the model directory
/// * `backend_type` - Type of backend ("parakeet", "whisper", "moonshine", "sensevoice")
///
/// # Returns
/// Boxed backend implementing TranscriptionBackend trait
pub fn create_backend<P: AsRef<Path>>(
    model_path: P,
    backend_type: &str,
) -> Result<Box<dyn TranscriptionBackend>> {
    let path = model_path.as_ref();

    match backend_type {
        "parakeet" | "parakeet-tdt" => {
            use super::parakeet::ParakeetBackend;
            Ok(Box::new(ParakeetBackend::new(path)?))
        }
        "whisper" => {
            use super::whisper::WhisperBackend;
            Ok(Box::new(WhisperBackend::new(path)?))
        }
        "moonshine" => {
            use super::moonshine::MoonshineBackend;
            Ok(Box::new(MoonshineBackend::new(path)?))
        }
        "sensevoice" => {
            use super::sensevoice::SenseVoiceBackend;
            Ok(Box::new(SenseVoiceBackend::new(path)?))
        }
        _ => Err(anyhow!(
            "Unknown backend type: {}. Supported: parakeet, whisper, moonshine, sensevoice",
            backend_type
        )),
    }
}

/// Detect backend type from model directory
///
/// Checks for characteristic files to determine the model type
pub fn detect_backend_type<P: AsRef<Path>>(model_path: P) -> Result<String> {
    let path = model_path.as_ref();

    // Check for Parakeet (has decoder_joint.onnx and vocab.txt)
    if path.join("decoder_joint.onnx").exists() || path.join("decoder_joint-int8.onnx").exists() {
        if path.join("vocab.txt").exists() {
            return Ok("parakeet".to_string());
        }
    }

    // Check for Whisper (has encoder.onnx, decoder.onnx, vocab.json)
    if (path.join("encoder.onnx").exists() || path.join("encoder_model.onnx").exists())
        && (path.join("decoder.onnx").exists() || path.join("decoder_model_merged.onnx").exists())
        && path.join("vocab.json").exists()
    {
        return Ok("whisper".to_string());
    }

    // Check for Moonshine (has encoder.onnx, decoder.onnx, tokenizer.json)
    if (path.join("encoder.onnx").exists() || path.join("encoder_model.onnx").exists())
        && (path.join("decoder.onnx").exists() || path.join("decoder_model_merged.onnx").exists())
        && path.join("tokenizer.json").exists()
    {
        return Ok("moonshine".to_string());
    }

    // Check for SenseVoice (has model.onnx, tokenizer.model)
    if path.join("model.onnx").exists() && path.join("tokenizer.model").exists() {
        return Ok("sensevoice".to_string());
    }

    Err(anyhow!(
        "Could not detect model type from directory: {:?}. Please ensure all model files are present.",
        path
    ))
}
