use anyhow::Result;

/// Trait for transcription backends
pub trait TranscriptionBackend: Send {
    /// Transcribe audio to text
    ///
    /// # Arguments
    /// * `audio` - Audio samples as f32 array (should be mono, 16kHz for most models)
    /// * `sample_rate` - Sample rate of the audio
    ///
    /// # Returns
    /// Transcribed text
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String>;

    /// Get the backend type identifier
    fn backend_type(&self) -> &'static str;

    /// Get the model name
    fn model_name(&self) -> &str;
}

pub mod audio;
pub mod factory;
pub mod moonshine;
pub mod parakeet;
pub mod sensevoice;
pub mod tokenizers;
pub mod whisper;
pub mod whisper_decoder_layers;

pub use factory::create_backend;
