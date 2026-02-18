use anyhow::Result;
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use std::path::Path;

use super::{audio, TranscriptionBackend};

pub struct ParakeetBackend {
    model: ParakeetTDT,
    model_name: String,
}

impl ParakeetBackend {
    pub fn new<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let path = model_path.as_ref();
        let model = ParakeetTDT::from_pretrained(path, None)?;

        // Extract model name from path
        let model_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("parakeet")
            .to_string();

        Ok(Self { model, model_name })
    }
}

impl TranscriptionBackend for ParakeetBackend {
    fn transcribe(&mut self, audio: &[f32], sample_rate: u32) -> Result<String> {
        // Parakeet expects 16kHz mono audio
        const EXPECTED_SAMPLE_RATE: u32 = 16000;

        let audio = if sample_rate != EXPECTED_SAMPLE_RATE {
            log::debug!(
                "Resampling audio from {}Hz to {}Hz",
                sample_rate,
                EXPECTED_SAMPLE_RATE
            );
            audio::resample_audio(audio, sample_rate, EXPECTED_SAMPLE_RATE)
        } else {
            audio.to_vec()
        };

        let result = self.model.transcribe_samples(
            audio,
            EXPECTED_SAMPLE_RATE,
            1, // mono
            Some(TimestampMode::Sentences),
        )?;

        Ok(result.text)
    }

    fn backend_type(&self) -> &'static str {
        "parakeet"
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }
}
