pub mod audio;
pub mod clipboard;
#[cfg(target_os = "linux")]
pub mod inject_linux;
pub mod ipc;
pub mod model_downloader;
pub mod models;
pub mod progress_window;
pub mod recording_overlay;

use std::path::PathBuf;

/// Get the CleverNote config directory (~/.config/clevernote/)
pub fn get_config_dir() -> PathBuf {
    // Use XDG_CONFIG_HOME if set, otherwise default to ~/.config
    let config_home = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join(".config")
            .to_string_lossy()
            .to_string()
    });
    PathBuf::from(config_home).join("clevernote")
}

/// Get the models directory (~/.config/clevernote/models/)
pub fn get_models_dir() -> PathBuf {
    get_config_dir().join("models")
}

/// Convert stereo (or multi-channel) audio to mono by averaging channels
pub fn convert_to_mono(audio: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return audio.to_vec();
    }

    let channels = channels as usize;
    audio
        .chunks(channels)
        .map(|chunk| {
            // Average all channels
            let sum: f32 = chunk.iter().sum();
            sum / chunk.len() as f32
        })
        .collect()
}

/// Resample audio from source sample rate to target sample rate using linear interpolation
pub fn resample_audio(audio: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return audio.to_vec();
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = (audio.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f64;

        // Linear interpolation
        let sample = audio[src_idx_floor] * (1.0 - frac as f32) + audio[src_idx_ceil] * frac as f32;
        resampled.push(sample);
    }

    resampled
}
