pub mod audio;
pub mod clipboard;
#[cfg(target_os = "linux")]
pub mod inject_linux;
pub mod ipc;
pub mod model_downloader;
pub mod models;
pub mod progress_window;
pub mod recording_overlay;

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::{FormatOptions, Packet};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Get the Vox config directory (~/.config/vox/)
pub fn get_config_dir() -> PathBuf {
    // Use XDG_CONFIG_HOME if set, otherwise default to ~/.config
    let config_home = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join(".config")
            .to_string_lossy()
            .to_string()
    });
    PathBuf::from(config_home).join("vox")
}

/// Get the models directory (~/.config/vox/models/)
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

/// Decode an audio file to a Vec<f32> of mono samples at the specified sample rate
pub fn decode_audio_file(
    path: &std::path::Path,
    target_sample_rate: u32,
) -> anyhow::Result<Vec<f32>> {
    // Open the media source.
    let src = std::fs::File::open(path)?;

    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Create a hint to help the format reader guess what format the source is and provide the extension.
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        hint.with_extension(ext);
    }

    // Use the default options for metadata and format readers.
    let meta_opts = MetadataOptions::default();
    let fmt_opts = FormatOptions::default();

    // Probe the media source.
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    // Find the first audio track with a known codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow::anyhow!("no supported audio track found"))?;

    // Use the default options for the decoder.
    let dec_opts = DecoderOptions::default();

    // Create a decoder for the track.
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    let track_id = track.id;
    let source_sample_rate = track.codec_params.sample_rate.unwrap_or(44100);

    let mut samples: Vec<f32> = Vec::new();

    // The decode loop.
    loop {
        // Get the next packet from the format reader.
        let packet: Packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break, // End of stream
            Err(e) => return Err(anyhow::anyhow!(e)),
        };

        // If the packet does not belong to the selected track, skip it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(decoded) => {
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        samples.extend_from_slice(buf.chan(0));
                    }
                    AudioBufferRef::U8(buf) => {
                        for &s in buf.chan(0) {
                            samples.push((s as f32 - 128.0) / 128.0);
                        }
                    }
                    AudioBufferRef::U16(buf) => {
                        for &s in buf.chan(0) {
                            samples.push((s as f32 - 32768.0) / 32768.0);
                        }
                    }
                    AudioBufferRef::U32(buf) => {
                        for &s in buf.chan(0) {
                            samples.push((s as f32 - 2147483648.0) / 2147483648.0);
                        }
                    }
                    AudioBufferRef::S8(buf) => {
                        for &s in buf.chan(0) {
                            samples.push(s as f32 / 128.0);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        for &s in buf.chan(0) {
                            samples.push(s as f32 / 32768.0);
                        }
                    }
                    AudioBufferRef::S32(buf) => {
                        for &s in buf.chan(0) {
                            samples.push(s as f32 / 2147483648.0);
                        }
                    }
                    _ => {
                        // Skip 24-bit and other complex formats for now to ensure compilation
                        // In practice, we can add more as needed.
                    }
                }
            }
            Err(Error::IoError(_)) => break,
            Err(Error::DecodeError(_)) => continue,
            Err(e) => return Err(anyhow::anyhow!(e)),
        }
    }

    // Resample if needed
    if source_sample_rate != target_sample_rate {
        Ok(resample_audio(
            &samples,
            source_sample_rate,
            target_sample_rate,
        ))
    } else {
        Ok(samples)
    }
}

#[no_mangle]
pub extern "C" fn vox_transcribe(
    audio_path_ptr: *const c_char,
    model_path_ptr: *const c_char,
) -> *mut c_char {
    if audio_path_ptr.is_null() || model_path_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let audio_path_str = unsafe { CStr::from_ptr(audio_path_ptr) }.to_string_lossy();
    let model_path_str = unsafe { CStr::from_ptr(model_path_ptr) }.to_string_lossy();

    let audio_path = Path::new(audio_path_str.as_ref());
    let model_path = Path::new(model_path_str.as_ref());

    // 1. Decode audio
    // Target sample rate for Parakeet is 16000
    let samples = match decode_audio_file(audio_path, 16000) {
        Ok(s) => s,
        Err(e) => {
            return CString::new(format!("Error decoding audio: {}", e))
                .unwrap()
                .into_raw()
        }
    };

    // 2. Load model (ParakeetTDT as default for multilingual)
    use parakeet_rs::{ParakeetTDT, Transcriber};
    let mut parakeet = match ParakeetTDT::from_pretrained(model_path, None) {
        Ok(p) => p,
        Err(e) => {
            return CString::new(format!("Error loading model from {:?}: {}", model_path, e))
                .unwrap()
                .into_raw()
        }
    };

    // 3. Transcribe
    let result = match parakeet.transcribe_samples(samples, 16000, 1, None) {
        Ok(r) => r.text,
        Err(e) => format!("Error transcribing: {}", e),
    };

    CString::new(result).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn vox_generate(
    prompt_ptr: *const c_char,
    model_path_ptr: *const c_char,
) -> *mut c_char {
    if prompt_ptr.is_null() || model_path_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let prompt = unsafe { CStr::from_ptr(prompt_ptr) }.to_string_lossy().to_string();
    let model_path_str = unsafe { CStr::from_ptr(model_path_ptr) }.to_string_lossy().to_string();
    let model_path = Path::new(&model_path_str);

    use models::backends::granite::GraniteBackend;

    let result = (|| -> anyhow::Result<String> {
        let mut backend = GraniteBackend::new(model_path, 2)?;
        backend.generate(&prompt)
    })();

    match result {
        Ok(text) => CString::new(text).unwrap_or_default().into_raw(),
        Err(e) => CString::new(format!("Error: {}", e)).unwrap_or_default().into_raw(),
    }
}

#[no_mangle]
pub extern "C" fn vox_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}
