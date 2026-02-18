use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, ArrayView1};

/// Compute mel spectrogram from audio samples
///
/// # Arguments
/// * `audio` - Audio samples (mono, f32)
/// * `sample_rate` - Sample rate of audio (typically 16000)
/// * `n_fft` - FFT size
/// * `hop_length` - Hop length between frames
/// * `n_mels` - Number of mel frequency bins
/// * `f_min` - Minimum frequency
/// * `f_max` - Maximum frequency
///
/// # Returns
/// Mel spectrogram as 2D array [n_mels, n_frames]
pub fn compute_mel_spectrogram(
    audio: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Result<Array2<f32>> {
    // STFT
    let stft = compute_stft(audio, n_fft, hop_length)?;

    // Compute power spectrogram
    let n_frames = stft.len_of(ndarray::Axis(1));
    let n_freq_bins = n_fft / 2 + 1;
    let mut power_spec = Array2::zeros((n_freq_bins, n_frames));

    for (i_frame, frame) in stft.axis_iter(ndarray::Axis(1)).enumerate() {
        // Only use the first n_fft/2 + 1 bins (positive frequencies)
        for (i_freq, &complex) in frame.iter().enumerate().take(n_freq_bins) {
            let power = complex.norm_sqr();
            power_spec[[i_freq, i_frame]] = power;
        }
    }

    // Create mel filterbank
    let mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)?;

    // Apply mel filterbank
    let mel_spec = mel_fb.dot(&power_spec);

    // Convert to log scale (add small epsilon to avoid log(0))
    let log_mel_spec = mel_spec.mapv(|x| x.max(1e-10).log10());

    Ok(log_mel_spec)
}

/// Compute Short-Time Fourier Transform
fn compute_stft(
    audio: &[f32],
    n_fft: usize,
    hop_length: usize,
) -> Result<Array2<num_complex::Complex<f32>>> {
    use num_complex::Complex;

    let n_samples = audio.len();
    let n_frames = (n_samples + hop_length - 1) / hop_length;
    let mut stft = Array2::zeros((n_fft, n_frames));

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos())
        .collect();

    for i_frame in 0..n_frames {
        let start = i_frame * hop_length;
        let end = (start + n_fft).min(n_samples);

        // Extract frame and apply window
        let mut frame = vec![0.0f32; n_fft];
        for (i, &sample) in audio[start..end].iter().enumerate() {
            frame[i] = sample * window[i];
        }

        // Compute FFT
        let fft_result = compute_fft(&frame)?;

        for (i, &value) in fft_result.iter().enumerate() {
            stft[[i, i_frame]] = value;
        }
    }

    Ok(stft)
}

/// Compute FFT using a simple DFT implementation
/// For production, consider using rustfft crate
fn compute_fft(signal: &[f32]) -> Result<Vec<num_complex::Complex<f32>>> {
    use num_complex::Complex;

    let n = signal.len();
    let mut result = vec![Complex::new(0.0, 0.0); n];

    for k in 0..n {
        let mut sum = Complex::new(0.0, 0.0);
        for (n_idx, &sample) in signal.iter().enumerate() {
            let angle = -2.0 * std::f32::consts::PI * (k as f32) * (n_idx as f32) / (n as f32);
            sum += Complex::new(sample, 0.0) * Complex::new(angle.cos(), angle.sin());
        }
        result[k] = sum;
    }

    Ok(result)
}

/// Create mel filterbank
fn create_mel_filterbank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Result<Array2<f32>> {
    let sr = sample_rate as f32;

    // Convert frequencies to mel scale
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Create mel points
    let mels: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * (i as f32) / ((n_mels + 1) as f32))
        .collect();

    // Convert back to Hz
    let freqs: Vec<f32> = mels.iter().map(|&m| mel_to_hz(m)).collect();

    // Create filterbank
    let mut filterbank = Array2::zeros((n_mels, n_fft / 2 + 1));
    let fft_freqs: Vec<f32> = (0..=n_fft / 2)
        .map(|i| i as f32 * sr / n_fft as f32)
        .collect();

    for i_mel in 0..n_mels {
        let f_left = freqs[i_mel];
        let f_center = freqs[i_mel + 1];
        let f_right = freqs[i_mel + 2];

        for (i_fft, &fft_freq) in fft_freqs.iter().enumerate() {
            let weight = if fft_freq >= f_left && fft_freq <= f_center {
                (fft_freq - f_left) / (f_center - f_left)
            } else if fft_freq > f_center && fft_freq <= f_right {
                (f_right - fft_freq) / (f_right - f_center)
            } else {
                0.0
            };
            filterbank[[i_mel, i_fft]] = weight;
        }
    }

    Ok(filterbank)
}

/// Convert Hz to Mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel to Hz
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Pad or truncate audio to target length
///
/// # Arguments
/// * `audio` - Input audio samples
/// * `target_samples` - Target number of samples
/// * `pad_mode` - "zero" for zero padding, "repeat" for repeating
pub fn pad_audio(audio: &[f32], target_samples: usize, pad_mode: &str) -> Vec<f32> {
    if audio.len() >= target_samples {
        audio[..target_samples].to_vec()
    } else {
        let mut padded = audio.to_vec();
        match pad_mode {
            "zero" => {
                padded.resize(target_samples, 0.0);
            }
            "repeat" => {
                while padded.len() < target_samples {
                    padded.extend_from_slice(audio);
                }
                padded.truncate(target_samples);
            }
            _ => {
                padded.resize(target_samples, 0.0);
            }
        }
        padded
    }
}

/// Resample audio using linear interpolation
pub fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return audio.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let output_len = (audio.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f64;

        let sample = audio[src_idx_floor] * (1.0 - frac as f32) + audio[src_idx_ceil] * frac as f32;
        resampled.push(sample);
    }

    resampled
}

/// Convert stereo to mono by averaging channels
pub fn stereo_to_mono(audio: &[f32]) -> Vec<f32> {
    audio
        .chunks(2)
        .map(|chunk| {
            if chunk.len() == 2 {
                (chunk[0] + chunk[1]) / 2.0
            } else {
                chunk[0]
            }
        })
        .collect()
}

/// Compute filter bank (fbank) features for SenseVoice
///
/// Similar to mel spectrogram but with different normalization
///
/// # Arguments
/// * `audio` - Audio samples (mono, f32)
/// * `sample_rate` - Sample rate (typically 16000)
/// * `n_fft` - FFT size (default 512)
/// * `hop_length` - Hop length (default 160)
/// * `n_mels` - Number of mel bins (default 80)
///
/// # Returns
/// Fbank features as 2D array [n_frames, n_mels]
pub fn compute_fbank_features(
    audio: &[f32],
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
) -> Result<Array2<f32>> {
    // STFT
    let stft = compute_stft(audio, n_fft, hop_length)?;

    // Compute power spectrogram
    let n_frames = stft.len_of(ndarray::Axis(1));
    let n_freq_bins = n_fft / 2 + 1;
    let mut power_spec = Array2::zeros((n_freq_bins, n_frames));

    for (i_frame, frame) in stft.axis_iter(ndarray::Axis(1)).enumerate() {
        for (i_freq, &complex) in frame.iter().enumerate().take(n_freq_bins) {
            let power = complex.norm_sqr();
            power_spec[[i_freq, i_frame]] = power;
        }
    }

    // Create mel filterbank
    let f_min = 20.0;
    let f_max = (sample_rate / 2) as f32;
    let mel_fb = create_mel_filterbank(sample_rate, n_fft, n_mels, f_min, f_max)?;

    // Apply mel filterbank: [n_mels, n_freq] @ [n_freq, n_frames] = [n_mels, n_frames]
    let mel_spec = mel_fb.dot(&power_spec);

    // Convert to log scale (add epsilon to avoid log(0))
    let log_mel_spec = mel_spec.mapv(|x| (x.max(1e-10)).ln());

    // Transpose to [n_frames, n_mels] for SenseVoice
    Ok(log_mel_spec.t().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_spectrogram() {
        // Create a simple test signal
        let sample_rate = 16000;
        let duration = 1.0; // 1 second
        let n_samples = (sample_rate as f32 * duration) as usize;
        let audio: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let mel_spec = compute_mel_spectrogram(
            &audio,
            sample_rate,
            400,    // n_fft
            160,    // hop_length
            80,     // n_mels
            0.0,    // f_min
            8000.0, // f_max
        )
        .unwrap();

        assert_eq!(mel_spec.shape()[0], 80); // n_mels
        assert!(mel_spec.shape()[1] > 0); // n_frames
    }

    #[test]
    fn test_hz_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);

        assert!((hz - hz_back).abs() < 0.1);
    }
}
