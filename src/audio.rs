/// Audio capture management
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Stream, StreamConfig};
use eyre::{Result, WrapErr};
use std::sync::{Arc, Mutex};

pub struct RecordingState {
    pub is_recording: bool,
    pub audio_data: Vec<f32>,
    pub device_sample_rate: u32,
    pub channels: u16,
}

impl RecordingState {
    pub fn new(device_sample_rate: u32, channels: u16) -> Self {
        Self {
            is_recording: false,
            audio_data: Vec::new(),
            device_sample_rate,
            channels,
        }
    }

    pub fn start_recording(&mut self) {
        self.is_recording = true;
        self.audio_data.clear();
    }

    pub fn stop_recording(&mut self) -> Vec<f32> {
        self.is_recording = false;
        let data = self.audio_data.clone();
        self.audio_data.clear();
        data
    }
}

pub struct AudioCapture {
    pub device: Device,
    pub config: StreamConfig,
    pub stream: Stream,
    pub state: Arc<Mutex<RecordingState>>,
}

impl AudioCapture {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| eyre::eyre!("No input device available"))?;

        println!("Input device: {}", device.name()?);

        let config = device.default_input_config()?;
        println!("Default input config: {:?}", config);

        let device_sample_rate = config.sample_rate().0;
        let channels = config.channels();
        let stream_config: StreamConfig = config.into();

        let state = Arc::new(Mutex::new(RecordingState::new(
            device_sample_rate,
            channels,
        )));
        let state_clone = state.clone();

        let err_fn = |err| eprintln!("An error occurred on the audio stream: {}", err);

        // Create audio stream that captures when is_recording is true
        let stream = device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock().unwrap();
                    if state.is_recording {
                        state.audio_data.extend_from_slice(data);
                    }
                },
                err_fn,
                None,
            )
            .wrap_err("Failed to build input stream")?;

        Ok(Self {
            device,
            config: stream_config,
            stream,
            state,
        })
    }

    pub fn start(&self) -> Result<()> {
        self.stream.play()?;
        Ok(())
    }

    pub fn pause(&self) -> Result<()> {
        self.stream.pause()?;
        Ok(())
    }

    pub fn device_sample_rate(&self) -> u32 {
        self.state.lock().unwrap().device_sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.state.lock().unwrap().channels
    }
}
