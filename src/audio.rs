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
    pub fn new(input_device: Option<&str>) -> Result<Self> {
        let host = cpal::default_host();

        let device = if let Some(device_name) = input_device {
            let normalized_device_name = device_name.trim();
            if normalized_device_name.is_empty() {
                return Err(eyre::eyre!("Input device name cannot be empty"));
            }

            let mut matching_device: Option<Device> = None;
            let mut available_devices: Vec<String> = Vec::new();

            for device in host
                .input_devices()
                .map_err(|e| eyre::eyre!("Failed to enumerate input devices: {}", e))?
            {
                let name = device.name().unwrap_or_else(|_| "<unknown>".to_string());
                available_devices.push(name.clone());

                if name == normalized_device_name
                    || name.eq_ignore_ascii_case(normalized_device_name)
                {
                    matching_device = Some(device);
                    break;
                }
            }

            match matching_device {
                Some(device) => device,
                None => {
                    let available = if available_devices.is_empty() {
                        "(none)".to_string()
                    } else {
                        available_devices.join(", ")
                    };
                    return Err(eyre::eyre!(
                        "Input device '{}' not found. Available devices: {}",
                        normalized_device_name,
                        available
                    ));
                }
            }
        } else {
            host.default_input_device()
                .ok_or_else(|| eyre::eyre!("No input device available"))?
        };

        log::info!("Selected input device: {}", device.name()?);

        let config = device.default_input_config()?;
        log::info!("Default input config: {:?}", config);

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
