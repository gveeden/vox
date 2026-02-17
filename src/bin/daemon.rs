/// CleverNote daemon - handles recording, transcription, and IPC
use chrono::Local;
use clevernote::{
    audio::AudioCapture,
    clipboard, convert_to_mono, get_config_dir,
    ipc::{self, Command, DaemonStatus, Response, SuccessResponse},
    model_downloader, resample_audio,
};
use eyre::{Result, WrapErr};
use log::{error, info, warn};
use parakeet_rs::{ParakeetTDT, TimestampMode, Transcriber};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Deserialize, Serialize)]
struct DaemonConfig {
    #[serde(default = "default_modifier_key")]
    modifier_key: String,
    #[serde(default = "default_trigger_key")]
    trigger_key: String,
    #[serde(default = "default_auto_inject")]
    auto_inject: bool,
}

fn default_modifier_key() -> String {
    "Alt".to_string()
}

fn default_trigger_key() -> String {
    "Space".to_string()
}

fn default_auto_inject() -> bool {
    false
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            modifier_key: default_modifier_key(),
            trigger_key: default_trigger_key(),
            auto_inject: default_auto_inject(),
        }
    }
}

impl DaemonConfig {
    fn load(path: &PathBuf) -> Result<Self> {
        if path.exists() {
            let contents = fs::read_to_string(path)?;
            let config: DaemonConfig = toml::from_str(&contents)?;
            Ok(config)
        } else {
            let config = DaemonConfig::default();
            let toml_string = toml::to_string_pretty(&config)?;
            fs::write(path, toml_string)?;
            Ok(config)
        }
    }
}

#[derive(Serialize, serde::Deserialize, Clone)]
struct TranscriptRecord {
    timestamp: String,
    text: String,
}

#[derive(Serialize, serde::Deserialize)]
struct TranscriptHistory {
    transcripts: Vec<TranscriptRecord>,
}

impl TranscriptHistory {
    fn load_or_new(path: &PathBuf) -> Self {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(history) = serde_json::from_str(&content) {
                    return history;
                }
            }
        }
        Self {
            transcripts: Vec::new(),
        }
    }

    fn add_transcript(&mut self, record: TranscriptRecord) {
        // Add to front (most recent first)
        self.transcripts.insert(0, record);

        // Keep only last 100
        if self.transcripts.len() > 100 {
            self.transcripts.truncate(100);
        }
    }

    fn save(&self, path: &PathBuf) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

struct TranscriptionJob {
    audio_data: Vec<f32>,
    device_sample_rate: u32,
    device_channels: u16,
    target_sample_rate: u32,
    output_dir: PathBuf,
    auto_inject: bool,
}

struct DaemonState {
    model_loaded: AtomicBool,
    recordings_processed: AtomicU64,
    start_time: Instant,
    shutdown: AtomicBool,
    overlay_close_handle: std::sync::Mutex<Option<Arc<AtomicBool>>>,
}

impl DaemonState {
    fn new() -> Self {
        Self {
            model_loaded: AtomicBool::new(false),
            recordings_processed: AtomicU64::new(0),
            start_time: Instant::now(),
            shutdown: AtomicBool::new(false),
            overlay_close_handle: std::sync::Mutex::new(None),
        }
    }

    fn status(&self, audio_capture: &AudioCapture) -> DaemonStatus {
        let is_recording = audio_capture.state.lock().unwrap().is_recording;
        DaemonStatus {
            is_recording,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            model_loaded: self.model_loaded.load(Ordering::Relaxed),
            recordings_processed: self.recordings_processed.load(Ordering::Relaxed),
        }
    }
}

fn main() -> Result<()> {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("CleverNote daemon starting...");

    // Create config directory
    let config_dir = get_config_dir();
    fs::create_dir_all(&config_dir).wrap_err("Failed to create config directory")?;

    // Load config
    let config_path = config_dir.join("config.toml");
    let config = DaemonConfig::load(&config_path)?;
    info!("⚙️  Config loaded: auto_inject={}", config.auto_inject);

    // Change to config directory
    std::env::set_current_dir(&config_dir)?;

    // Default model path
    let model_path = "models/parakeet-tdt";

    // Ensure model exists (download if needed)
    info!("Checking for model...");
    let model_path = model_downloader::ensure_model_exists(model_path, true)?;
    info!("Model found at: {}", model_path.display());

    // Create output directory
    let output_dir = config_dir.join("transcripts");
    fs::create_dir_all(&output_dir)?;

    // Initialize audio capture
    info!("Initializing audio capture...");
    let audio_capture = AudioCapture::new()?;
    let device_sample_rate = audio_capture.device_sample_rate();

    // Start audio stream (paused initially)
    audio_capture.start()?;
    thread::sleep(Duration::from_millis(100)); // Warm up
    audio_capture.pause()?;

    info!("Audio capture initialized");

    // Create transcription channel
    let (tx, rx) = std::sync::mpsc::channel::<TranscriptionJob>();

    // Daemon state
    let daemon_state = Arc::new(DaemonState::new());
    let daemon_state_clone = daemon_state.clone();

    // Spawn transcription thread
    info!("Loading transcription model...");
    thread::spawn(move || {
        if let Err(e) = transcription_worker(rx, model_path, daemon_state_clone) {
            error!("Transcription worker error: {}", e);
        }
    });

    // Wait for model to load
    for _ in 0..100 {
        if daemon_state.model_loaded.load(Ordering::Relaxed) {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }

    info!("Model loaded successfully");

    // Setup Unix socket server
    let socket_path = ipc::socket_path();

    // Remove old socket if it exists
    if socket_path.exists() {
        fs::remove_file(&socket_path)?;
    }

    // Ensure parent directory exists
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }

    info!("Starting IPC server at: {}", socket_path.display());
    let listener = UnixListener::bind(&socket_path)?;

    info!("✅ CleverNote daemon ready!");
    info!("Connect with: clevernote toggle");

    // Accept connections (no threading - keep stream in main thread)
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(e) = handle_client(
                    stream,
                    &audio_capture,
                    &tx,
                    &daemon_state,
                    &output_dir,
                    device_sample_rate,
                    config.auto_inject,
                ) {
                    error!("Error handling client: {}", e);
                }
            }
            Err(e) => {
                error!("Connection failed: {}", e);
            }
        }

        // Check for shutdown signal
        if daemon_state.shutdown.load(Ordering::Relaxed) {
            info!("Shutting down daemon...");
            break;
        }
    }

    // Cleanup
    if socket_path.exists() {
        fs::remove_file(&socket_path)?;
    }

    info!("Daemon stopped");
    Ok(())
}

fn handle_client(
    mut stream: UnixStream,
    audio_capture: &AudioCapture,
    tx: &std::sync::mpsc::Sender<TranscriptionJob>,
    daemon_state: &Arc<DaemonState>,
    output_dir: &PathBuf,
    device_sample_rate: u32,
    auto_inject: bool,
) -> Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut line = String::new();

    reader.read_line(&mut line)?;

    let command: Command = ipc::deserialize_command(line.trim().as_bytes())
        .wrap_err("Failed to deserialize command")?;

    let response = match command {
        Command::Toggle => handle_toggle(
            audio_capture,
            tx,
            output_dir,
            device_sample_rate,
            auto_inject,
            daemon_state,
        ),
        Command::Start => handle_start(audio_capture, daemon_state),
        Command::Stop => handle_stop(
            audio_capture,
            tx,
            output_dir,
            device_sample_rate,
            auto_inject,
            daemon_state,
        ),
        Command::Status => {
            let status = daemon_state.status(&audio_capture);
            Response::Success(SuccessResponse::Status(status))
        }
        Command::Quit => {
            daemon_state.shutdown.store(true, Ordering::Relaxed);
            Response::Success(SuccessResponse::Shutdown)
        }
    };

    // Send response
    let response_bytes = ipc::serialize_response(&response)?;
    stream.write_all(&response_bytes)?;
    stream.flush()?;

    Ok(())
}

fn handle_start(audio_capture: &AudioCapture, daemon_state: &Arc<DaemonState>) -> Response {
    let mut state = audio_capture.state.lock().unwrap();

    if state.is_recording {
        return Response::Success(SuccessResponse::AlreadyRecording);
    }

    state.start_recording();
    drop(state);

    if let Err(e) = audio_capture.start() {
        return Response::Error(format!("Failed to start audio stream: {}", e));
    }

    info!("🎙️  Recording started");

    // Show overlay window and store close handle
    let close_handle = clevernote::recording_overlay::show_recording_overlay();
    *daemon_state.overlay_close_handle.lock().unwrap() = Some(close_handle);

    Response::Success(SuccessResponse::RecordingStarted)
}

fn handle_stop(
    audio_capture: &AudioCapture,
    tx: &std::sync::mpsc::Sender<TranscriptionJob>,
    output_dir: &PathBuf,
    device_sample_rate: u32,
    auto_inject: bool,
    daemon_state: &Arc<DaemonState>,
) -> Response {
    let mut state = audio_capture.state.lock().unwrap();

    if !state.is_recording {
        return Response::Success(SuccessResponse::NotRecording);
    }

    let audio_data = state.stop_recording();
    drop(state);

    if let Err(e) = audio_capture.pause() {
        return Response::Error(format!("Failed to pause audio stream: {}", e));
    }

    info!("⏹️  Recording stopped - {} samples", audio_data.len());

    // Hide overlay window
    info!("handle_stop: About to hide overlay");
    if let Some(close_handle) = daemon_state.overlay_close_handle.lock().unwrap().take() {
        clevernote::recording_overlay::hide_recording_overlay(close_handle);
    }
    info!("handle_stop: Overlay hidden, continuing...");

    // Send to transcription thread
    let device_channels = audio_capture.channels();
    let job = TranscriptionJob {
        audio_data,
        device_sample_rate,
        device_channels,
        target_sample_rate: 16000, // Parakeet expects 16kHz
        output_dir: output_dir.clone(),
        auto_inject,
    };

    info!("handle_stop: Sending job to transcription thread");
    if let Err(e) = tx.send(job) {
        return Response::Error(format!("Failed to queue transcription: {}", e));
    }
    info!("handle_stop: Job sent, returning response");

    Response::Success(SuccessResponse::RecordingStopped)
}

fn handle_toggle(
    audio_capture: &AudioCapture,
    tx: &std::sync::mpsc::Sender<TranscriptionJob>,
    output_dir: &PathBuf,
    device_sample_rate: u32,
    auto_inject: bool,
    daemon_state: &Arc<DaemonState>,
) -> Response {
    let state = audio_capture.state.lock().unwrap();
    let is_recording = state.is_recording;
    drop(state);

    if !is_recording {
        handle_start(audio_capture, daemon_state)
    } else {
        handle_stop(
            audio_capture,
            tx,
            output_dir,
            device_sample_rate,
            auto_inject,
            daemon_state,
        )
    }
}

fn transcription_worker(
    rx: std::sync::mpsc::Receiver<TranscriptionJob>,
    model_path: PathBuf,
    daemon_state: Arc<DaemonState>,
) -> Result<()> {
    // Load model once
    info!("Loading Parakeet TDT model from: {}", model_path.display());
    let mut parakeet = ParakeetTDT::from_pretrained(&model_path, None)?;
    daemon_state.model_loaded.store(true, Ordering::Relaxed);
    info!("Model loaded and ready");

    // Process jobs
    while let Ok(job) = rx.recv() {
        if let Err(e) = process_transcription(&mut parakeet, job, &daemon_state) {
            error!("Transcription error: {}", e);
        }
    }

    Ok(())
}

fn process_transcription(
    parakeet: &mut ParakeetTDT,
    job: TranscriptionJob,
    daemon_state: &Arc<DaemonState>,
) -> Result<()> {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Convert to mono if needed (Parakeet expects mono)
    info!(
        "Converting audio: {} channels -> 1 channel (mono)",
        job.device_channels
    );
    let mono_audio = convert_to_mono(&job.audio_data, job.device_channels);

    // Resample if needed (Parakeet expects 16kHz)
    info!(
        "Resampling audio: {}Hz -> {}Hz",
        job.device_sample_rate, job.target_sample_rate
    );
    let resampled_audio =
        resample_audio(&mono_audio, job.device_sample_rate, job.target_sample_rate);

    // Transcribe
    info!("🤖 Transcribing...");
    let start = Instant::now();

    let result = parakeet.transcribe_samples(
        resampled_audio,
        job.target_sample_rate,
        1, // mono
        Some(TimestampMode::Sentences),
    )?;

    let elapsed = start.elapsed();
    info!("✅ Transcription complete in {:.2}s", elapsed.as_secs_f32());
    info!("📝 Text: {}", result.text);

    // Add to history.json (keep last 100)
    let history_path = job.output_dir.join("history.json");
    let mut history = TranscriptHistory::load_or_new(&history_path);

    let record = TranscriptRecord {
        timestamp: timestamp.clone(),
        text: result.text.clone(),
    };

    history.add_transcript(record);
    history.save(&history_path)?;
    info!(
        "💾 Updated history.json ({} total)",
        history.transcripts.len()
    );

    // Copy to clipboard and optionally inject
    if let Err(e) = clipboard::copy_and_paste(&result.text, job.auto_inject) {
        warn!("Failed to copy: {}", e);
    }

    // Increment counter
    daemon_state
        .recordings_processed
        .fetch_add(1, Ordering::Relaxed);

    Ok(())
}
