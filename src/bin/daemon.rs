/// CleverNote daemon - handles recording, transcription, and IPC
use chrono::Local;
use clap::Parser;
use clevernote::{
    audio::AudioCapture,
    clipboard, convert_to_mono, get_config_dir,
    ipc::{self, Command, DaemonStatus, Response, SuccessResponse},
    model_downloader,
};
use eyre::{Result, WrapErr};
use log::{error, info, warn};
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

#[derive(Parser, Debug)]
#[command(author, version, about = "CleverNote daemon", long_about = None)]
struct Args {
    /// Input device name to use for recording (runtime-only override)
    #[arg(long)]
    device: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct DaemonConfig {
    #[serde(default = "default_modifier_key")]
    modifier_key: String,
    #[serde(default = "default_trigger_key")]
    trigger_key: String,
    #[serde(default = "default_auto_paste")]
    auto_paste: bool,
    #[serde(default = "default_auto_inject")]
    auto_inject: bool,
    #[serde(default = "default_active_model")]
    active_model: String,
    #[serde(default = "default_input_device")]
    input_device: Option<String>,
}

fn default_modifier_key() -> String {
    "Alt".to_string()
}

fn default_trigger_key() -> String {
    "Space".to_string()
}

fn default_auto_paste() -> bool {
    true
}

fn default_auto_inject() -> bool {
    false
}

fn default_active_model() -> String {
    "parakeet-tdt-0.6b-v3-int8".to_string()
}

fn default_input_device() -> Option<String> {
    None
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            modifier_key: default_modifier_key(),
            trigger_key: default_trigger_key(),
            auto_paste: default_auto_paste(),
            auto_inject: default_auto_inject(),
            active_model: default_active_model(),
            input_device: default_input_device(),
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

    fn save(&self, path: &PathBuf) -> Result<()> {
        let toml_string = toml::to_string_pretty(&self)?;
        fs::write(path, toml_string)?;
        Ok(())
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
    output_dir: PathBuf,
    auto_paste: bool,
    auto_inject: bool,
}

struct DaemonState {
    model_loaded: AtomicBool,
    recordings_processed: AtomicU64,
    start_time: Instant,
    shutdown: AtomicBool,
    overlay_close_handle: std::sync::Mutex<Option<Arc<AtomicBool>>>,
    active_model: std::sync::Mutex<String>,
    config_path: PathBuf,
}

impl DaemonState {
    fn new(active_model: String, config_path: PathBuf) -> Self {
        Self {
            model_loaded: AtomicBool::new(false),
            recordings_processed: AtomicU64::new(0),
            start_time: Instant::now(),
            shutdown: AtomicBool::new(false),
            overlay_close_handle: std::sync::Mutex::new(None),
            active_model: std::sync::Mutex::new(active_model),
            config_path,
        }
    }

    fn status(&self, audio_capture: &AudioCapture) -> DaemonStatus {
        let is_recording = audio_capture.state.lock().unwrap().is_recording;
        let active_model = self.active_model.lock().unwrap().clone();
        DaemonStatus {
            is_recording,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            model_loaded: self.model_loaded.load(Ordering::Relaxed),
            recordings_processed: self.recordings_processed.load(Ordering::Relaxed),
            active_model: Some(active_model),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

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
    info!(
        "⚙️  Config loaded: auto_paste={}, auto_inject={}, active_model={}",
        config.auto_paste, config.auto_inject, config.active_model
    );

    let selected_input_device = if let Some(device) = args.device {
        let normalized_device = device.trim().to_string();
        if normalized_device.is_empty() {
            return Err(eyre::eyre!("--device cannot be empty"));
        }
        Some(normalized_device)
    } else if let Some(configured_device) = config.input_device.clone() {
        let normalized_device = configured_device.trim().to_string();
        if normalized_device.is_empty() {
            None
        } else {
            Some(normalized_device)
        }
    } else {
        None
    };

    // Get models directory
    let models_dir = clevernote::get_models_dir();
    fs::create_dir_all(&models_dir)?;

    // Copy default model registry to config dir if it doesn't exist
    // This allows users to edit models.json at runtime
    match clevernote::models::ModelRegistry::copy_default_to_config() {
        Ok(path) => info!("Model registry available at: {:?}", path),
        Err(e) => warn!("Failed to copy model registry: {}", e),
    }

    // Ensure active model is downloaded
    info!("Checking for model '{}'...", config.active_model);
    let model_path =
        match clevernote::models::ensure_model_downloaded(&config.active_model, &models_dir, false)
        {
            Ok(path) => path,
            Err(e) => {
                return Err(eyre::eyre!("Failed to ensure model is downloaded: {}", e));
            }
        };
    info!("Model found at: {}", model_path.display());

    // Create output directory
    let output_dir = config_dir.join("transcripts");
    fs::create_dir_all(&output_dir)?;

    // Initialize audio capture
    info!("Initializing audio capture...");
    let audio_capture = AudioCapture::new(selected_input_device.as_deref())?;
    let device_sample_rate = audio_capture.device_sample_rate();

    // Start audio stream (paused initially)
    audio_capture.start()?;
    thread::sleep(Duration::from_millis(100)); // Warm up
    audio_capture.pause()?;

    info!("Audio capture initialized");

    // Create transcription channel
    let (tx, rx) = std::sync::mpsc::channel::<TranscriptionJob>();

    // Daemon state
    let daemon_state = Arc::new(DaemonState::new(
        config.active_model.clone(),
        config_path.clone(),
    ));
    let daemon_state_clone = daemon_state.clone();

    // Spawn transcription thread
    info!("Loading transcription model...");
    let model_id = config.active_model.clone();
    thread::spawn(move || {
        if let Err(e) = transcription_worker(rx, model_path, model_id, daemon_state_clone) {
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

    if !daemon_state.model_loaded.load(Ordering::Relaxed) {
        error!("Model failed to load, exiting");
        std::process::exit(1);
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
                    config.auto_paste,
                    config.auto_inject,
                ) {
                    error!("Error handling client: {}", e);
                }
            }
            Err(e) => {
                error!("Connection failed: {}", e);
            }
        }

        if daemon_state.shutdown.load(Ordering::Relaxed) {
            info!("Shutting down daemon...");
            break;
        }
    }

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
    auto_paste: bool,
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
            auto_paste,
            auto_inject,
            daemon_state,
        ),
        Command::Start => handle_start(audio_capture, daemon_state),
        Command::Stop => handle_stop(
            audio_capture,
            tx,
            output_dir,
            device_sample_rate,
            auto_paste,
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
        Command::ListModels => handle_list_models(daemon_state),
        Command::ModelInfo { model_id } => handle_model_info(&model_id, daemon_state),
        Command::SetModel { model_id } => handle_set_model(&model_id, daemon_state),
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
    auto_paste: bool,
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
        output_dir: output_dir.clone(),
        auto_paste,
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
    auto_paste: bool,
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
            auto_paste,
            auto_inject,
            daemon_state,
        )
    }
}

fn handle_list_models(daemon_state: &Arc<DaemonState>) -> Response {
    use clevernote::models::ModelRegistry;

    let registry = match ModelRegistry::load() {
        Ok(r) => r,
        Err(e) => return Response::Error(format!("Failed to load model registry: {}", e)),
    };

    let models_dir = clevernote::get_models_dir();
    let active_model = daemon_state.active_model.lock().unwrap().clone();

    let model_infos: Vec<ipc::ModelInfo> = registry
        .list_models()
        .iter()
        .map(|m| ipc::ModelInfo {
            id: m.id.clone(),
            name: m.name.clone(),
            description: m.description.clone(),
            quantization: m.quantization.clone(),
            size_mb: m.size_mb,
            downloaded: registry.is_model_downloaded(&m.id, &models_dir),
            active: m.id == active_model,
        })
        .collect();

    Response::Success(SuccessResponse::ModelList {
        models: model_infos,
    })
}

fn handle_model_info(model_id: &str, daemon_state: &Arc<DaemonState>) -> Response {
    use clevernote::models::ModelRegistry;

    let registry = match ModelRegistry::load() {
        Ok(r) => r,
        Err(e) => return Response::Error(format!("Failed to load model registry: {}", e)),
    };

    let model = match registry.get_model(model_id) {
        Some(m) => m,
        None => return Response::Error(format!("Model '{}' not found", model_id)),
    };

    let models_dir = clevernote::get_models_dir();
    let active_model = daemon_state.active_model.lock().unwrap().clone();

    let info = ipc::ModelInfo {
        id: model.id.clone(),
        name: model.name.clone(),
        description: model.description.clone(),
        quantization: model.quantization.clone(),
        size_mb: model.size_mb,
        downloaded: registry.is_model_downloaded(&model.id, &models_dir),
        active: model.id == active_model,
    };

    Response::Success(SuccessResponse::ModelInfo { info })
}

fn handle_set_model(model_id: &str, daemon_state: &Arc<DaemonState>) -> Response {
    use clevernote::models::ModelRegistry;

    let registry = match ModelRegistry::load() {
        Ok(r) => r,
        Err(e) => return Response::Error(format!("Failed to load model registry: {}", e)),
    };

    // Check if model exists in registry
    if registry.get_model(model_id).is_none() {
        return Response::Error(format!("Model '{}' not found in registry", model_id));
    }

    // Check if model is downloaded
    let models_dir = clevernote::get_models_dir();
    if !registry.is_model_downloaded(model_id, &models_dir) {
        return Response::Error(format!(
            "Model '{}' is not downloaded. Run 'clevernote model pull {}' first.",
            model_id, model_id
        ));
    }

    // Update config
    let mut config = match DaemonConfig::load(&daemon_state.config_path) {
        Ok(c) => c,
        Err(e) => return Response::Error(format!("Failed to load config: {}", e)),
    };

    config.active_model = model_id.to_string();

    if let Err(e) = config.save(&daemon_state.config_path) {
        return Response::Error(format!("Failed to save config: {}", e));
    }

    // Update daemon state
    *daemon_state.active_model.lock().unwrap() = model_id.to_string();

    info!("Model set to '{}'. Initiating daemon restart...", model_id);

    // Trigger shutdown (systemd will auto-restart)
    daemon_state.shutdown.store(true, Ordering::Relaxed);

    Response::Success(SuccessResponse::ModelSet {
        model_id: model_id.to_string(),
    })
}

fn transcription_worker(
    rx: std::sync::mpsc::Receiver<TranscriptionJob>,
    model_path: PathBuf,
    model_id: String,
    daemon_state: Arc<DaemonState>,
) -> Result<()> {
    use clevernote::models::backends::create_backend;

    // Load registry to get backend type from model definition
    let registry = clevernote::models::ModelRegistry::load()
        .map_err(|e| eyre::eyre!("Failed to load model registry: {}", e))?;
    let backend_type = if let Some(model) = registry.get_model(&model_id) {
        model.backend.clone()
    } else {
        // Fallback to auto-detection if model not in registry
        use clevernote::models::backends::factory::detect_backend_type;
        match detect_backend_type(&model_path) {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to detect backend type: {}", e);
                return Err(eyre::eyre!(
                    "Cannot determine model type for: {}",
                    model_path.display()
                ));
            }
        }
    };

    info!(
        "Loading {} model from: {}",
        backend_type,
        model_path.display()
    );

    // Create the appropriate backend
    let mut backend = match create_backend(&model_path, &backend_type) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to create backend: {}", e);
            return Err(eyre::eyre!("Failed to load model: {}", e));
        }
    };

    daemon_state.model_loaded.store(true, Ordering::Relaxed);
    info!("Model '{}' loaded and ready", backend.model_name());

    // Process jobs
    while let Ok(job) = rx.recv() {
        if let Err(e) = process_transcription_with_backend(&mut *backend, job, &daemon_state) {
            error!("Transcription error: {}", e);
        }
    }

    Ok(())
}

fn process_transcription_with_backend(
    backend: &mut dyn clevernote::models::backends::TranscriptionBackend,
    job: TranscriptionJob,
    daemon_state: &Arc<DaemonState>,
) -> Result<()> {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Convert to mono if needed
    info!(
        "Converting audio: {} channels -> 1 channel (mono)",
        job.device_channels
    );
    let mono_audio = convert_to_mono(&job.audio_data, job.device_channels);

    // Transcribe
    info!("🤖 Transcribing with {}...", backend.backend_type());
    let start = Instant::now();

    let text = backend
        .transcribe(&mono_audio, job.device_sample_rate)
        .map_err(|e| eyre::eyre!("Transcription failed: {}", e))?;

    let elapsed = start.elapsed();
    info!("✅ Transcription complete in {:.2}s", elapsed.as_secs_f32());
    info!("📝 Text: {}", text);

    // Add to history.json (keep last 100)
    let history_path = job.output_dir.join("history.json");
    let mut history = TranscriptHistory::load_or_new(&history_path);

    let record = TranscriptRecord {
        timestamp: timestamp.clone(),
        text: text.clone(),
    };

    history.add_transcript(record);
    history.save(&history_path)?;
    info!(
        "💾 Updated history.json ({} total)",
        history.transcripts.len()
    );

    // Signal overlay: transcription is done (shows checkmark, auto-closes after 1.5 s)
    clevernote::recording_overlay::show_transcription_complete();

    // Copy to clipboard and optionally inject
    if let Err(e) = clipboard::copy_and_paste(&text, job.auto_paste, job.auto_inject) {
        warn!("Failed to copy: {}", e);
    }

    // Increment counter
    daemon_state
        .recordings_processed
        .fetch_add(1, Ordering::Relaxed);

    Ok(())
}
