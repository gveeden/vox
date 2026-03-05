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
    /// Enable verbose logging (includes ORT memory allocator messages and
    /// per-token LLM debug output)
    #[arg(long, short)]
    verbose: bool,
}

/// Read the current process RSS (resident set size) from /proc/self/status.
/// Returns MB as f64, or 0.0 if unavailable (non-Linux platforms).
fn rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
    }
    0.0
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
    /// Process every transcription through the LLM by default.
    /// The --llm flag on `clevernote start` overrides this per-recording.
    /// The LLM model is always loaded in the background if llm_model is set
    /// and has been downloaded — this flag only controls default behaviour.
    #[serde(default = "default_process_transcription", alias = "process_with_llm")]
    process_transcription: bool,
    #[serde(default = "default_process_prompt")]
    process_prompt: String,
    #[serde(default = "default_llm_model")]
    llm_model: String,
    #[serde(default = "default_intra_threads")]
    intra_threads: usize,
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

fn default_process_transcription() -> bool {
    false
}

fn default_process_prompt() -> String {
    "Clean up the following voice transcription: remove filler words (um, uh, like, you know), fix grammar and punctuation, and return only the cleaned text with no additional commentary: {text}".to_string()
}

fn default_llm_model() -> String {
    "qwen3.5-0.8b-fp16".to_string()
}

fn default_intra_threads() -> usize {
    4
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
            process_transcription: default_process_transcription(),
            process_prompt: default_process_prompt(),
            llm_model: default_llm_model(),
            intra_threads: default_intra_threads(),
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
    /// When the recording stop signal was received (for end-to-end latency).
    received_at: Instant,
    /// Per-recording LLM override. None = use config value.
    use_llm: Option<bool>,
    /// Per-recording prompt override. None = use config value.
    prompt_override: Option<String>,
}

/// A request sent to the LLM worker thread.
struct LlmRequest {
    text: String,
    prompt_template: String,
    /// Sender to return the processed result (or error) on.
    reply: std::sync::mpsc::SyncSender<Result<String, String>>,
}

struct DaemonState {
    model_loaded: AtomicBool,
    recordings_processed: AtomicU64,
    start_time: Instant,
    shutdown: AtomicBool,
    overlay_close_handle: std::sync::Mutex<Option<Arc<AtomicBool>>>,
    active_model: std::sync::Mutex<String>,
    config_path: PathBuf,
    /// LLM override set by `handle_start` for the current recording.
    /// None = use config value; Some(true/false) = force on/off.
    pending_use_llm: std::sync::Mutex<Option<bool>>,
    /// Prompt override set by `handle_start` for the current recording.
    pending_prompt_override: std::sync::Mutex<Option<String>>,
    /// Persistent uinput paste device (Linux only). Created once at startup so
    /// the Wayland registration cost is paid then, not at each paste.
    #[cfg(target_os = "linux")]
    paste_device: std::sync::Mutex<Option<clevernote::inject_linux::PasteDevice>>,
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
            pending_use_llm: std::sync::Mutex::new(None),
            pending_prompt_override: std::sync::Mutex::new(None),
            #[cfg(target_os = "linux")]
            paste_device: std::sync::Mutex::new(None),
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

    // Initialize logger — verbose mode also shows DEBUG from this crate and
    // lets the ORT arena allocator messages through (they log at INFO in the
    // ort::logging target).
    let log_level = if args.verbose {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn) // silence noisy deps by default
        .filter_module("clevernote", log_level)
        .filter_module("clevernote_daemon", log_level)
        .filter_module(
            "ort",
            if args.verbose {
                log::LevelFilter::Info
            } else {
                log::LevelFilter::Warn
            },
        )
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

    // Create persistent paste device (Wayland registration cost paid once here).
    // If creation fails (e.g. capability not granted) we fall back to the
    // one-shot path for each paste.
    #[cfg(target_os = "linux")]
    let initial_paste_device: Option<clevernote::inject_linux::PasteDevice> = {
        if config.auto_paste && !config.auto_inject {
            match clevernote::inject_linux::PasteDevice::new() {
                Ok(dev) => {
                    info!("Persistent paste device ready");
                    Some(dev)
                }
                Err(e) => {
                    warn!(
                        "Could not create persistent paste device (will use fallback): {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        }
    };

    // Start LLM worker if the configured model has been downloaded.
    // The worker is loaded independently of process_transcription — that flag
    // only controls whether LLM runs by default; --llm on `start` also triggers it.
    let llm_tx: Option<std::sync::mpsc::Sender<LlmRequest>> = {
        let llm_model_id = config.llm_model.clone();
        match clevernote::models::ensure_model_downloaded(&llm_model_id, &models_dir, false) {
            Ok(llm_path) => {
                info!(
                    "Loading LLM model ({}) from: {} (RSS before: {:.0} MB)",
                    llm_model_id,
                    llm_path.display(),
                    rss_mb()
                );
                let (ltx, lrx) = std::sync::mpsc::channel::<LlmRequest>();
                // Look up backend type from registry so Gemma vs Qwen dispatch works correctly
                let llm_registry = clevernote::models::ModelRegistry::load();
                let llm_backend_type = llm_registry
                    .ok()
                    .and_then(|r| r.get_model(&llm_model_id).map(|m| m.backend.clone()))
                    .unwrap_or_else(|| "llm".to_string());
                let llm_intra_threads = config.intra_threads;
                thread::spawn(move || {
                    if let Err(e) = llm_worker(lrx, llm_path, &llm_backend_type, llm_intra_threads)
                    {
                        error!("LLM worker error: {}", e);
                    }
                });
                Some(ltx)
            }
            Err(e) => {
                info!(
                    "LLM model '{}' not downloaded, LLM post-processing unavailable ({})",
                    llm_model_id, e
                );
                None
            }
        }
    };

    // Create transcription channel
    let (tx, rx) = std::sync::mpsc::channel::<TranscriptionJob>();

    // Daemon state
    let daemon_state = Arc::new(DaemonState::new(
        config.active_model.clone(),
        config_path.clone(),
    ));
    // Seed the persistent paste device into daemon state so the transcription
    // worker thread can access it.
    #[cfg(target_os = "linux")]
    {
        *daemon_state.paste_device.lock().unwrap() = initial_paste_device;
    }
    let daemon_state_clone = daemon_state.clone();

    // Spawn transcription thread
    info!("Loading transcription model...");
    let model_id = config.active_model.clone();
    let process_prompt = config.process_prompt.clone();
    let process_transcription = config.process_transcription;
    let intra_threads = config.intra_threads;
    thread::spawn(move || {
        if let Err(e) = transcription_worker(
            rx,
            model_path,
            model_id,
            daemon_state_clone,
            llm_tx,
            process_prompt,
            process_transcription,
            intra_threads,
        ) {
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
        Command::Start {
            use_llm,
            prompt_override,
        } => handle_start(audio_capture, daemon_state, use_llm, prompt_override),
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

fn handle_start(
    audio_capture: &AudioCapture,
    daemon_state: &Arc<DaemonState>,
    use_llm: Option<bool>,
    prompt_override: Option<String>,
) -> Response {
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

    // Store per-recording LLM overrides for handle_stop to pick up
    *daemon_state.pending_use_llm.lock().unwrap() = use_llm;
    *daemon_state.pending_prompt_override.lock().unwrap() = prompt_override;

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

    // Consume per-recording LLM overrides set by handle_start
    let use_llm = daemon_state.pending_use_llm.lock().unwrap().take();
    let prompt_override = daemon_state.pending_prompt_override.lock().unwrap().take();

    // Send to transcription thread
    let device_channels = audio_capture.channels();
    let job = TranscriptionJob {
        audio_data,
        device_sample_rate,
        device_channels,
        output_dir: output_dir.clone(),
        auto_paste,
        auto_inject,
        received_at: Instant::now(),
        use_llm,
        prompt_override,
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
        // Toggle has no LLM override — use config values
        handle_start(audio_capture, daemon_state, None, None)
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

/// A trait object wrapper so we can hold either LlmBackend or GemmaBackend
/// in the same worker loop without boxing each call site.
trait LlmProcessor: Send {
    fn process(&mut self, text: &str, prompt_template: &str) -> anyhow::Result<String>;
}

impl LlmProcessor for clevernote::models::backends::llm::LlmBackend {
    fn process(&mut self, text: &str, prompt_template: &str) -> anyhow::Result<String> {
        self.process(text, prompt_template)
    }
}

impl LlmProcessor for clevernote::models::backends::gemma::GemmaBackend {
    fn process(&mut self, text: &str, prompt_template: &str) -> anyhow::Result<String> {
        self.process(text, prompt_template)
    }
}

impl LlmProcessor for clevernote::models::backends::qwen3::Qwen3Backend {
    fn process(&mut self, text: &str, prompt_template: &str) -> anyhow::Result<String> {
        self.process(text, prompt_template)
    }
}

impl LlmProcessor for clevernote::models::backends::gemma3::Gemma3Backend {
    fn process(&mut self, text: &str, prompt_template: &str) -> anyhow::Result<String> {
        self.process(text, prompt_template)
    }
}

fn llm_worker(
    rx: std::sync::mpsc::Receiver<LlmRequest>,
    model_path: PathBuf,
    backend_type: &str,
    intra_threads: usize,
) -> Result<()> {
    info!(
        "Loading LLM model ({}) from: {} (RSS before: {:.0} MB)",
        backend_type,
        model_path.display(),
        rss_mb()
    );

    let mut backend: Box<dyn LlmProcessor> = match backend_type {
        "gemma" => {
            use clevernote::models::backends::gemma::GemmaBackend;
            match GemmaBackend::new(&model_path, intra_threads) {
                Ok(b) => Box::new(b),
                Err(e) => {
                    error!("Failed to load Gemma model: {}", e);
                    return Err(eyre::eyre!("Failed to load Gemma model: {}", e));
                }
            }
        }
        "qwen3" => {
            use clevernote::models::backends::qwen3::Qwen3Backend;
            match Qwen3Backend::new(&model_path, intra_threads) {
                Ok(b) => Box::new(b),
                Err(e) => {
                    error!("Failed to load Qwen3 model: {}", e);
                    return Err(eyre::eyre!("Failed to load Qwen3 model: {}", e));
                }
            }
        }
        "gemma3" => {
            use clevernote::models::backends::gemma3::Gemma3Backend;
            match Gemma3Backend::new(&model_path, intra_threads) {
                Ok(b) => Box::new(b),
                Err(e) => {
                    error!("Failed to load Gemma3 model: {}", e);
                    return Err(eyre::eyre!("Failed to load Gemma3 model: {}", e));
                }
            }
        }
        _ => {
            // Default: Qwen3.5 hybrid LlmBackend
            use clevernote::models::backends::llm::LlmBackend;
            match LlmBackend::new(&model_path, intra_threads) {
                Ok(b) => Box::new(b),
                Err(e) => {
                    error!("Failed to load LLM model: {}", e);
                    return Err(eyre::eyre!("Failed to load LLM model: {}", e));
                }
            }
        }
    };

    info!("LLM model loaded and ready — RSS: {:.0} MB", rss_mb());

    while let Ok(req) = rx.recv() {
        let result = match backend.process(&req.text, &req.prompt_template) {
            Ok(processed) => Ok(processed),
            Err(e) => {
                error!("LLM processing error: {}", e);
                Err(format!("{}", e))
            }
        };
        let _ = req.reply.send(result);
    }

    Ok(())
}

fn transcription_worker(
    rx: std::sync::mpsc::Receiver<TranscriptionJob>,
    model_path: PathBuf,
    model_id: String,
    daemon_state: Arc<DaemonState>,
    llm_tx: Option<std::sync::mpsc::Sender<LlmRequest>>,
    process_prompt: String,
    process_transcription: bool,
    intra_threads: usize,
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
        "Loading {} model from: {} (RSS before: {:.0} MB)",
        backend_type,
        model_path.display(),
        rss_mb()
    );

    // Create the appropriate backend
    let mut backend = match create_backend(&model_path, &backend_type, intra_threads) {
        Ok(b) => b,
        Err(e) => {
            error!("Failed to create backend: {}", e);
            return Err(eyre::eyre!("Failed to load model: {}", e));
        }
    };

    daemon_state.model_loaded.store(true, Ordering::Relaxed);
    info!(
        "Model '{}' loaded and ready — RSS: {:.0} MB",
        backend.model_name(),
        rss_mb()
    );

    // Process jobs
    while let Ok(job) = rx.recv() {
        if let Err(e) = process_transcription_with_backend(
            &mut *backend,
            job,
            &daemon_state,
            llm_tx.as_ref(),
            &process_prompt,
            process_transcription,
        ) {
            error!("Transcription error: {}", e);
        }
    }

    Ok(())
}

fn process_transcription_with_backend(
    backend: &mut dyn clevernote::models::backends::TranscriptionBackend,
    job: TranscriptionJob,
    daemon_state: &Arc<DaemonState>,
    llm_tx: Option<&std::sync::mpsc::Sender<LlmRequest>>,
    process_prompt: &str,
    process_transcription: bool,
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
    let end_to_end = job.received_at.elapsed();
    info!(
        "✅ Transcription complete — model: {:.2}s, end-to-end: {:.2}s",
        elapsed.as_secs_f32(),
        end_to_end.as_secs_f32()
    );
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

    // Resolve effective LLM flag:
    //   - job.use_llm (set by --llm / --prompt CLI flags) takes precedence
    //   - otherwise fall back to the config's process_transcription default
    let effective_use_llm = job.use_llm.unwrap_or(process_transcription);
    let effective_prompt = job.prompt_override.as_deref().unwrap_or(process_prompt);

    // Optionally run LLM post-processing before clipboard
    let final_text = if effective_use_llm {
        if let Some(ltx) = llm_tx {
            info!("🧠 Running LLM post-processing...");
            let llm_start = Instant::now();

            let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel::<Result<String, String>>(1);
            let req = LlmRequest {
                text: text.clone(),
                prompt_template: effective_prompt.to_string(),
                reply: reply_tx,
            };

            match ltx.send(req) {
                Ok(()) => {
                    // Wait up to 30 seconds for the LLM to respond
                    match reply_rx.recv_timeout(Duration::from_secs(30)) {
                        Ok(Ok(processed)) => {
                            info!(
                                "✅ LLM cleanup complete — {:.2}s (total: {:.2}s)",
                                llm_start.elapsed().as_secs_f32(),
                                job.received_at.elapsed().as_secs_f32()
                            );
                            info!("📝 LLM output: {}", processed);
                            processed
                        }
                        Ok(Err(e)) => {
                            warn!("LLM processing failed (using raw transcript): {}", e);
                            text.clone()
                        }
                        Err(_) => {
                            warn!("LLM processing timed out (using raw transcript)");
                            text.clone()
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to send to LLM worker (using raw transcript): {}", e);
                    text.clone()
                }
            }
        } else {
            warn!("--llm requested but LLM worker is not running (is llm_model configured and downloaded?)");
            text.clone()
        }
    } else {
        text.clone()
    };

    // Copy to clipboard and optionally inject
    #[cfg(target_os = "linux")]
    let paste_result = {
        let mut dev_guard = daemon_state.paste_device.lock().unwrap();
        clipboard::copy_and_paste(
            &final_text,
            job.auto_paste,
            job.auto_inject,
            dev_guard.as_mut(),
        )
    };
    #[cfg(not(target_os = "linux"))]
    let paste_result = clipboard::copy_and_paste(&final_text, job.auto_paste, job.auto_inject);
    if let Err(e) = paste_result {
        warn!("Failed to copy: {}", e);
    }

    // Increment counter
    daemon_state
        .recordings_processed
        .fetch_add(1, Ordering::Relaxed);

    Ok(())
}
