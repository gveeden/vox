/// IPC protocol for daemon-client communication
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Commands sent from client to daemon
#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Start or stop recording (toggle)
    Toggle,
    /// Start recording (push-to-talk)
    Start {
        /// Override config's process_with_llm for this recording only.
        /// None = use config value; Some(true/false) = force on/off.
        use_llm: Option<bool>,
        /// Override the LLM prompt for this recording only.
        /// None = use config value.
        prompt_override: Option<String>,
    },
    /// Stop recording (push-to-talk)
    Stop,
    /// Get current daemon status
    Status,
    /// Gracefully shut down the daemon
    Quit,
    /// List available models
    ListModels,
    /// Get information about a specific model
    ModelInfo { model_id: String },
    /// Set the active model (requires daemon restart)
    SetModel { model_id: String },
    /// Transcribe a local audio file
    TranscribeFile { path: String },
}

/// Responses sent from daemon to client
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    /// Command executed successfully
    Success(SuccessResponse),
    /// Command failed with error
    Error(String),
}

#[derive(Debug, Serialize, Deserialize)]
pub enum SuccessResponse {
    /// Recording started
    RecordingStarted,
    /// Recording stopped, transcription in progress
    RecordingStopped,
    /// Already recording (tried to start when already started)
    AlreadyRecording,
    /// Not recording (tried to stop when not recording)
    NotRecording,
    /// Current daemon status
    Status(DaemonStatus),
    /// Daemon shutting down
    Shutdown,
    /// List of available models
    ModelList { models: Vec<ModelInfo> },
    /// Information about a specific model
    ModelInfo { info: ModelInfo },
    /// Model set successfully (daemon will restart)
    ModelSet { model_id: String },
    /// Transcription of a file completed
    FileTranscribed { text: String },
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub is_recording: bool,
    pub uptime_seconds: u64,
    pub model_loaded: bool,
    pub recordings_processed: u64,
    pub active_model: Option<String>,
    /// Whether an LLM worker is configured and running (local or API).
    pub llm_configured: bool,
    /// Human-readable description of the LLM backend, e.g. "local: qwen3.5-0.8b-fp16"
    /// or "api: anthropic / claude-haiku-4-5". None when no LLM is configured.
    pub llm_backend: Option<String>,
    /// Whether LLM post-processing runs on every recording by default.
    pub process_transcription: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub quantization: String,
    pub size_mb: u64,
    pub downloaded: bool,
    pub active: bool,
}

/// Get the socket path for daemon communication
pub fn socket_path() -> PathBuf {
    crate::get_config_dir().join("daemon.sock")
}

/// Serialize a command to JSON bytes
pub fn serialize_command(cmd: &Command) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_string(cmd)?;
    let mut bytes = json.into_bytes();
    bytes.push(b'\n'); // Newline delimiter
    Ok(bytes)
}

/// Deserialize a response from JSON bytes
pub fn deserialize_response(bytes: &[u8]) -> Result<Response, serde_json::Error> {
    serde_json::from_slice(bytes)
}

/// Serialize a response to JSON bytes
pub fn serialize_response(resp: &Response) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_string(resp)?;
    let mut bytes = json.into_bytes();
    bytes.push(b'\n'); // Newline delimiter
    Ok(bytes)
}

/// Deserialize a command from JSON bytes
pub fn deserialize_command(bytes: &[u8]) -> Result<Command, serde_json::Error> {
    serde_json::from_slice(bytes)
}
