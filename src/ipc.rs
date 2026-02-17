/// IPC protocol for daemon-client communication
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Commands sent from client to daemon
#[derive(Debug, Serialize, Deserialize)]
pub enum Command {
    /// Start or stop recording (toggle)
    Toggle,
    /// Start recording (push-to-talk)
    Start,
    /// Stop recording (push-to-talk)
    Stop,
    /// Get current daemon status
    Status,
    /// Gracefully shut down the daemon
    Quit,
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
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub is_recording: bool,
    pub uptime_seconds: u64,
    pub model_loaded: bool,
    pub recordings_processed: u64,
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
