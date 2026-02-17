/// CleverNote client - sends commands to the daemon
use clap::{Parser, Subcommand};
use clevernote::ipc::{self, Command, Response, SuccessResponse};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

#[derive(Parser)]
#[command(author, version, about = "CleverNote voice transcription client", long_about = None)]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Toggle recording on/off
    Toggle,
    /// Start recording (push-to-talk mode)
    Start,
    /// Stop recording (push-to-talk mode)
    Stop,
    /// Get daemon status
    Status,
    /// Quit the daemon
    Quit,
}

fn main() {
    let cli = Cli::parse();

    let command = match cli.command {
        Commands::Toggle => Command::Toggle,
        Commands::Start => Command::Start,
        Commands::Stop => Command::Stop,
        Commands::Status => Command::Status,
        Commands::Quit => Command::Quit,
    };

    if let Err(e) = send_command(command) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn send_command(command: Command) -> Result<(), String> {
    let socket_path = ipc::socket_path();

    // Check if socket exists
    if !socket_path.exists() {
        return Err(format!(
            "Daemon not running. Socket not found at: {}\n\
             Start the daemon with: clevernote-daemon",
            socket_path.display()
        ));
    }

    // Connect to daemon
    let mut stream = UnixStream::connect(&socket_path)
        .map_err(|e| format!("Failed to connect to daemon: {}", e))?;

    // Set timeout for read operations
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .map_err(|e| format!("Failed to set timeout: {}", e))?;

    // Send command
    let cmd_bytes = ipc::serialize_command(&command)
        .map_err(|e| format!("Failed to serialize command: {}", e))?;

    stream
        .write_all(&cmd_bytes)
        .map_err(|e| format!("Failed to send command: {}", e))?;

    stream
        .flush()
        .map_err(|e| format!("Failed to flush stream: {}", e))?;

    // Read response
    let mut reader = BufReader::new(stream);
    let mut response_line = String::new();
    reader
        .read_line(&mut response_line)
        .map_err(|e| format!("Failed to read response: {}", e))?;

    let response: Response = ipc::deserialize_response(response_line.trim().as_bytes())
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    // Handle response
    match response {
        Response::Success(success) => {
            match success {
                SuccessResponse::RecordingStarted => {
                    println!("🔴 Recording started");
                }
                SuccessResponse::RecordingStopped => {
                    println!("⏹️  Recording stopped - transcribing...");
                }
                SuccessResponse::AlreadyRecording => {
                    println!("⚠️  Already recording");
                }
                SuccessResponse::NotRecording => {
                    println!("⚠️  Not recording");
                }
                SuccessResponse::Status(status) => {
                    println!("CleverNote Daemon Status:");
                    println!(
                        "  Recording: {}",
                        if status.is_recording { "Yes" } else { "No" }
                    );
                    println!(
                        "  Model loaded: {}",
                        if status.model_loaded { "Yes" } else { "No" }
                    );
                    println!("  Uptime: {} seconds", status.uptime_seconds);
                    println!("  Recordings processed: {}", status.recordings_processed);
                }
                SuccessResponse::Shutdown => {
                    println!("Daemon shutting down");
                }
            }
            Ok(())
        }
        Response::Error(err) => Err(format!("Daemon error: {}", err)),
    }
}
