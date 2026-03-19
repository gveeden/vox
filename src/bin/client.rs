/// Vox client - sends commands to the daemon
use clap::{Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;
use vox::ipc::{self, Command, Response, SuccessResponse};
use vox::models::{ensure_model_downloaded, remove_model, ModelRegistry};

#[derive(Parser)]
#[command(author, version, about = "Vox voice transcription client", long_about = None)]
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
    Start {
        /// Run LLM post-processing on this recording, overriding config
        #[arg(long)]
        llm: bool,
        /// Skip LLM post-processing for this recording, even if process_transcription = true in config
        #[arg(long, conflicts_with = "llm")]
        no_llm: bool,
        /// Custom LLM prompt for this recording (implies --llm).
        /// Use {text} as placeholder for the transcript.
        #[arg(long, value_name = "PROMPT")]
        prompt: Option<String>,
    },
    /// Stop recording (push-to-talk mode)
    Stop,
    /// Get daemon status
    Status,
    /// Quit the daemon
    Quit,
    /// Model management commands
    Model {
        #[command(subcommand)]
        cmd: ModelCommands,
    },
    /// Audio input device commands (local, no daemon IPC)
    Device {
        #[command(subcommand)]
        cmd: DeviceCommands,
    },
    /// Transcribe a local audio file
    Transcribe {
        /// Path to the audio file
        path: String,
    },
}

#[derive(Subcommand)]
enum ModelCommands {
    /// List available models
    List,
    /// Download a model
    Pull {
        /// Model ID to download
        model_id: String,
    },
    /// Set the active model (daemon will auto-restart)
    Set {
        /// Model ID to set as active
        model_id: String,
    },
    /// Remove a downloaded model
    Rm {
        /// Model ID to remove
        model_id: String,
    },
}

#[derive(Subcommand)]
enum DeviceCommands {
    /// List local input audio devices
    List,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Toggle => handle_command(Command::Toggle),
        Commands::Start {
            llm,
            no_llm,
            prompt,
        } => handle_command(Command::Start {
            use_llm: if llm || prompt.is_some() {
                Some(true)
            } else if no_llm {
                Some(false)
            } else {
                None
            },
            prompt_override: prompt,
        }),
        Commands::Stop => handle_command(Command::Stop),
        Commands::Status => handle_command(Command::Status),
        Commands::Quit => handle_command(Command::Quit),
        Commands::Model { cmd } => handle_model_command(cmd),
        Commands::Device { cmd } => handle_device_command(cmd),
        Commands::Transcribe { path } => handle_command(Command::TranscribeFile { path }),
    }
}

fn handle_device_command(cmd: DeviceCommands) {
    let result = match cmd {
        DeviceCommands::List => list_input_devices(),
    };

    if let Err(err) = result {
        eprintln!("Error: {}", err);
        std::process::exit(1);
    }
}

fn list_input_devices() -> Result<(), String> {
    let host = cpal::default_host();
    let default_device_name = host.default_input_device().and_then(|d| d.name().ok());

    let input_devices = host
        .input_devices()
        .map_err(|err| format!("Failed to enumerate input devices: {}", err))?;

    let mut device_names = Vec::new();
    for device in input_devices {
        let name = device
            .name()
            .unwrap_or_else(|_| "<unavailable device name>".to_string());
        device_names.push(name);
    }

    if device_names.is_empty() {
        return Err("No input audio devices found".to_string());
    }

    println!("Local input devices:");
    for (index, name) in device_names.iter().enumerate() {
        let default_marker = if default_device_name.as_deref() == Some(name.as_str()) {
            " [default]"
        } else {
            ""
        };
        println!("  {}: {}{}", index, name, default_marker);
    }

    Ok(())
}

fn handle_command(command: Command) {
    if let Err(e) = send_command(command) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn handle_model_command(cmd: ModelCommands) {
    let models_dir = vox::get_models_dir();

    match cmd {
        ModelCommands::List => {
            if let Err(e) = list_models() {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        ModelCommands::Pull { model_id } => {
            if let Err(e) = pull_model(&model_id, &models_dir) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
        ModelCommands::Set { model_id } => {
            // Check if daemon is running by trying to connect
            let socket_path = ipc::socket_path();
            let daemon_running = socket_path.exists()
                && std::os::unix::net::UnixStream::connect(&socket_path).is_ok();

            if daemon_running {
                // Daemon is running, send command to update and restart
                handle_command(Command::SetModel { model_id });
            } else {
                // Daemon not running, update config directly
                if let Err(e) = set_model_direct(&model_id) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
            }
        }
        ModelCommands::Rm { model_id } => {
            if let Err(e) = remove_model(&model_id, &models_dir) {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
            println!("Model '{}' removed successfully", model_id);
        }
    }
}

fn list_models() -> Result<(), String> {
    let registry =
        ModelRegistry::load().map_err(|e| format!("Failed to load model registry: {}", e))?;
    let models_dir = vox::get_models_dir();

    println!("Available models:");
    println!();

    for model in registry.list_models() {
        let downloaded = registry.is_model_downloaded(&model.id, &models_dir);
        let status = if downloaded {
            "✓ Downloaded"
        } else {
            "  Not downloaded"
        };
        let default = if model.default { " (default)" } else { "" };

        println!("{} - {}{}", model.id, model.name, default);
        println!("  {}", model.description);
        println!(
            "  Size: {} MB | Quantization: {} | {}",
            model.size_mb, model.quantization, status
        );
        println!();
    }

    Ok(())
}

fn pull_model(model_id: &str, models_dir: &std::path::PathBuf) -> Result<(), String> {
    let registry =
        ModelRegistry::load().map_err(|e| format!("Failed to load model registry: {}", e))?;

    let model = registry
        .get_model(model_id)
        .ok_or_else(|| format!("Model '{}' not found in registry", model_id))?;

    if registry.is_model_downloaded(model_id, models_dir) {
        println!("Model '{}' is already downloaded", model_id);
        return Ok(());
    }

    println!("Downloading model '{}'...", model_id);
    println!("Total size: {} MB", model.size_mb);
    println!();

    match ensure_model_downloaded(model_id, models_dir, true) {
        Ok(path) => {
            println!();
            println!("✓ Model downloaded successfully to: {}", path.display());
            Ok(())
        }
        Err(e) => Err(format!("Failed to download model: {}", e)),
    }
}

fn set_model_direct(model_id: &str) -> Result<(), String> {
    use std::fs;
    use vox::models::ModelRegistry;

    // Verify model exists in registry
    let registry =
        ModelRegistry::load().map_err(|e| format!("Failed to load model registry: {}", e))?;

    if registry.get_model(model_id).is_none() {
        return Err(format!("Model '{}' not found in registry", model_id));
    }

    // Check if model is downloaded
    let models_dir = vox::get_models_dir();
    if !registry.is_model_downloaded(model_id, &models_dir) {
        return Err(format!(
            "Model '{}' is not downloaded. Run 'vox model pull {}' first.",
            model_id, model_id
        ));
    }

    // Load config
    let config_dir = vox::get_config_dir();
    let config_path = config_dir.join("config.toml");

    let config_contents = if config_path.exists() {
        fs::read_to_string(&config_path).map_err(|e| format!("Failed to read config: {}", e))?
    } else {
        String::new()
    };

    // Parse/update config
    let mut config: toml::Value = if config_contents.trim().is_empty() {
        toml::Value::Table(toml::map::Map::new())
    } else {
        config_contents
            .parse()
            .map_err(|e| format!("Failed to parse config: {}", e))?
    };

    // Update active_model
    if let toml::Value::Table(ref mut table) = config {
        table.insert(
            "active_model".to_string(),
            toml::Value::String(model_id.to_string()),
        );
    }

    // Save config
    let config_str = toml::to_string_pretty(&config)
        .map_err(|e| format!("Failed to serialize config: {}", e))?;

    fs::write(&config_path, config_str).map_err(|e| format!("Failed to write config: {}", e))?;

    println!("✓ Active model set to '{}'", model_id);
    println!("  Config updated at: {:?}", config_path);
    println!("  Start the daemon to use this model.");

    Ok(())
}

fn send_command(command: Command) -> Result<(), String> {
    let socket_path = ipc::socket_path();

    // Check if socket exists
    if !socket_path.exists() {
        return Err(format!(
            "Daemon not running. Socket not found at: {}\n\
             Start the daemon with: vox-daemon",
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
                    println!("Vox Daemon Status:");
                    println!(
                        "  Recording: {}",
                        if status.is_recording { "Yes" } else { "No" }
                    );
                    println!(
                        "  Model loaded: {}",
                        if status.model_loaded { "Yes" } else { "No" }
                    );
                    if let Some(model) = status.active_model {
                        println!("  Active model: {}", model);
                    }
                    println!("  Uptime: {} seconds", status.uptime_seconds);
                    println!("  Recordings processed: {}", status.recordings_processed);
                    println!(
                        "  LLM configured: {}",
                        if status.llm_configured { "Yes" } else { "No" }
                    );
                    if let Some(backend) = status.llm_backend {
                        println!("  LLM backend: {}", backend);
                    }
                    println!(
                        "  LLM on every recording: {}",
                        if status.process_transcription {
                            "Yes"
                        } else {
                            "No (use --llm to enable per-recording)"
                        }
                    );
                }
                SuccessResponse::Shutdown => {
                    println!("Daemon shutting down");
                }
                SuccessResponse::ModelList { models } => {
                    println!("Available models (via daemon):");
                    for m in models {
                        let downloaded = if m.downloaded { "✓" } else { " " };
                        let active = if m.active { "[ACTIVE] " } else { "" };
                        println!("  [{}] {} - {}{}", downloaded, m.id, active, m.name);
                    }
                }
                SuccessResponse::ModelInfo { info } => {
                    println!("Model: {}", info.name);
                    println!("  ID: {}", info.id);
                    println!("  Description: {}", info.description);
                    println!("  Size: {} MB", info.size_mb);
                    println!("  Quantization: {}", info.quantization);
                    println!(
                        "  Downloaded: {}",
                        if info.downloaded { "Yes" } else { "No" }
                    );
                    if info.active {
                        println!("  Status: ACTIVE");
                    }
                }
                SuccessResponse::ModelSet { model_id } => {
                    println!("✓ Active model set to '{}'", model_id);
                    println!("  Daemon is restarting...");
                    println!("  Wait a moment, then check status with: vox status");
                }
                SuccessResponse::FileTranscribed { text } => {
                    println!("{}", text);
                }
            }
            Ok(())
        }
        Response::Error(err) => Err(format!("Daemon error: {}", err)),
    }
}
