mod model_downloader;
mod progress_window;

use arboard::Clipboard;
use chrono::Local;
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use eyre::{Result, WrapErr};
use hound::{WavSpec, WavWriter};
use parakeet_rs::{Parakeet, ParakeetTDT, TimestampMode, Transcriber};
use global_hotkey::{
    hotkey::{Code, HotKey, Modifiers},
    GlobalHotKeyEvent, GlobalHotKeyManager,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tao::{
    dpi::{LogicalSize, PhysicalPosition},
    event_loop::{ControlFlow, EventLoopBuilder},
    window::WindowBuilder,
};
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    TrayIconBuilder,
};

#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

#[cfg(target_os = "macos")]
use cocoa::appkit::NSColor;
#[cfg(target_os = "macos")]
use cocoa::base::{id, nil, NO};


#[derive(Parser, Debug)]
#[command(author, version, about = "Voice transcription with Parakeet ASR", long_about = None)]
struct Args {
    /// Path to ONNX model directory (for TDT model) or model file (for CTC model)
    /// If not specified, defaults to "models/parakeet-tdt" and will download if missing
    #[arg(short, long)]
    model: Option<String>,

    /// Use TDT model (multilingual). If not set, uses CTC model (English-only)
    #[arg(short, long, default_value_t = true)]
    tdt: bool,

    /// Output directory for transcripts (relative to ~/.clevernote/)
    #[arg(short, long, default_value = "transcripts")]
    output_dir: String,

    /// Keep audio recordings (don't delete after transcription)
    #[arg(short, long, default_value_t = false)]
    keep_audio: bool,

    /// Audio sample rate in Hz (Parakeet models expect 16kHz)
    #[arg(short, long, default_value_t = 16000)]
    sample_rate: u32,

    /// Path to config file (defaults to config.toml)
    #[arg(short = 'c', long, default_value = "config.toml")]
    config_file: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    #[serde(default = "default_modifier_key")]
    modifier_key: String,
    #[serde(default = "default_trigger_key")]
    trigger_key: String,
}

fn default_modifier_key() -> String {
    "Alt".to_string()
}

fn default_trigger_key() -> String {
    "Space".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            modifier_key: default_modifier_key(),
            trigger_key: default_trigger_key(),
        }
    }
}

impl Config {
    fn load(path: &str) -> Result<Self> {
        if PathBuf::from(path).exists() {
            let contents = fs::read_to_string(path)?;
            let config: Config = toml::from_str(&contents)?;
            Ok(config)
        } else {
            // Create default config
            let config = Config::default();
            let toml_string = toml::to_string_pretty(&config)?;
            fs::write(path, toml_string)?;
            println!("📝 Created default config at: {}", path);
            Ok(config)
        }
    }
}

struct RecordingState {
    is_recording: bool,
    audio_data: Vec<f32>,
    device_sample_rate: u32,
}

struct TranscriptionJob {
    audio_data: Vec<f32>,
    device_sample_rate: u32,
    target_sample_rate: u32,
    output_dir: String,
    keep_audio: bool,
}



impl RecordingState {
    fn new(device_sample_rate: u32) -> Self {
        Self {
            is_recording: false,
            audio_data: Vec::new(),
            device_sample_rate,
        }
    }
}

fn parse_modifier_key(key_name: &str) -> Modifiers {
    match key_name.to_lowercase().as_str() {
        "alt" | "option" => Modifiers::ALT,
        "ctrl" | "control" => Modifiers::CONTROL,
        "cmd" | "command" | "meta" | "super" => Modifiers::META,
        "shift" => Modifiers::SHIFT,
        _ => Modifiers::ALT, // default to Alt
    }
}

fn parse_trigger_key(key_name: &str) -> Code {
    match key_name.to_lowercase().as_str() {
        "space" => Code::Space,
        "return" | "enter" => Code::Enter,
        "tab" => Code::Tab,
        "escape" | "esc" => Code::Escape,
        _ => Code::Space, // default to Space
    }
}

/// Resample audio from source sample rate to target sample rate using linear interpolation
fn resample_audio(audio: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate {
        return audio.to_vec();
    }

    let ratio = source_rate as f64 / target_rate as f64;
    let output_len = (audio.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = i as f64 * ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f64;

        // Linear interpolation
        let sample = audio[src_idx_floor] * (1.0 - frac as f32) + audio[src_idx_ceil] * frac as f32;
        resampled.push(sample);
    }

    resampled
}

#[cfg(target_os = "macos")]
fn check_accessibility_permissions() {
    let exe_path = std::env::current_exe()
        .unwrap_or_default()
        .display()
        .to_string();
    
    // Print helpful information at startup
    eprintln!("\n📝 Starting CleverNote...");
    eprintln!("   Binary: {}", exe_path);
    eprintln!();
}

#[cfg(target_os = "macos")]
fn check_and_offer_installation() -> Result<()> {
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    let plist_path = format!("{}/Library/LaunchAgents/com.clevernote.parakeet.plist", home_dir);
    let first_run_marker = format!("{}/.clevernote/first_run_complete", home_dir);
    
    // Check if we've already prompted the user (or if LaunchAgent is installed)
    if PathBuf::from(&first_run_marker).exists() || PathBuf::from(&plist_path).exists() {
        return Ok(());
    }
    
    // Show a native macOS dialog for first run
    let message = "CleverNote can run automatically on startup as a background service.\n\n\
                   Benefits:\n\
                   • Always available - no need to start manually\n\
                   • Runs in the background - no dock icon\n\
                   • Starts on login automatically\n\n\
                   Would you like to install CleverNote as a background service?";
    
    let result = show_dialog("Welcome to CleverNote!", message, &["Install", "Skip"]);
    
    if result != 0 {
        // User clicked "Skip"
        // Create marker to not ask again
        if let Some(parent) = PathBuf::from(&first_run_marker).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&first_run_marker, "skipped")?;
        
        return Ok(());
    }
    
    // User wants to install - get paths
    let exe_path = std::env::current_exe()?;
    let exe_path_str = exe_path.display().to_string();
    
    // Check if we're running from an app bundle
    let is_app_bundle = exe_path_str.contains(".app/Contents/MacOS/");
    let bundle_path = if is_app_bundle {
        // Extract the .app path from the full executable path
        if let Some(app_pos) = exe_path_str.find(".app/") {
            exe_path_str[..app_pos + 4].to_string()
        } else {
            exe_path_str.clone()
        }
    } else {
        exe_path_str.clone()
    };
    
    let working_dir = std::env::current_dir()?;
    let working_dir_str = working_dir.display().to_string();
    
    // Create the plist content with actual paths
    // Use the bundle path if running from .app, otherwise use the executable path
    let launch_path = if is_app_bundle {
        // For app bundles, use 'open' command to launch properly
        bundle_path.clone()
    } else {
        exe_path_str.clone()
    };
    
    let plist_content = if is_app_bundle {
        // Use 'open' command for app bundles to launch them properly
        // Don't set WorkingDirectory - the app will set it to Resources internally
        format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.clevernote.parakeet</string>
    
    <key>ProgramArguments</key>
<array>
    <string>CleverNote.app/Contents/MacOS/CleverNote</string>
</array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
<dict>
    <key>SuccessfulExit</key>
    <false/>
</dict>
    
    <key>StandardOutPath</key>
    <string>/tmp/clevernote.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/clevernote.err.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"#)
    } else {
        // Direct binary execution for non-bundle builds
        format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.clevernote.parakeet</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
    </array>
    
    <key>WorkingDirectory</key>
    <string>{}</string>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <true/>
    
    <key>StandardOutPath</key>
    <string>/tmp/clevernote.log</string>
    
    <key>StandardErrorPath</key>
    <string>/tmp/clevernote.err.log</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"#, exe_path_str, working_dir_str)
    };
    
    // Create LaunchAgents directory if it doesn't exist
    let launch_agents_dir = format!("{}/Library/LaunchAgents", home_dir);
    fs::create_dir_all(&launch_agents_dir)?;
    
    // Check if binary is code-signed
    let is_signed = std::process::Command::new("codesign")
        .args(&["-dv", &exe_path_str])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    if !is_signed {
        // Binary is not signed - show error and instructions
        let sign_msg = format!(
            "CleverNote needs to be code-signed to work with macOS Accessibility.\n\n\
             Please run these commands in Terminal:\n\n\
             cd {}\n\
             cargo build --release\n\
             codesign --force --deep --sign - ./target/release/parakeet\n\n\
             Then run CleverNote again to complete installation.",
            working_dir_str
        );
        
        show_dialog("Code Signing Required", &sign_msg, &["OK"]);
        
        // Create marker to skip next time
        if let Some(parent) = PathBuf::from(&first_run_marker).parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&first_run_marker, "needs_signing")?;
        
        std::process::exit(1);
    }
    
    // Write the plist file
    fs::write(&plist_path, plist_content)?;
    
    // Try to load the LaunchAgent
    let output = std::process::Command::new("launchctl")
        .args(&["load", &plist_path])
        .output()?;
    
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        eprintln!("Warning: Failed to load LaunchAgent: {}", error);
    }
    
    // Create marker file
    if let Some(parent) = PathBuf::from(&first_run_marker).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&first_run_marker, "installed")?;
    
    // Show accessibility permissions dialog
    let permission_path = if is_app_bundle {
        &bundle_path
    } else {
        &exe_path_str
    };
    
    let permission_file = if is_app_bundle {
        "CleverNote.app"
    } else {
        "parakeet"
    };
    
    let permission_msg = format!(
        "Installation complete! CleverNote is now running as a background service.\n\n\
         IMPORTANT: Grant Accessibility permissions:\n\n\
         1. System Settings → Privacy & Security → Accessibility\n\
         2. Click the lock to unlock\n\
         3. Click '+' button\n\
         4. Press Cmd+Shift+G and paste:\n   {}\n\
         5. Select '{}' and click Open\n\
         6. Enable the checkbox\n\n\
         After granting permissions, CleverNote will be ready to use!",
        permission_path, permission_file
    );
    
    show_dialog("Setup Complete", &permission_msg, &["Open System Settings"]);
    
    // Open System Settings
    let _ = std::process::Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")
        .spawn();
    
    // Don't exit - let the app continue running so user can grant permissions
    // The LaunchAgent will handle restarting if needed
    println!("💡 Continuing... Grant permissions in System Settings, then the hotkey will work.");
    
    Ok(())
}

#[cfg(target_os = "macos")]
fn show_dialog(title: &str, message: &str, buttons: &[&str]) -> i32 {
    use std::process::Command;
    
    // Build AppleScript dialog
    let mut script = format!(
        "display dialog \"{}\" with title \"{}\" buttons {{",
        message.replace("\"", "\\\""),
        title.replace("\"", "\\\"")
    );
    
    for (i, button) in buttons.iter().enumerate() {
        if i > 0 {
            script.push_str(", ");
        }
        script.push_str(&format!("\"{}\"", button.replace("\"", "\\\"")));
    }
    
    script.push_str("} default button 1");
    
    // Run the dialog
    let output = Command::new("osascript")
        .arg("-e")
        .arg(&script)
        .output();
    
    match output {
        Ok(result) => {
            if result.status.success() {
                let output_str = String::from_utf8_lossy(&result.stdout);
                // Check which button was clicked based on output
                if buttons.len() > 1 && output_str.contains(buttons[1]) {
                    return 1; // Second button clicked
                }
                return 0; // First button clicked
            }
            1 // Error or cancelled
        }
        Err(_) => 1, // Error
    }
}

fn get_config_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".clevernote")
}

#[cfg(target_os = "macos")]
fn check_and_offer_move_to_applications() -> Result<()> {
    // Check if we're running from an app bundle
    let exe_path = std::env::current_exe()?;
    let exe_str = exe_path.display().to_string();
    
    if !exe_str.contains(".app/Contents/MacOS/") {
        return Ok(()); // Not an app bundle, skip
    }
    
    // Extract the .app path
    let app_bundle_path = if let Some(app_pos) = exe_str.find(".app/") {
        PathBuf::from(&exe_str[..app_pos + 4])
    } else {
        return Ok(());
    };
    
    // Check if already in /Applications
    if app_bundle_path.starts_with("/Applications") {
        return Ok(());
    }
    
    // Check if we've already asked
    let config_dir = get_config_dir();
    let asked_marker = config_dir.join(".asked_move_to_applications");
    if asked_marker.exists() {
        return Ok(());
    }
    
    // Ask user if they want to move to Applications
    let message = format!(
        "For the best experience, CleverNote should be installed in your Applications folder.\n\n\
         Current location: {}\n\n\
         Would you like to move it to /Applications now?",
        app_bundle_path.display()
    );
    
    let result = show_dialog("Move to Applications?", &message, &["Move to Applications", "Keep Here"]);
    
    // Mark that we've asked
    fs::write(&asked_marker, "asked")?;
    
    if result == 0 {
        // User wants to move - show instructions since we can't move ourselves
        let instructions = format!(
            "Please follow these steps:\n\n\
             1. Quit CleverNote (click menu bar icon → Quit)\n\
             2. Drag CleverNote.app to your Applications folder\n\
             3. Launch CleverNote from Applications\n\n\
             The app will then update its settings automatically.",
        );
        
        show_dialog("How to Move", &instructions, &["OK"]);
        
        // Exit so user can move the app
        std::process::exit(0);
    }
    
    Ok(())
}

fn main() -> Result<()> {
    // Ensure config directory exists
    let config_dir = get_config_dir();
    fs::create_dir_all(&config_dir)?;
    
    // Set working directory to ~/.clevernote/
    // This is where models, config, and transcripts will be stored
    std::env::set_current_dir(&config_dir)?;
    eprintln!("📂 Working directory: {}", config_dir.display());
    
    // Check if app should be moved to Applications folder
    #[cfg(target_os = "macos")]
    check_and_offer_move_to_applications()?;
    
    // Check accessibility permissions on macOS
    #[cfg(target_os = "macos")]
    check_accessibility_permissions();
    
    // Check if LaunchAgent should be installed (first launch)
    #[cfg(target_os = "macos")]
    check_and_offer_installation()?;
    
    let args = Args::parse();

    // Determine model path and ensure it exists
    let model_path = match &args.model {
        Some(path) => path.clone(),
        None => {
            let default_path = model_downloader::get_default_model_path();
            default_path.to_string_lossy().to_string()
        }
    };
    
    // Check if model exists, offer to download if missing
    let model_path = model_downloader::ensure_model_exists(&model_path, args.tdt)?;
    let model_path_str = model_path.to_string_lossy().to_string();
    
    println!("📦 Using model: {}", model_path_str);

    // Load configuration
    let config = Config::load(&args.config_file)?;
    println!("⚙️  Hotkey: {}+{}", config.modifier_key, config.trigger_key);

    // Create output directory
    fs::create_dir_all(&args.output_dir)
        .wrap_err("Failed to create output directory")?;

    // Create tao event loop FIRST (required for macOS hotkey support)
    let event_loop = EventLoopBuilder::new().build();
    
    // CRITICAL: Set activation policy AFTER event loop creation but BEFORE window creation
    // This must be done after EventLoop::new() to work properly with tao
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let app = cocoa::appkit::NSApp();
            // NSApplicationActivationPolicyAccessory = 1
            // This prevents the app from appearing in Dock and from stealing focus
            let _: () = msg_send![app, setActivationPolicy: 1i64];
        }
    }
    
    // Create circular recording indicator window (hidden initially)
    let recording_window = WindowBuilder::new()
        .with_title("")
        .with_inner_size(LogicalSize::new(25.0, 25.0))
        .with_decorations(false)
        .with_transparent(true)
        .with_always_on_top(true)
        .with_focusable(false)
        .with_visible(false)
        .build(&event_loop)
        .unwrap();
    
    // Set window to be circular and non-focusing using macOS native API
    #[cfg(target_os = "macos")]
    {
        use tao::platform::macos::WindowExtMacOS;
        
        let ns_window = recording_window.ns_window() as id;
        unsafe {
            // Prevent window from stealing focus - CRITICAL settings
            // NSFloatingWindowLevel = 3 (floats above normal windows)
            let _: () = msg_send![ns_window, setLevel: 3i32];
            
            // Set collection behavior to prevent activation and focus stealing
            // NSWindowCollectionBehaviorCanJoinAllSpaces = 1 << 0 (appear on all spaces)
            // NSWindowCollectionBehaviorStationary = 1 << 4 (don't switch spaces)
            // NSWindowCollectionBehaviorIgnoresCycle = 1 << 6 (ignore Cmd+Tab)
            let collection_behavior: i32 = (1 << 0) | (1 << 4) | (1 << 6);
            let _: () = msg_send![ns_window, setCollectionBehavior: collection_behavior];
            
            // Ignore mouse events so it doesn't steal focus
            let _: () = msg_send![ns_window, setIgnoresMouseEvents: 1i32];
            
            // Make window non-opaque for transparency
            let _: () = msg_send![ns_window, setOpaque: NO];
            
            // Prevent the window from becoming key or main window
            let _: () = msg_send![ns_window, setStyleMask: 0u64];
            
            // Get the content view and make it circular with corner radius
            let content_view: id = msg_send![ns_window, contentView];
            let _: () = msg_send![content_view, setWantsLayer: 1i32];
            let layer: id = msg_send![content_view, layer];
            
            // Set corner radius to half the size to make it circular
            let _: () = msg_send![layer, setCornerRadius: 12.5f64];
            let _: () = msg_send![layer, setMasksToBounds: 1i32];
            
            // Set initial red background on the layer (fully transparent initially)
            let red_color = NSColor::colorWithRed_green_blue_alpha_(
                nil,
                1.0,
                0.0,
                0.0,
                0.0,
            );
            let cg_color: id = msg_send![red_color, CGColor];
            let _: () = msg_send![layer, setBackgroundColor: cg_color];
        }
    }
    
    let recording_window = Arc::new(recording_window);
    let recording_start_time = Arc::new(Mutex::new(None::<Instant>));
    
    // Create tray icon menu
    let tray_menu = Menu::new();
    let status_item = MenuItem::new(format!("Hotkey: {}+{}", config.modifier_key, config.trigger_key), false, None);
    let model_item = MenuItem::new(format!("Model: {}", model_path_str.split('/').last().unwrap_or(&model_path_str)), false, None);
    
    // Hotkey configuration submenu
    let change_hotkey_item = MenuItem::new("Change Hotkey...", true, None);
    
    let quit_item = MenuItem::new("Quit", true, None);
    
    tray_menu.append(&status_item).unwrap();
    tray_menu.append(&model_item).unwrap();
    tray_menu.append(&change_hotkey_item).unwrap();
    tray_menu.append(&quit_item).unwrap();
    
    // Create tray icon (using built-in icon for now)
    let icon = load_icon();
    let _tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(tray_menu))
        .with_tooltip("CleverNote - Voice Transcription")
        .with_icon(icon)
        .build()
        .unwrap();
    
    let menu_channel = MenuEvent::receiver();
    
    // Parse hotkey configuration
    let modifiers = parse_modifier_key(&config.modifier_key);
    let key_code = parse_trigger_key(&config.trigger_key);
    
    // Create global hotkey manager
    println!("🔑 Registering global hotkey...");
    let manager = GlobalHotKeyManager::new().wrap_err("Failed to initialize global hotkey manager")?;
    
    // Register the hotkey
    let hotkey = HotKey::new(Some(modifiers), key_code);
    match manager.register(hotkey) {
        Ok(_) => {
            println!("✅ Hotkey registered successfully!");
        }
        Err(e) => {
            let exe_path = std::env::current_exe()
                .unwrap_or_default()
                .display()
                .to_string();
            
            eprintln!("❌ Failed to register hotkey: {}", e);
            eprintln!("\n⚠️  ACCESSIBILITY PERMISSIONS REQUIRED ⚠️");
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!();
            eprintln!("This app needs Accessibility permissions to:");
            eprintln!("  • Capture global hotkeys ({}+{})", config.modifier_key, config.trigger_key);
            eprintln!("  • Paste transcribed text automatically");
            eprintln!();
            eprintln!("📍 IMPORTANT: Add THIS specific binary to Accessibility:");
            eprintln!("   {}", exe_path);
            eprintln!();
            eprintln!("To grant permissions:");
            eprintln!("  1. Open System Settings (or System Preferences)");
            eprintln!("  2. Go to 'Privacy & Security' → 'Accessibility'");
            eprintln!("  3. Click the lock icon (🔒) to unlock");
            eprintln!("  4. Click the '+' button");
            eprintln!("  5. Navigate to and select:");
            eprintln!("     {}", exe_path);
            eprintln!("  6. Ensure the checkbox next to it is enabled");
            eprintln!("  7. Restart this app");
            eprintln!();
            eprintln!("Quick command to open System Settings:");
            eprintln!("  open 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'");
            eprintln!();
            eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            eprintln!();
            return Err(e.into());
        }
    }
    
    // Setup hotkey event receiver
    let hotkey_rx = GlobalHotKeyEvent::receiver();
    let hotkey_id = hotkey.id();

    println!("🔊 Loading ASR model (this may take a bit)...");
    let start = Instant::now();

    // Load model based on type
    let model_type = if args.tdt { "TDT" } else { "CTC" };
    println!("📦 Loading {} model from: {}", model_type, model_path_str);

    // Create channel for sending transcription jobs
    let (tx, rx) = channel::<TranscriptionJob>();

    // NOW spawn dedicated transcription thread with loaded model
    let model_path_for_thread = model_path_str.clone();
    let use_tdt = args.tdt;
    std::thread::spawn(move || {
        // Load model once in this thread
        if use_tdt {
            let mut parakeet = match ParakeetTDT::from_pretrained(&model_path_for_thread, None) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("❌ Failed to load TDT model: {}", e);
                    return;
                }
            };
            println!("✅ TDT Model loaded in {:.2}s!", start.elapsed().as_secs_f32());

            // Process jobs as they arrive
            while let Ok(job) = rx.recv() {
                if let Err(e) = process_recording_tdt(&mut parakeet, job) {
                    eprintln!("❌ Transcription error: {}", e);
                }
            }
        } else {
            let mut parakeet = match Parakeet::from_pretrained(&model_path_for_thread, None) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("❌ Failed to load CTC model: {}", e);
                    return;
                }
            };
            println!("✅ CTC Model loaded in {:.2}s!", start.elapsed().as_secs_f32());

            // Process jobs as they arrive
            while let Ok(job) = rx.recv() {
                if let Err(e) = process_recording_ctc(&mut parakeet, job) {
                    eprintln!("❌ Transcription error: {}", e);
                }
            }
        }
    });

    // Give the model loading thread a moment to load
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Audio setup - prefer internal microphone over AirPods/Bluetooth
    let host = cpal::default_host();
    
    // Try to find internal microphone first
    let device = host
        .input_devices()
        .ok()
        .and_then(|devices| {
            // Look for built-in/internal microphone
            devices
                .filter_map(|d| {
                    let name = d.name().ok()?;
                    // Prefer internal mic (avoid AirPods, Bluetooth, etc.)
                    if name.to_lowercase().contains("built-in") 
                        || name.to_lowercase().contains("internal") 
                        || name.to_lowercase().contains("macbook")
                    {
                        Some((d, 0)) // Priority 0 (highest)
                    } else if !name.to_lowercase().contains("airpods") 
                        && !name.to_lowercase().contains("bluetooth")
                    {
                        Some((d, 1)) // Priority 1 (medium)
                    } else {
                        Some((d, 2)) // Priority 2 (lowest - AirPods/BT)
                    }
                })
                .min_by_key(|(_, priority)| *priority)
                .map(|(device, _)| device)
        })
        .or_else(|| host.default_input_device())
        .ok_or_else(|| eyre::eyre!("No input device available"))?;
    
    let default_config = device
        .default_input_config()
        .wrap_err("Failed to get default input config")?;

    let device_sample_rate = default_config.sample_rate().0;
    let target_sample_rate = args.sample_rate;

    println!("🎤 Audio device: {}", device.name().unwrap_or("Unknown".to_string()));
    println!("📊 Device sample rate: {} Hz", device_sample_rate);
    println!("🎯 Target sample rate: {} Hz (will resample if needed)", target_sample_rate);

    // Shared state for recording
    let state = Arc::new(Mutex::new(RecordingState::new(device_sample_rate)));
    let state_clone = Arc::clone(&state);
    let state_keyboard = Arc::clone(&state);

    // Use device's native config for recording
    let stream_config = default_config.config();

    // Create audio stream based on device's native sample format
    // Create it NOW, before the tao event loop, to avoid macOS dialog issues
    let stream = match default_config.sample_format() {
        cpal::SampleFormat::F32 => {
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock().unwrap();
                    if state.is_recording {
                        state.audio_data.extend_from_slice(data);
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut state = state_clone.lock().unwrap();
                    if state.is_recording {
                        // Convert i16 to f32
                        let samples: Vec<f32> = data.iter()
                            .map(|&s| s as f32 / i16::MAX as f32)
                            .collect();
                        state.audio_data.extend(samples);
                    }
                },
                |err| eprintln!("Stream error: {}", err),
                None,
            )?
        }
        _ => return Err(eyre::eyre!("Unsupported sample format")),
    };

    // Pre-warm the audio stream to avoid delays (especially on AirPods)
    let stream = Arc::new(Mutex::new(stream));
    let stream_for_hotkey = Arc::clone(&stream);
    
    // ✅ PRE-WARM: Start and immediately pause to initialize audio hardware
    // This eliminates the ~2 second delay when first pressing the hotkey
    println!("🔧 Pre-warming audio stream...");
    if let Ok(stream_lock) = stream.lock() {
        match stream_lock.play() {
            Ok(_) => {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let _ = stream_lock.pause();
                println!("✅ Audio stream ready (pre-warmed)");
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Could not pre-warm audio stream: {}", e);
            }
        }
    }
    
    println!("🚀 Ready! Press {}+{} to start recording, press again to stop.", 
             config.modifier_key, config.trigger_key);
    println!("   Check the menu bar icon for options\n");

    let recording_window_for_hotkey = Arc::clone(&recording_window);
    let recording_window_for_pulse = Arc::clone(&recording_window);
    let recording_start_time_for_loop = Arc::clone(&recording_start_time);
    let recording_start_time_for_hotkey = Arc::clone(&recording_start_time);

    // Run the tao event loop - this is required for macOS hotkey events
    event_loop.run(move |event, _, control_flow| {
        // Only poll continuously when recording (for animation), otherwise wait for events to save CPU
        *control_flow = if recording_window_for_pulse.is_visible() {
            ControlFlow::Poll
        } else {
            ControlFlow::Wait
        };
        
        // Handle pulsing animation when window is visible
        if recording_window_for_pulse.is_visible() {
            if let Some(start) = *recording_start_time_for_loop.lock().unwrap() {
                #[cfg(target_os = "macos")]
                {
                    use tao::platform::macos::WindowExtMacOS;
                    let elapsed = start.elapsed().as_secs_f32();
                    let pulse = (elapsed * 2.0).sin() * 0.475 + 0.525; // Pulses between 0.05 and 1.0
                    
                    let ns_window = recording_window_for_pulse.ns_window() as id;
                    unsafe {
                        let content_view: id = msg_send![ns_window, contentView];
                        let layer: id = msg_send![content_view, layer];
                        
                        // Update the layer's background color alpha
                        let red_color = NSColor::colorWithRed_green_blue_alpha_(
                            nil,
                            1.0,
                            0.0,
                            0.0,
                            pulse as f64,
                        );
                        let cg_color: id = msg_send![red_color, CGColor];
                        let _: () = msg_send![layer, setBackgroundColor: cg_color];
                    }
                }
                recording_window_for_pulse.request_redraw();
            }
        }
        
        // Process window events to ensure window updates work
        match event {
            tao::event::Event::WindowEvent { event, .. } => {
                match event {
                    _ => {}
                }
            }
            tao::event::Event::RedrawRequested(_) => {
                // Redraw handled above in the pulsing animation
            }
            tao::event::Event::MainEventsCleared => {
                if recording_window_for_pulse.is_visible() {
                    recording_window_for_pulse.request_redraw();
                }
            }
            _ => {}
        }
        
        // Handle menu events
        if let Ok(event) = menu_channel.try_recv() {
            if event.id == quit_item.id() {
                println!("👋 Shutting down...");
                *control_flow = ControlFlow::Exit;
            } else if event.id == change_hotkey_item.id() {
                println!("💡 To change hotkey, edit config.toml and restart the app");
                println!("   Current hotkey: {}+{}", config.modifier_key, config.trigger_key);
            }
        }

        // Poll for hotkey events
        if let Ok(event) = hotkey_rx.try_recv() {
            if event.id == hotkey_id {
                let is_currently_recording = {
                    let state = state_keyboard.lock().unwrap();
                    state.is_recording
                };
                
                // Toggle recording on/off with each hotkey press
                if !is_currently_recording {
                    // Start recording
                    {
                        let mut state = state_keyboard.lock().unwrap();
                        state.is_recording = true;
                        state.audio_data.clear();
                    }
                    
                    // Start audio stream
                    if let Ok(stream) = stream_for_hotkey.lock() {
                        match stream.play() {
                            Ok(_) => println!("🔊 Audio stream started"),
                            Err(e) => eprintln!("❌ Failed to start stream: {}", e),
                        }
                    }
                    
                    // Position window on current screen before showing
                    if let Some(monitor) = recording_window_for_hotkey.current_monitor().or_else(|| recording_window_for_hotkey.primary_monitor()) {
                        let screen_size = monitor.size();
                        let scale_factor = monitor.scale_factor();
                        let monitor_position = monitor.position();
                        
                        // Position in top-right corner using physical pixels, relative to current monitor
                        let window_width_physical = (25.0 * scale_factor) as i32;
                        let x = monitor_position.x + screen_size.width as i32 - window_width_physical - 25;
                        let y = monitor_position.y + 20;
                        
                        recording_window_for_hotkey.set_outer_position(PhysicalPosition::new(x, y));
                    }
                    
                    recording_window_for_hotkey.set_visible_on_all_workspaces(true);
                    
                    // Show window without stealing focus using macOS-specific API
                    #[cfg(target_os = "macos")]
                    {
                        use tao::platform::macos::WindowExtMacOS;
                        let ns_window = recording_window_for_hotkey.ns_window() as id;
                        unsafe {
                            // Get the currently active application to restore focus later
                            let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
                            let active_app: id = msg_send![workspace, frontmostApplication];
                            
                            // Critical: Set hidesOnDeactivate to NO so window stays visible
                            // when we prevent app activation
                            let _: () = msg_send![ns_window, setHidesOnDeactivate: 0i32];
                            
                            // Use orderWindow:relativeTo: with NSWindowAbove to show without activating
                            // NSWindowAbove = 1, relativeTo: 0 means "show above all at this level"
                            let _: () = msg_send![ns_window, orderWindow: 1i32 relativeTo: 0i32];
                            
                            // Immediately restore focus to the previously active application
                            if !active_app.is_null() {
                                // activateWithOptions: with NSApplicationActivateIgnoringOtherApps = 1 << 1
                                let _: bool = msg_send![active_app, activateWithOptions: 2i32];
                            }
                        }
                    }
                    
                    #[cfg(not(target_os = "macos"))]
                    recording_window_for_hotkey.set_visible(true);
                    
                    // Start pulse timer
                    *recording_start_time_for_hotkey.lock().unwrap() = Some(Instant::now());
                    
                    println!("🎙️  Recording... Speak now.");
                } else {
                    // Stop recording and get audio data
                    let audio_data = {
                        let mut state = state_keyboard.lock().unwrap();
                        state.is_recording = false;
                        let data = state.audio_data.clone();
                        state.audio_data.clear();
                        data
                    };
                    
                    // Stop pulse timer and hide window
                    *recording_start_time_for_hotkey.lock().unwrap() = None;
                    recording_window_for_hotkey.set_visible(false);
                    
                    println!("🛑 Recording stopped. Transcribing...");
                    
                    // Stop audio stream
                    if let Ok(stream) = stream_for_hotkey.lock() {
                        let _ = stream.pause();
                    }
                    
                    // Check if we have audio data
                    println!("📊 Captured {} audio samples", audio_data.len());
                    if !audio_data.is_empty() {
                        // Create transcription job
                        let job = TranscriptionJob {
                            audio_data: audio_data,
                            device_sample_rate: {
                                let state = state_keyboard.lock().unwrap();
                                state.device_sample_rate
                            },
                            target_sample_rate: args.sample_rate,
                            output_dir: args.output_dir.clone(),
                            keep_audio: args.keep_audio,
                        };
                        
                        // Send job to transcription thread
                        println!("📤 Sending transcription job to thread...");
                        if let Err(e) = tx.send(job) {
                            eprintln!("❌ Failed to send transcription job: {}", e);
                        } else {
                            println!("✅ Transcription job sent successfully");
                        }
                    } else {
                        println!("⚠️  No audio data recorded");
                    }
                }
            }
        }
    });
}

fn process_recording_tdt(parakeet: &mut ParakeetTDT, job: TranscriptionJob) -> Result<()> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let audio_path = format!("recording_{}.wav", timestamp);

    // Resample audio if needed
    let resampled_audio = if job.device_sample_rate != job.target_sample_rate {
        println!("🔄 Resampling from {} Hz to {} Hz...", job.device_sample_rate, job.target_sample_rate);
        resample_audio(&job.audio_data, job.device_sample_rate, job.target_sample_rate)
    } else {
        job.audio_data
    };

    // Save audio file at target sample rate
    let spec = WavSpec {
        channels: 1,
        sample_rate: job.target_sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(&audio_path, spec)?;
    for sample in &resampled_audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    println!("💾 Saved audio to: {}", audio_path);

    // Transcribe using samples directly (model is already loaded)
    let transcription_start = Instant::now();
    let result = parakeet.transcribe_samples(
        resampled_audio.clone(),
        job.target_sample_rate,
        1, // mono
        Some(TimestampMode::Sentences)
    )?;
    
    println!("\nSentences:");
    for segment in result.tokens.iter() {
        println!("[{:.2}s - {:.2}s]: {}", segment.start, segment.end, segment.text);
    }
    
    let text = result.text;
    let transcription_time = transcription_start.elapsed().as_secs_f32();
    println!("⚡ Transcribed in {:.2}s", transcription_time);

    // Save transcript
    let transcript_path = PathBuf::from(&job.output_dir)
        .join(format!("transcript_{}.txt", timestamp));
    fs::write(&transcript_path, &text)?;

    println!("💾 Saved transcript to: {}", transcript_path.display());
    println!("💬 Transcript: {}...", text.chars().take(100).collect::<String>());

    // Delete audio file if not keeping
    if !job.keep_audio {
        fs::remove_file(&audio_path)?;
        println!("🗑️  Deleted audio file");
    }

    // Copy to clipboard and paste
    copy_and_paste(&text)?;

    // Note: Notifications disabled to avoid macOS application picker dialog

    Ok(())
}

fn process_recording_ctc(parakeet: &mut Parakeet, job: TranscriptionJob) -> Result<()> {
    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
    let audio_path = format!("recording_{}.wav", timestamp);

    // Resample audio if needed
    let resampled_audio = if job.device_sample_rate != job.target_sample_rate {
        println!("🔄 Resampling from {} Hz to {} Hz...", job.device_sample_rate, job.target_sample_rate);
        resample_audio(&job.audio_data, job.device_sample_rate, job.target_sample_rate)
    } else {
        job.audio_data
    };

    // Save audio file at target sample rate
    let spec = WavSpec {
        channels: 1,
        sample_rate: job.target_sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(&audio_path, spec)?;
    for sample in &resampled_audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    println!("💾 Saved audio to: {}", audio_path);

    // Transcribe using samples directly (model is already loaded)
    let transcription_start = Instant::now();
    let result = parakeet.transcribe_samples(
        resampled_audio.clone(),
        job.target_sample_rate,
        1, // mono
        Some(TimestampMode::Words)
    )?;
    
    println!("\nWords (first 10):");
    for word in result.tokens.iter().take(10) {
        println!("[{:.2}s - {:.2}s]: {}", word.start, word.end, word.text);
    }
    
    let text = result.text;
    let transcription_time = transcription_start.elapsed().as_secs_f32();
    println!("⚡ Transcribed in {:.2}s", transcription_time);

    // Save transcript
    let transcript_path = PathBuf::from(&job.output_dir)
        .join(format!("transcript_{}.txt", timestamp));
    fs::write(&transcript_path, &text)?;

    println!("💾 Saved transcript to: {}", transcript_path.display());
    println!("💬 Transcript: {}...", text.chars().take(100).collect::<String>());

    // Delete audio file if not keeping
    if !job.keep_audio {
        fs::remove_file(&audio_path)?;
        println!("🗑️  Deleted audio file");
    }

    // Copy to clipboard and paste
    copy_and_paste(&text)?;

    // Note: Notifications disabled to avoid macOS application picker dialog

    Ok(())
}

fn load_icon() -> tray_icon::Icon {
    // Create a simple 32x32 microphone icon using RGBA data
    let width = 32;
    let height = 32;
    let mut rgba = vec![0u8; width * height * 4];
    
    // Draw a simple microphone shape
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 4;
            
            // Simple mic icon: white mic on transparent background
            let is_mic = (x >= 12 && x <= 19 && y >= 8 && y <= 20) || // body
                         (x >= 10 && x <= 21 && y >= 20 && y <= 22) || // stand
                         (x >= 8 && x <= 23 && y >= 22 && y <= 24);     // base
            
            if is_mic {
                rgba[idx] = 255;     // R
                rgba[idx + 1] = 255; // G
                rgba[idx + 2] = 255; // B
                rgba[idx + 3] = 255; // A
            } else {
                rgba[idx + 3] = 0; // Transparent
            }
        }
    }
    
    tray_icon::Icon::from_rgba(rgba, width as u32, height as u32).unwrap()
}

fn copy_and_paste(text: &str) -> Result<()> {
    // Copy to clipboard
    let mut clipboard = Clipboard::new()?;
    clipboard.set_text(text)?;
    println!("📋 Copied to clipboard");

    // Simulate paste (Cmd+V on macOS)
    std::thread::sleep(std::time::Duration::from_millis(100));
    let mut enigo = Enigo::new(&Settings::default())?;
    
    #[cfg(target_os = "macos")]
    {
        enigo.key(Key::Meta, Direction::Press)?;
        enigo.key(Key::Unicode('v'), Direction::Click)?;
        enigo.key(Key::Meta, Direction::Release)?;
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        enigo.key(Key::Control, Direction::Press)?;
        enigo.key(Key::Unicode('v'), Direction::Click)?;
        enigo.key(Key::Control, Direction::Release)?;
    }

    println!("⌨️  Pasted text");
    Ok(())
}









