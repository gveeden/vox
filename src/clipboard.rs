/// Clipboard and paste functionality
use arboard::Clipboard;
use eyre::Result;
use log::warn;

#[cfg(not(target_os = "linux"))]
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

#[cfg(target_os = "linux")]
use crate::inject_linux;

#[cfg(target_os = "linux")]
use std::process::Command;

/// Copy text to clipboard and optionally inject it
pub fn copy_and_paste(text: &str, auto_paste: bool, auto_inject: bool) -> Result<()> {
    // Copy to clipboard as backup
    #[cfg(target_os = "linux")]
    {
        // Check if we're on Wayland
        if std::env::var("WAYLAND_DISPLAY").is_ok() {
            // Use wl-copy for Wayland
            match Command::new("wl-copy").arg(text).status() {
                Ok(status) if status.success() => {
                    println!("📋 Copied to clipboard (wl-copy)");
                }
                _ => {
                    warn!("wl-copy failed, falling back to arboard");
                    let mut clipboard = Clipboard::new()?;
                    clipboard.set_text(text)?;
                    println!("📋 Copied to clipboard");
                }
            }
        } else {
            // Use arboard for X11
            let mut clipboard = Clipboard::new()?;
            clipboard.set_text(text)?;
            println!("📋 Copied to clipboard");
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let mut clipboard = Clipboard::new()?;
        clipboard.set_text(text)?;
        println!("📋 Copied to clipboard");
    }

    // CRITICAL: Wait for clipboard to fully sync before attempting paste
    use std::thread;
    use std::time::Duration;
    thread::sleep(Duration::from_millis(200));

    // Auto-paste using system tools
    #[cfg(target_os = "linux")]
    {
        if auto_paste {
            if auto_inject {
                // Try evdev text injection (requires CAP_DAC_OVERRIDE)
                match inject_linux::inject_text(text) {
                    Ok(_) => {
                        println!("⌨️  Typed text (evdev+XKB)");
                    }
                    Err(e) => {
                        warn!("Failed to auto-type via evdev: {}", e);
                        println!("⚠️  Auto-type failed - Press Ctrl+V to paste");
                        println!("   To enable: sudo setcap \"cap_dac_override+p\" $(which clevernote-daemon)");
                    }
                }
            } else {
                // Try to simulate paste with evdev (detects terminal vs regular app)
                match inject_linux::inject_paste() {
                    Ok(_) => {
                        println!("⌨️  Auto-pasted");
                    }
                    Err(e) => {
                        warn!("Failed to auto-paste: {}", e);
                        println!("💡 Press Ctrl+V to paste (or Ctrl+Shift+V in terminals)");
                    }
                }
            }
        } else {
            println!("💡 Press Ctrl+V to paste (or Ctrl+Shift+V in terminals)");
        }
    }

    #[cfg(target_os = "macos")]
    {
        if auto_paste {
            let mut enigo = Enigo::new(&Settings::default())?;
            enigo.key(Key::Meta, Direction::Press)?;
            enigo.key(Key::Unicode('v'), Direction::Click)?;
            enigo.key(Key::Meta, Direction::Release)?;
            println!("⌨️  Pasted text");
        } else {
            println!("💡 Press Cmd+V to paste");
        }
    }

    #[cfg(target_os = "windows")]
    {
        if auto_paste {
            let mut enigo = Enigo::new(&Settings::default())?;
            enigo.key(Key::Control, Direction::Press)?;
            enigo.key(Key::Unicode('v'), Direction::Click)?;
            enigo.key(Key::Control, Direction::Release)?;
            println!("⌨️  Pasted text");
        } else {
            println!("💡 Press Ctrl+V to paste");
        }
    }

    Ok(())
}
