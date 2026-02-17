/// Recording overlay using notify-send with proper D-Bus close
use std::process::Command;
use std::sync::{atomic::AtomicBool, Arc, Mutex, OnceLock};

// Store the notification ID so we can close it later
static NOTIFICATION_ID: OnceLock<Mutex<Option<u32>>> = OnceLock::new();

pub fn show_recording_overlay() -> Arc<AtomicBool> {
    // Create persistent notification and capture its ID
    let output = Command::new("notify-send")
        .args(&[
            "--print-id",
            "--urgency=low",
            "--expire-time=0", // Persistent
            "--app-name=CleverNote",
            "--icon=media-record",
            "🔴 Recording",
            "Recording in progress...",
        ])
        .output();

    // Store the notification ID
    if let Ok(output) = output {
        if let Ok(id_str) = String::from_utf8(output.stdout) {
            if let Ok(id) = id_str.trim().parse::<u32>() {
                let id_lock = NOTIFICATION_ID.get_or_init(|| Mutex::new(None));
                *id_lock.lock().unwrap() = Some(id);
                log::info!("Notification created with ID: {}", id);
            }
        }
    }

    Arc::new(AtomicBool::new(false))
}

pub fn hide_recording_overlay(_should_close: Arc<AtomicBool>) {
    // Close the notification via D-Bus using dbus-send (faster than gdbus)
    log::info!("hide_recording_overlay: Entry");

    if let Some(id_lock) = NOTIFICATION_ID.get() {
        log::info!("hide_recording_overlay: Got NOTIFICATION_ID");

        // Acquire lock once and hold it for the entire operation
        let mut lock = id_lock.lock().unwrap();
        if let Some(id) = *lock {
            log::info!("Closing notification ID: {}", id);

            // Use dbus-send instead of gdbus - it's faster and doesn't output anything
            let spawn_result = Command::new("dbus-send")
                .args(&[
                    "--session",
                    "--dest=org.freedesktop.Notifications",
                    "--type=method_call",
                    "/org/freedesktop/Notifications",
                    "org.freedesktop.Notifications.CloseNotification",
                    &format!("uint32:{}", id),
                ])
                .spawn();

            log::info!("hide_recording_overlay: spawn result: {:?}", spawn_result);

            // Clear the stored ID (we already have the lock)
            *lock = None;
            log::info!("hide_recording_overlay: Cleared ID");
        }
        // Lock is automatically released when 'lock' goes out of scope
    }

    log::info!("hide_recording_overlay: Exit");
}
