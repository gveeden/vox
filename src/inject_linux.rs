/// Linux keyboard injection using evdev (/dev/uinput) with xkbcommon
/// This approach works on both X11 and Wayland
#[cfg(target_os = "linux")]
use evdev::{uinput::VirtualDeviceBuilder, AttributeSet, EventType, Key};
use eyre::Result;
use std::collections::{HashMap, HashSet};
use std::thread;
use std::time::Duration;

#[cfg(target_os = "linux")]
use xkbcommon::xkb;

// Offset between evdev keycodes (where KEY_ESCAPE is 1), and the evdev XKB
// keycode set (where ESC is 9).
const EVDEV_OFFSET: u32 = 8;

#[cfg(target_os = "linux")]
#[derive(Clone, Debug)]
struct KeyRecord {
    code: u32,           // evdev keycode
    modifiers: Vec<u32>, // List of modifier keycodes (Shift, etc)
}

#[cfg(target_os = "linux")]
type CharMap = HashMap<char, KeyRecord>;

#[cfg(target_os = "linux")]
fn build_char_map() -> Result<CharMap> {
    let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
    let keymap = xkb::Keymap::new_from_names(
        &context,
        "",   // rules
        "",   // model
        "",   // layout (empty = use system default)
        "",   // variant
        None, // options
        xkb::KEYMAP_COMPILE_NO_FLAGS,
    )
    .ok_or_else(|| eyre::eyre!("Failed to create XKB keymap"))?;

    let state = xkb::State::new(&keymap);
    let mut char_map = HashMap::new();

    // Modifier key codes (evdev codes)
    let shift_keys = vec![42, 54]; // KEY_LEFTSHIFT, KEY_RIGHTSHIFT

    // Iterate through all keycodes and modifier combinations
    for keycode in 8..256u32 {
        // Try without modifiers
        let keysym = state.key_get_one_sym(xkb::Keycode::new(keycode));
        let c = xkb::keysym_to_utf32(keysym);
        if c > 0 && c < 0x110000 {
            if let Some(ch) = char::from_u32(c) {
                char_map.entry(ch).or_insert_with(|| KeyRecord {
                    code: keycode - EVDEV_OFFSET,
                    modifiers: vec![],
                });
            }
        }

        // Try with Shift
        for shift_key in &shift_keys {
            let mut state_with_shift = xkb::State::new(&keymap);
            state_with_shift.update_key(
                xkb::Keycode::new(shift_key + EVDEV_OFFSET),
                xkb::KeyDirection::Down,
            );

            let keysym = state_with_shift.key_get_one_sym(xkb::Keycode::new(keycode));
            let c = xkb::keysym_to_utf32(keysym);
            if c > 0 && c < 0x110000 {
                if let Some(ch) = char::from_u32(c) {
                    char_map.entry(ch).or_insert_with(|| KeyRecord {
                        code: keycode - EVDEV_OFFSET,
                        modifiers: vec![*shift_key],
                    });
                }
            }
        }
    }

    Ok(char_map)
}

#[cfg(target_os = "linux")]
pub fn inject_text(text: &str) -> Result<()> {
    use log::{info, warn};

    // Build character map using XKB
    let char_map = build_char_map()?;

    // Verify all characters can be mapped
    for c in text.chars() {
        if !char_map.contains_key(&c) && !c.is_whitespace() {
            warn!("Character '{}' not found in keymap, will be skipped", c);
        }
    }

    // Temporarily raise CAP_DAC_OVERRIDE to open /dev/uinput
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};

        if let Err(e) = caps::raise(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE) {
            return Err(eyre::eyre!(
                "Failed to raise CAP_DAC_OVERRIDE capability: {}\n\
                 Run: sudo setcap \"cap_dac_override+p\" $(which clevernote-daemon)",
                e
            ));
        }
    }

    // Create virtual keyboard device with all needed keys
    let mut keys = AttributeSet::<Key>::new();

    // Add all letter keys (evdev keycodes 1-83 cover most keyboard keys)
    for key_code in 1..=83 {
        let key = Key(key_code);
        keys.insert(key);
    }

    // Add modifiers
    keys.insert(Key::KEY_LEFTSHIFT);
    keys.insert(Key::KEY_RIGHTSHIFT);
    keys.insert(Key::KEY_LEFTCTRL);
    keys.insert(Key::KEY_RIGHTCTRL);
    keys.insert(Key::KEY_LEFTALT);
    keys.insert(Key::KEY_RIGHTALT);

    let device_result = VirtualDeviceBuilder::new()?
        .name("CleverNote Virtual Keyboard")
        .with_keys(&keys)?
        .build();

    // Drop capability after opening device
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};
        let _ = caps::drop(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE);
    }

    let mut device = device_result?;

    info!("Created virtual keyboard device with XKB mapping");

    // Delay to let device register with the system
    thread::sleep(Duration::from_millis(100));

    // Track currently pressed modifiers to minimize state changes
    let mut current_modifiers: HashSet<u32> = HashSet::new();

    // Type each character
    for c in text.chars() {
        if let Some(record) = char_map.get(&c) {
            let record_modifiers: HashSet<u32> = record.modifiers.iter().copied().collect();

            // Release modifiers that are no longer needed
            for expired_modifier in current_modifiers.difference(&record_modifiers) {
                device.emit(&[evdev::InputEvent::new(
                    EventType::KEY,
                    *expired_modifier as u16,
                    0,
                )])?;
                thread::sleep(Duration::from_millis(10));
            }

            // Press new modifiers
            for new_modifier in record_modifiers.difference(&current_modifiers) {
                device.emit(&[evdev::InputEvent::new(
                    EventType::KEY,
                    *new_modifier as u16,
                    1,
                )])?;
                thread::sleep(Duration::from_millis(10));
            }

            // Press key
            device.emit(&[evdev::InputEvent::new(
                EventType::KEY,
                record.code as u16,
                1,
            )])?;
            thread::sleep(Duration::from_millis(5));

            // Release key
            device.emit(&[evdev::InputEvent::new(
                EventType::KEY,
                record.code as u16,
                0,
            )])?;
            thread::sleep(Duration::from_millis(5));

            // Sync
            device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;

            current_modifiers = record_modifiers;
        }
    }

    // Release any remaining modifiers
    for modifier in current_modifiers {
        device.emit(&[evdev::InputEvent::new(EventType::KEY, modifier as u16, 0)])?;
        thread::sleep(Duration::from_millis(10));
    }

    info!("⌨️  Typed {} characters via evdev+XKB", text.len());

    Ok(())
}

/// Detect if the focused window is a terminal that needs Ctrl+Shift+V
#[cfg(target_os = "linux")]
fn is_terminal_window() -> bool {
    use std::process::Command;

    // Try to detect Hyprland active window
    if std::env::var("HYPRLAND_INSTANCE_SIGNATURE").is_ok() {
        if let Ok(output) = Command::new("hyprctl")
            .args(&["activewindow", "-j"])
            .output()
        {
            if let Ok(json_str) = String::from_utf8(output.stdout) {
                // Check for terminal window classes
                let terminal_classes = [
                    "alacritty",
                    "kitty",
                    "konsole",
                    "gnome-terminal",
                    "xterm",
                    "terminator",
                    "tilix",
                    "foot",
                    "wezterm",
                    "ghostty",
                    "com.mitchellh.ghostty",
                    "org.wezfurlong.wezterm",
                    "Alacritty",
                    "kitty",
                    "urxvt",
                    "st-256color",
                ];

                for term in &terminal_classes {
                    if json_str.to_lowercase().contains(&term.to_lowercase()) {
                        return true;
                    }
                }
            }
        }
    }

    // TODO: Add detection for other compositors (Sway, etc.)

    false
}

/// Inject Ctrl+V or Ctrl+Shift+V based on focused window
#[cfg(target_os = "linux")]
pub fn inject_paste() -> Result<()> {
    use log::info;

    let needs_shift = is_terminal_window();

    if needs_shift {
        info!("Detected terminal window, using Ctrl+Shift+V");
    } else {
        info!("Using Ctrl+V");
    }

    // Temporarily raise CAP_DAC_OVERRIDE to open /dev/uinput
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};

        if let Err(e) = caps::raise(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE) {
            return Err(eyre::eyre!(
                "Failed to raise CAP_DAC_OVERRIDE capability: {}\n\
                 Run: sudo setcap \"cap_dac_override+p\" $(which clevernote-daemon)",
                e
            ));
        }
    }

    // Create virtual keyboard with needed keys
    let mut keys = AttributeSet::<Key>::new();
    keys.insert(Key::KEY_LEFTCTRL);
    keys.insert(Key::KEY_V);
    if needs_shift {
        keys.insert(Key::KEY_LEFTSHIFT);
    }

    let device_result = VirtualDeviceBuilder::new()?
        .name("CleverNote Paste")
        .with_keys(&keys)?
        .build();

    // Drop capability after opening device
    #[cfg(target_os = "linux")]
    {
        use caps::{CapSet, Capability};
        let _ = caps::drop(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE);
    }

    let mut device = device_result?;

    info!("Virtual keyboard device created");

    // CRITICAL: Longer delay to let Wayland/Hyprland register and trust the device
    thread::sleep(Duration::from_millis(300));

    // Press Ctrl
    device.emit(&[evdev::InputEvent::new(
        EventType::KEY,
        Key::KEY_LEFTCTRL.code(),
        1,
    )])?;
    device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
    thread::sleep(Duration::from_millis(100));

    // Press Shift if needed (for terminals)
    if needs_shift {
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            Key::KEY_LEFTSHIFT.code(),
            1,
        )])?;
        device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
        thread::sleep(Duration::from_millis(100));
    }

    // Press V
    device.emit(&[evdev::InputEvent::new(EventType::KEY, Key::KEY_V.code(), 1)])?;
    device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
    thread::sleep(Duration::from_millis(100));

    // Release V
    device.emit(&[evdev::InputEvent::new(EventType::KEY, Key::KEY_V.code(), 0)])?;
    device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
    thread::sleep(Duration::from_millis(100));

    // Release Shift if pressed
    if needs_shift {
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            Key::KEY_LEFTSHIFT.code(),
            0,
        )])?;
        device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
        thread::sleep(Duration::from_millis(100));
    }

    // Release Ctrl
    device.emit(&[evdev::InputEvent::new(
        EventType::KEY,
        Key::KEY_LEFTCTRL.code(),
        0,
    )])?;
    device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;

    if needs_shift {
        info!("⌨️  Sent Ctrl+Shift+V via evdev");
    } else {
        info!("⌨️  Sent Ctrl+V via evdev");
    }

    Ok(())
}

/// Inject a single character (for ellipsis animation)
#[cfg(target_os = "linux")]
pub fn inject_char(c: char) -> Result<()> {
    use caps::{CapSet, Capability};

    // Raise capability
    caps::raise(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE)?;

    let char_map = build_char_map()?;

    if let Some(record) = char_map.get(&c) {
        let mut keys = AttributeSet::<Key>::new();
        keys.insert(Key(record.code as u16));
        for modifier in &record.modifiers {
            keys.insert(Key(*modifier as u16));
        }

        let mut device = VirtualDeviceBuilder::new()?
            .name("CleverNote Char")
            .with_keys(&keys)?
            .build()?;

        caps::drop(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE).ok();
        thread::sleep(Duration::from_millis(50));

        // Press modifiers
        for modifier in &record.modifiers {
            device.emit(&[evdev::InputEvent::new(EventType::KEY, *modifier as u16, 1)])?;
        }

        // Press key
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            record.code as u16,
            1,
        )])?;
        thread::sleep(Duration::from_millis(10));

        // Release key
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            record.code as u16,
            0,
        )])?;

        // Release modifiers
        for modifier in &record.modifiers {
            device.emit(&[evdev::InputEvent::new(EventType::KEY, *modifier as u16, 0)])?;
        }

        device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;
    }

    Ok(())
}

/// Inject backspace key
#[cfg(target_os = "linux")]
pub fn inject_backspace(count: usize) -> Result<()> {
    use caps::{CapSet, Capability};

    caps::raise(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE)?;

    let mut keys = AttributeSet::<Key>::new();
    keys.insert(Key::KEY_BACKSPACE);

    let mut device = VirtualDeviceBuilder::new()?
        .name("CleverNote Backspace")
        .with_keys(&keys)?
        .build()?;

    caps::drop(None, CapSet::Effective, Capability::CAP_DAC_OVERRIDE).ok();
    thread::sleep(Duration::from_millis(50));

    for _ in 0..count {
        // Press backspace
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            Key::KEY_BACKSPACE.code(),
            1,
        )])?;
        thread::sleep(Duration::from_millis(10));

        // Release backspace
        device.emit(&[evdev::InputEvent::new(
            EventType::KEY,
            Key::KEY_BACKSPACE.code(),
            0,
        )])?;

        device.emit(&[evdev::InputEvent::new(EventType::SYNCHRONIZATION, 0, 0)])?;

        thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn inject_text(_text: &str) -> Result<()> {
    Err(eyre::eyre!("evdev injection only available on Linux"))
}

#[cfg(not(target_os = "linux"))]
pub fn inject_paste() -> Result<()> {
    Err(eyre::eyre!("evdev injection only available on Linux"))
}

#[cfg(not(target_os = "linux"))]
pub fn inject_char(_c: char) -> Result<()> {
    Err(eyre::eyre!("evdev injection only available on Linux"))
}

#[cfg(not(target_os = "linux"))]
pub fn inject_backspace(_count: usize) -> Result<()> {
    Err(eyre::eyre!("evdev injection only available on Linux"))
}
