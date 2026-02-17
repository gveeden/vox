# CleverNote for Linux/Wayland

Voice transcription daemon for Linux with Wayland/Hyprland support.

## Features

- ✅ **Daemon + Client Architecture** - Model stays hot for instant transcription
- ✅ **Native Wayland Support** - Tested on Hyprland, Sway, and wlroots compositors
- ✅ **Compositor Hotkey Binding** - Native integration with your window manager
- ✅ **Smart Auto-paste** - Detects terminals vs apps (Ctrl+Shift+V vs Ctrl+V)
- ✅ **Low Latency** - Pre-loaded model for instant transcription (<1s typically)
- ✅ **Clean Notifications** - Single notification via notify-send (no duplicates)
- ✅ **History Tracking** - Last 100 transcripts in `history.json` (no audio files)
- 🎯 **Multilingual** - Supports 25+ languages via Parakeet TDT model

## Architecture

```
┌──────────────────┐         Unix Socket          ┌─────────────────┐
│  Compositor      │      ~/.config/clevernote/          │   clevernote    │
│  (Hyprland/Sway) │◄───────daemon.sock───────────┤   (client)      │
│                  │                               └─────────────────┘
│  Hotkey: Alt+Space                                        │
│      │                                                    │
│      ▼                                                    ▼
│  exec clevernote toggle                         Sends: Toggle command
└──────────────────┘                                       │
                                                           ▼
         ┌─────────────────────────────────────────────────────────┐
         │           clevernote-daemon                              │
         ├─────────────────────────────────────────────────────────┤
         │  • Parakeet TDT model (loaded once, stays hot)          │
         │  • Audio capture (cpal)                                  │
         │  • Transcription worker thread                           │
         │  • Clipboard + paste (enigo)                             │
         └─────────────────────────────────────────────────────────┘
```

## Installation

### Quick Install

```bash
./install_linux.sh
```

This will:
1. Build the release binaries
2. Install to `~/.local/bin/`
3. Optionally set up systemd service
4. Show compositor-specific config

### Manual Install

```bash
# Build
cargo build --release --features daemon --bin clevernote-daemon --bin clevernote

# Install
cp target/release/clevernote-daemon ~/.local/bin/
cp target/release/clevernote ~/.local/bin/

# Add to PATH (if needed)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## Configuration

### Hyprland

Add to `~/.config/hypr/hyprland.conf`:

```conf
# Start daemon on login
exec-once = clevernote-daemon

# Bind Alt+Space to toggle recording
bind = ALT, SPACE, exec, clevernote toggle
```

Then reload Hyprland config or restart.

### Sway

Add to `~/.config/sway/config`:

```conf
# Start daemon on login
exec clevernote-daemon

# Bind Alt+Space to toggle recording
bindsym Mod1+Space exec clevernote toggle
```

Then reload: `swaymsg reload`

### Other Compositors

The daemon works with any compositor that can:
- Run background processes (`exec clevernote-daemon`)
- Bind hotkeys to commands (`exec clevernote toggle`)

### Systemd (Auto-start on Login)

```bash
# Install service
cp install/systemd/clevernote-daemon.service ~/.config/systemd/user/

# Enable and start
systemctl --user enable clevernote-daemon.service
systemctl --user start clevernote-daemon.service

# Check status
systemctl --user status clevernote-daemon
```

## Usage

### Starting the Daemon

**Option 1: Manual**
```bash
clevernote-daemon
```

**Option 2: Systemd**
```bash
systemctl --user start clevernote-daemon
```

**Option 3: Compositor autostart** (see Configuration above)

### Recording

Once the daemon is running and hotkeys are bound:

1. Press **Alt+Space** to start recording
2. Speak your message
3. Press **Alt+Space** again to stop

The transcribed text will automatically:
- ✅ Copy to clipboard (wl-copy on Wayland)
- ✅ Paste at cursor position (smart detection: Ctrl+Shift+V for terminals, Ctrl+V for apps)
- ✅ Save to `~/.config/clevernote/transcripts/history.json` (last 100 transcripts)

### Manual Control

```bash
# Toggle recording
clevernote toggle

# Check daemon status
clevernote status

# Stop daemon
clevernote quit
```

## Requirements

- **Rust** 1.70+
- **Audio**: PulseAudio or PipeWire
- **Keyboard Injection**: Linux kernel with `/dev/uinput` support (standard on modern systems)
- **Clipboard**: `wl-clipboard` (Wayland) - must have `wl-copy` command available
- **Notifications**: `notify-send` and `dbus-send` for recording indicator
- **D-Bus**: Session bus for notification management
- **On Hyprland**: `hyprctl` for terminal detection
- **Libraries**: cpal, onnxruntime, evdev, xkbcommon, etc.

### Post-Install Setup (Required)

After installation, grant the daemon permission to create virtual keyboard devices:

```bash
sudo setcap cap_dac_override+ep ~/.local/bin/clevernote-daemon
```

**Why this is needed:**
- The daemon uses evdev to create a virtual keyboard for auto-paste
- Requires access to `/dev/uinput` (normally root-only)
- `CAP_DAC_OVERRIDE` capability allows this without running as root
- Capability is used only when opening `/dev/uinput` and dropped immediately after
- Much safer than running the entire daemon as root

**Without this capability:**
- Recording and transcription still work
- Text is copied to clipboard
- Auto-paste will fail (you'll need to manually Ctrl+V)

## Troubleshooting

### Daemon not starting

```bash
# Check if running
pgrep clevernote-daemon

# Check logs
journalctl --user -u clevernote-daemon -f

# Or run manually to see errors
clevernote-daemon
```

### Hotkey not working

1. Make sure daemon is running: `clevernote status`
2. Check compositor config syntax
3. Reload compositor config
4. Try manual test: `clevernote toggle`

### Paste not working

CleverNote uses **evdev** (`/dev/uinput`) for keyboard simulation, which works on both X11 and Wayland.

**Setup required:**
```bash
# Grant the daemon permission to access /dev/uinput
sudo setcap cap_dac_override+ep $(which clevernote-daemon)
```

**How it works:**
1. Text is copied to clipboard via `wl-copy` (Wayland) or `arboard` (X11)
2. Daemon raises `CAP_DAC_OVERRIDE` capability temporarily
3. Opens `/dev/uinput` to create a virtual keyboard device
4. Drops the capability immediately after device creation
5. Detects if active window is a terminal (via `hyprctl` on Hyprland)
6. Sends appropriate paste command:
   - Terminals: Ctrl+Shift+V
   - Other apps: Ctrl+V
7. Waits for paste to complete with proper timing delays

**Terminal Detection:**
On Hyprland, the daemon queries `hyprctl activewindow -j` to check window class against known terminals:
- ghostty, kitty, alacritty, foot, wezterm, konsole, gnome-terminal, etc.

**Security:**
- Capability is only active for ~300ms during device creation
- Daemon cannot access arbitrary files
- Virtual keyboard device is destroyed after paste

**Troubleshooting:**
- Verify capability: `getcap ~/.local/bin/clevernote-daemon`
- Should show: `cap_dac_override=ep`
- If auto-paste fails, text is still in clipboard (manual Ctrl+V works)
- Check logs: `journalctl --user -u clevernote-daemon -f`

### Audio issues

```bash
# List input devices
arecord -l

# Check PulseAudio/PipeWire
pactl list sources

# Test recording
clevernote toggle
# Speak for a few seconds
clevernote toggle
# Check ~/.config/clevernote/transcripts/
```

### Model download

On first run, the daemon will download the Parakeet TDT model (~3GB) to:
```
~/.config/clevernote/models/parakeet-tdt/
```

This happens automatically. If it fails, check internet connection.

## Differences from macOS Version

| Feature | macOS | Linux/Wayland |
|---------|-------|---------------|
| Hotkeys | Global (accessibility) | Compositor bindings |
| Recording Indicator | Native NSWindow | notify-send + D-Bus |
| Auto-start | LaunchAgent | Systemd or compositor |
| Model Loading | On demand | Daemon (stays hot) |
| Paste Method | Accessibility API | evdev + clipboard |
| Terminal Detection | N/A | hyprctl (Hyprland) |
| Performance | Fast | **Faster** (pre-loaded) |
| Dependencies | None (native) | wl-clipboard, notify-send |

## Future Enhancements

- [ ] Wayland layer-shell recording indicator (pulsing red dot)
- [ ] XDG Desktop Portal hotkey support (compositor-agnostic)
- [ ] Hyprland native plugin
- [ ] GUI configuration tool
- [ ] D-Bus interface

## Technical Details

### Why Daemon Architecture?

1. **Model stays hot** - Loading Parakeet TDT takes ~3-5 seconds. Daemon loads it once.
2. **Instant transcription** - No startup delay when recording.
3. **Lower resource usage** - One model instance vs. reloading per recording.
4. **Better Wayland integration** - Long-running process matches Wayland security model.

### IPC Protocol

Daemon and client communicate via Unix socket at `~/.config/clevernote/daemon.sock`.

Messages are JSON-encoded:

**Commands**:
- `{"Toggle"}` - Start/stop recording
- `{"Status"}` - Get daemon status
- `{"Quit"}` - Shutdown daemon

**Responses**:
- `{"Success":{"RecordingStarted"}}` 
- `{"Success":{"RecordingStopped"}}`
- `{"Success":{"Status":{...}}}`
- `{"Error":"message"}`

### Threading Model

```
Main Thread:
  └─ Unix socket server (blocking)
  └─ Audio capture (cpal stream)
  └─ IPC command handling

Background Thread:
  └─ Transcription worker
      └─ Parakeet model inference
      └─ Clipboard + paste
```

Audio stream stays in main thread (cpal::Stream is !Send).

## License

Same as main CleverNote project (MIT/Apache-2.0).

## Credits

- [parakeet-rs](https://github.com/altunenes/parakeet-rs) - Rust bindings
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) - Parakeet models
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio
- [enigo](https://github.com/enigo-rs/enigo) - Keyboard simulation
- [arboard](https://github.com/1Password/arboard) - Clipboard access
