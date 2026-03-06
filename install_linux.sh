#!/bin/bash
# Vox Linux Installation Script

set -e

echo "🎙️  Installing Vox for Linux..."

# Build binaries
echo "📦 Building binaries..."
cargo build --release --features daemon --bin vox-daemon --bin vox

# Create directories
mkdir -p ~/.local/bin
mkdir -p ~/.config/systemd/user

# Install binaries
echo "📂 Installing binaries to ~/.local/bin/..."
cp target/release/vox-daemon ~/.local/bin/
cp target/release/vox ~/.local/bin/
chmod +x ~/.local/bin/vox-daemon
chmod +x ~/.local/bin/vox

# Set up CAP_DAC_OVERRIDE capability (required for auto-paste)
echo ""
echo "🔐 Setting up auto-paste capability..."
echo "   Vox requires CAP_DAC_OVERRIDE to access /dev/uinput for keyboard simulation"
echo "   This is safer than running as root - capability is used only when creating virtual keyboard"
echo ""
read -p "Set up auto-paste capability? (requires sudo, recommended) (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Setting up CAP_DAC_OVERRIDE capability..."
    if command -v setcap > /dev/null 2>&1; then
        sudo setcap cap_dac_override+ep ~/.local/bin/vox-daemon && {
            echo "✅ Capability set successfully"
            echo "   • Auto-paste will work (Ctrl+V for apps, Ctrl+Shift+V for terminals)"
            echo "   • Verify with: getcap ~/.local/bin/vox-daemon"
        } || {
            echo "⚠️  Failed to set capability"
            echo "   Auto-paste will NOT work. Text will still be copied to clipboard."
        }
    else
        echo "⚠️  setcap command not found. Install libcap package:"
        echo "   sudo apt install libcap2-bin  # Debian/Ubuntu"
        echo "   sudo pacman -S libcap         # Arch"
        echo "   sudo dnf install libcap       # Fedora"
    fi
else
    echo "⚠️  Skipping capability setup"
    echo "   Auto-paste will NOT work. Text will be copied to clipboard only."
    echo "   You can set it up later with:"
    echo "   sudo setcap cap_dac_override+ep ~/.local/bin/vox-daemon"
fi

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "⚠️  Warning: ~/.local/bin is not in your PATH"
    echo "   Add this to your ~/.bashrc or ~/.zshrc:"
    echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

# Install systemd service (optional)
read -p "Install systemd service to start on login? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📋 Installing systemd service..."
    cp install/systemd/vox-daemon.service ~/.config/systemd/user/
    systemctl --user enable vox-daemon.service
    systemctl --user start vox-daemon.service
    echo "✅ Systemd service installed and started"
    echo "   Control with: systemctl --user {start|stop|restart|status} vox-daemon"
fi

# Detect compositor
echo ""
echo "🖥️  Compositor-specific configuration:"
if [ -n "$HYPRLAND_INSTANCE_SIGNATURE" ] || pgrep -x Hyprland > /dev/null; then
    echo "Detected: Hyprland"
    echo ""
    cat install/hyprland/vox.conf
elif [ -n "$SWAYSOCK" ] || pgrep -x sway > /dev/null; then
    echo "Detected: Sway"
    echo "Add this to your ~/.config/sway/config:"
    echo ""
    cat install/sway/vox.conf
else
    echo "Unknown compositor. See install/hyprland/ or install/sway/ for examples."
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Reload your shell or run: source ~/.bashrc"
echo "   2. Add compositor config (see above)"
echo "   3. Start daemon: vox-daemon"
echo "   4. Test: vox toggle"
echo ""
echo "🎯 Usage:"
echo "   • vox toggle  - Start/stop recording"
echo "   • vox start   - Start recording (push-to-talk)"
echo "   • vox stop    - Stop recording (push-to-talk)"
echo "   • vox status  - Check daemon status"
echo "   • vox quit    - Stop daemon"
echo ""
echo "⚙️  Configuration:"
echo "   • Config file: ~/.config/vox/config.toml"
echo "   • Customize hotkey: modifier_key = \"Alt\" / trigger_key = \"Space\""
echo ""
echo "📜 History:"
echo "   • Transcripts stored in: ~/.config/vox/transcripts/history.json"
echo "   • Last 100 transcripts kept automatically (no audio files saved)"
echo ""
echo "🔍 Troubleshooting:"
echo "   • Verify capability: getcap ~/.local/bin/vox-daemon"
echo "   • Check logs: journalctl --user -u vox-daemon -f"
echo "   • If paste fails, text is still in clipboard (manual Ctrl+V works)"
