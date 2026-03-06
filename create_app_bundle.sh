#!/bin/bash

set -e

echo "📦 Creating Vox Application"
echo "=================================="
echo

# Detect platform
OS="$(uname -s)"
APP_NAME="Vox"
BINARY_SOURCE="./target/release/parakeet"

# Check for required system dependencies on Linux
if [ "$OS" = "Linux" ]; then
    echo "🔍 Checking system dependencies..."
    
    if ! command -v pkg-config >/dev/null 2>&1; then
        echo "⚠️  pkg-config not found. Install it using your package manager."
    else
        MISSING_DEPS=""
        for pkg in gdk-3.0 appindicator-0.1 libsecret-1 xdo; do
            if ! pkg-config --exists "$pkg" 2>/dev/null; then
                case "$pkg" in
                    gdk-3.0) MISSING_DEPS="$MISSING_DEPS libgtk-3-dev" ;;
                    appindicator-0.1) MISSING_DEPS="$MISSING_DEPS libappindicator3-dev" ;;
                    libsecret-1) MISSING_DEPS="$MISSING_DEPS libsecret-1-dev" ;;
                    xdo) MISSING_DEPS="$MISSING_DEPS libxdo-dev" ;;
                esac
            fi
        done
        
        if [ -n "$MISSING_DEPS" ]; then
            echo "⚠️  Missing system dependencies:$MISSING_DEPS"
            echo "   Build may fail without these. Install using your package manager:"
            echo "  # Debian/Ubuntu:"
            echo "    sudo apt-get install$MISSING_DEPS"
            echo
            echo "  # Fedora:"
            echo "    sudo dnf install gtk3-devel libappindicator-gtk3-devel libsecret-devel libxdo-devel"
            echo
            echo "  # Arch:"
            echo "    sudo pacman -S gtk3 libappindicator-gtk3 libsecret xdotool"
            echo
        else
            echo "✅ System dependencies found"
        fi
    fi
fi

# Check if binary exists, build if not
if [ ! -f "$BINARY_SOURCE" ]; then
    echo "🔨 Binary not found, building..."
    cargo build --release
    echo "✅ Build complete"
fi

if [ "$OS" = "Darwin" ]; then
    # macOS
    APP_BUNDLE="$APP_NAME.app"
    CONTENTS="$APP_BUNDLE/Contents"
    MACOS="$CONTENTS/MacOS"
    RESOURCES="$CONTENTS/Resources"

    # Remove old bundle if exists
    if [ -d "$APP_BUNDLE" ]; then
        echo "🗑️  Removing old app bundle..."
        rm -rf "$APP_BUNDLE"
    fi

    # Create bundle structure
    echo "📁 Creating bundle structure..."
    mkdir -p "$MACOS"
    mkdir -p "$RESOURCES"

    # Copy binary
    echo "📋 Copying binary..."
    cp "$BINARY_SOURCE" "$MACOS/$APP_NAME"
    chmod +x "$MACOS/$APP_NAME"

    # Don't copy models - they will be downloaded to ~/.vox/
    echo "ℹ️  Models will be downloaded to ~/.vox/ on first run"

    # Don't copy config.toml - it will be created in ~/.vox/
    echo "ℹ️  Config will be created in ~/.vox/ on first run"

    # Create Info.plist
    echo "📝 Creating Info.plist..."
    cat > "$CONTENTS/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>Vox</string>
    <key>CFBundleIdentifier</key>
    <string>com.vox.parakeet</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleName</key>
    <string>Vox</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Vox needs microphone access to transcribe your voice.</string>
</dict>
</plist>
EOF

    echo "🎨 Copying app icon..."
    if [ -d "icons" ]; then
        mkdir -p "$RESOURCES"
        if [ -f "icons/AppIcon.icns" ]; then
            cp "icons/AppIcon.icns" "$RESOURCES/AppIcon.icns"
        elif command -v iconutil >/dev/null 2>&1; then
            ICONSET_DIR="$RESOURCES/AppIcon.iconset"
            mkdir -p "$ICONSET_DIR"
            for size in 16 32 128 256 512; do
                cp "icons/icon-${size}.png" "$ICONSET_DIR/icon_${size}x${size}.png"
                cp "icons/icon-$((size*2)).png" "$ICONSET_DIR/icon_${size}x${size}@2x.png" 2>/dev/null || true
            done
            iconutil -c icns "$ICONSET_DIR" -o "$RESOURCES/AppIcon.icns"
            rm -rf "$ICONSET_DIR"
        else
            cp "icons/icon-256.png" "$RESOURCES/AppIcon.icns"
        fi
        echo "✅ Icon copied"
    elif [ -f "icon.icns" ]; then
        cp "icon.icns" "$RESOURCES/AppIcon.icns"
        echo "✅ Custom icon copied"
    else
        echo "⚠️  No icon found, using placeholder"
    fi

    # Code sign the bundle
    echo "✍️  Code signing app bundle..."
    codesign --force --deep --sign - "$APP_BUNDLE"

    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ App bundle created successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "📍 Location: $(pwd)/$APP_BUNDLE"
    echo
    echo "To use:"
    echo "  • Double-click Vox.app to launch"
    echo "  • Or drag to /Applications folder"
    echo "  • No terminal window will appear!"
    echo
    echo "⚠️  Important:"
    echo "  • The app still needs Accessibility permissions"
    echo "  • Grant permissions to Vox.app (not the binary)"
    echo "  • Path for permissions: $(pwd)/$APP_BUNDLE"
    echo
    echo "To install as LaunchAgent:"
    echo "  • Run Vox.app once"
    echo "  • It will offer to install automatically"
    echo

elif [ "$OS" = "Linux" ]; then
    # Linux
    INSTALL_DIR="/usr/local/bin"
    DESKTOP_FILE="vox.desktop"
    BINARY_NAME="vox"

    echo "📁 Setting up directories..."
    mkdir -p ~/.local/bin
    mkdir -p ~/.local/applications
    mkdir -p ~/.local/share/icons/hicolor/256x256/apps
    mkdir -p ~/.config/vox

    echo "📋 Installing binary..."
    cp "$BINARY_SOURCE" ~/.local/bin/$BINARY_NAME
    chmod +x ~/.local/bin/$BINARY_NAME

    echo "🎨 Installing icon..."
    ICON_PATH=""
    if [ -d "icons" ]; then
        for size in 16 22 24 32 48 64 128 256 512; do
            DIR="$HOME/.local/share/icons/hicolor/${size}x${size}/apps"
            mkdir -p "$DIR"
            cp "icons/icon-${size}.png" "$DIR/$BINARY_NAME.png" 2>/dev/null || true
        done
        ICON_PATH="$HOME/.local/share/icons/hicolor/256x256/apps/$BINARY_NAME.png"
        echo "✅ Custom icons installed"
    elif [ -f "icon.png" ]; then
        mkdir -p ~/.local/share/icons/hicolor/256x256/apps
        cp "icon.png" ~/.local/share/icons/hicolor/256x256/apps/$BINARY_NAME.png
        ICON_PATH="$HOME/.local/share/icons/hicolor/256x256/apps/$BINARY_NAME.png"
        echo "✅ Custom icon installed"
    else
        echo "⚠️  No icon.png found, using default"
    fi

    echo "📝 Creating desktop entry..."
    cat > ~/.local/share/$DESKTOP_FILE << EOF
[Desktop Entry]
Name=Vox
Comment=Voice-to-text transcription with LLM processing
Exec=$HOME/.local/bin/vox
Icon=$ICON_PATH
Type=Application
Categories=Office;Utility;
Terminal=false
StartupNotify=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=0
EOF

    echo "🔄 Updating desktop database..."
    update-desktop-database ~/.local/share/applications 2>/dev/null || true

    echo
    while true; do
        read -p "🔧 Install globally (requires sudo)? (y/n) " yn
        case $yn in
            [Yy]* )
                echo "📋 Installing to $INSTALL_DIR..."
                sudo cp "$BINARY_SOURCE" $INSTALL_DIR/$BINARY_NAME
                sudo chmod +x $INSTALL_DIR/$BINARY_NAME
                
                if [ -n "$ICON_PATH" ]; then
                    if [ -d "icons" ]; then
                        for size in 16 22 24 32 48 64 128 256 512; do
                            sudo mkdir -p /usr/share/icons/hicolor/${size}x${size}/apps
                            sudo cp "icons/icon-${size}.png" /usr/share/icons/hicolor/${size}x${size}/apps/$BINARY_NAME.png 2>/dev/null || true
                        done
                    else
                        sudo mkdir -p /usr/share/icons/hicolor/256x256/apps
                        sudo cp "icon.png" /usr/share/icons/hicolor/256x256/apps/$BINARY_NAME.png
                    fi
                    GLOBAL_ICON_PATH="/usr/share/icons/hicolor/256x256/apps/$BINARY_NAME.png"
                fi
                
                sudo sed -i "s|Icon=.*|Icon=$GLOBAL_ICON_PATH|" ~/.local/share/applications/$DESKTOP_FILE
                sudo cp ~/.local/share/applications/$DESKTOP_FILE /usr/share/applications/
                echo "✅ Global installation complete"
                break
                ;;
            [Nn]* )
                echo "ℹ️  Skipped global installation"
                break
                ;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    echo
    while true; do
        read -p "⚡ Start Vox now? (y/n) " yn
        case $yn in
            [Yy]* )
                ~/.local/bin/$BINARY_NAME &
                echo "✅ Vox started in background"
                break
                ;;
            [Nn]* )
                echo "ℹ️  Vox not started"
                break
                ;;
            * ) echo "Please answer yes or no.";;
        esac
    done

    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Linux application setup complete!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    echo "📍 Binary: ~/.local/bin/$BINARY_NAME"
    echo "📍 Desktop: ~/.local/share/applications/$DESKTOP_FILE"
    echo
    echo "To use:"
    echo "  • Run 'vox' from terminal"
    echo "  • Find it in applications menu"
    echo "  • Or run: ~/.local/bin/$BINARY_NAME"
    echo
    echo "⚠️  Important:"
    echo "  • Grant Accessibility permissions if prompted"
    echo "  • Models will download to ~/.vox/ on first run"
    echo

else
    echo "❌ Unsupported operating system: $OS"
    exit 1
fi
