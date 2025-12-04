#!/bin/bash

set -e

echo "📦 Creating CleverNote.app Bundle"
echo "================================="
echo

# Define paths
APP_NAME="CleverNote"
APP_BUNDLE="$APP_NAME.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
BINARY_SOURCE="./target/release/parakeet"

# Check if binary exists
if [ ! -f "$BINARY_SOURCE" ]; then
    echo "❌ Binary not found at $BINARY_SOURCE"
    echo "   Run: cargo build --release"
    exit 1
fi

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

# Don't copy models - they will be downloaded to ~/.config/clevernote/
echo "ℹ️  Models will be downloaded to ~/.config/clevernote/ on first run"

# Don't copy config.toml - it will be created in ~/.config/clevernote/
echo "ℹ️  Config will be created in ~/.config/clevernote/ on first run"

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
    <string>CleverNote</string>
    <key>CFBundleIdentifier</key>
    <string>com.clevernote.parakeet</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleName</key>
    <string>CleverNote</string>
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
    <string>CleverNote needs microphone access to transcribe your voice.</string>
</dict>
</plist>
EOF

   echo "🎨 Copying app icon..."
   if [ -f "icon.icns" ]; then
       cp "icon.icns" "$RESOURCES/AppIcon.icns"
       echo "✅ Custom icon copied"
   else
       echo "⚠️  No icon.icns found, using placeholder"
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
echo "  • Double-click CleverNote.app to launch"
echo "  • Or drag to /Applications folder"
echo "  • No terminal window will appear!"
echo
echo "⚠️  Important:"
echo "  • The app still needs Accessibility permissions"
echo "  • Grant permissions to CleverNote.app (not the binary)"
echo "  • Path for permissions: $(pwd)/$APP_BUNDLE"
echo
echo "To install as LaunchAgent:"
echo "  • Run CleverNote.app once"
echo "  • It will offer to install automatically"
echo
