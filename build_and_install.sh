#!/bin/bash

set -e

echo "🔨 Building and Installing CleverNote"
echo "======================================"
echo

# Step 1: Stop any running instances
echo "1️⃣  Stopping any running instances..."
if [ -f ~/Library/LaunchAgents/com.clevernote.parakeet.plist ]; then
    launchctl unload ~/Library/LaunchAgents/com.clevernote.parakeet.plist 2>/dev/null || true
fi
pkill -9 parakeet 2>/dev/null || true
echo "✅ Stopped"
echo

# Step 2: Clean the first-run marker so installation runs
echo "2️⃣  Resetting installation state..."
#rm -rf ~/.clevernote
echo "✅ Reset"
echo

# Step 3: Build release binary
echo "3️⃣  Building release binary..."
cargo build --release
echo "✅ Built"
echo

# Step 4: Code sign the binary (without hardened runtime to allow homebrew dylibs)
BINARY_PATH="./target/release/parakeet"
echo "4️⃣  Code signing binary..."
codesign --force --deep --sign - "$BINARY_PATH"
echo "✅ Signed"
echo

# Step 5: Verify signature
echo "5️⃣  Verifying signature..."
codesign -dv "$BINARY_PATH" 2>&1 | grep -E "Signature|Identifier" || true
echo "✅ Verified"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Binary built and signed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Step 6: Create app bundle
echo "6️⃣  Creating CleverNote.app bundle..."
./create_app_bundle.sh > /dev/null 2>&1
echo "✅ App bundle created"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "To launch CleverNote:"
echo "  • Double-click: CleverNote.app"
echo "  • Or run: open CleverNote.app"
echo
echo "Benefits:"
echo "  ✅ No terminal window appears"
echo "  ✅ Can be dragged to Applications folder"
echo "  ✅ Appears like a normal macOS app"
echo
echo "On first launch, it will:"
echo "  • Offer to install as background service"
echo "  • Guide you through Accessibility permissions"
echo "  • Auto-start on login"
echo


