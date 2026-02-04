#!/bin/bash

echo "🛑 Stopping CleverNote..."
echo

# Unload the LaunchAgent
if [ -f ~/Library/LaunchAgents/com.clevernote.parakeet.plist ]; then
    echo "Unloading LaunchAgent..."
    launchctl unload ~/Library/LaunchAgents/com.clevernote.parakeet.plist 2>/dev/null || true
    echo "✅ LaunchAgent unloaded"
fi

# Kill any running processes
PIDS=$(pgrep -f "parakeet" | grep -v grep)
if [ ! -z "$PIDS" ]; then
    echo "Killing running processes: $PIDS"
    kill -9 $PIDS 2>/dev/null || true
    echo "✅ Processes killed"
else
    echo "No running processes found"
fi

echo
echo "✅ CleverNote stopped"
echo
echo "To remove completely:"
echo "  rm ~/Library/LaunchAgents/com.clevernote.parakeet.plist"
echo "  rm -rf ~/.clevernote"
