#!/usr/bin/env python3
"""Simple overlay window helper using PyObjC with blur + rounded border"""
import sys
from AppKit import (
    NSApplication, NSWindow, NSTextField, NSColor, NSFont,
    NSMakeRect, NSBackingStoreBuffered, NSScreen, NSApp, NSVisualEffectView
)
from PyObjCTools import AppHelper


def create_overlay_window():
    """Create a borderless, blurred overlay window"""
    screen = NSScreen.mainScreen()
    screen_frame = screen.frame()

    width, height = 220, 70
    x = screen_frame.size.width - width - 20
    y = screen_frame.size.height - height - 20

    # Create a borderless transparent window
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, width, height),
        0,  # borderless
        NSBackingStoreBuffered,
        False
    )

    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setIgnoresMouseEvents_(True)
    window.setLevel_(1000)  # Always-on-top
    window.setCollectionBehavior_(1 << 6)  # All spaces

    # 🔮 Add a blurred background using NSVisualEffectView
    blur_view = NSVisualEffectView.alloc().initWithFrame_(window.contentView().bounds())
    blur_view.setAutoresizingMask_(18)  # Width + Height resizing
    blur_view.setMaterial_(0)  # NSVisualEffectMaterialAppearanceBased
    blur_view.setBlendingMode_(0)  # Behind window content
    blur_view.setState_(1)  # Active
    blur_view.setWantsLayer_(True)

    # 🌀 Add rounded corners and a red border
    layer = blur_view.layer()
    layer.setCornerRadius_(20.0)
    layer.setMasksToBounds_(True)
    layer.setBorderColor_(NSColor.colorWithRed_green_blue_alpha_(1.0, 0.23, 0.19, 0.8).CGColor())
    layer.setBorderWidth_(2.0)

    window.setContentView_(blur_view)

    # Add label — centered in the blur view
    label = NSTextField.alloc().initWithFrame_(NSMakeRect(10, (height - 35) / 2, width - 20, 30))
    label.setStringValue_("🎙️ Recording...")
    label.setFont_(NSFont.boldSystemFontOfSize_(16))
    label.setTextColor_(NSColor.whiteColor())
    label.setBackgroundColor_(NSColor.clearColor())
    label.setBezeled_(False)
    label.setEditable_(False)
    label.setAlignment_(1)  # Centered
    label.setDrawsBackground_(False)

    blur_view.addSubview_(label)
    window.orderFrontRegardless()

    return window


if __name__ == "__main__":
    app = NSApplication.sharedApplication()
    window = create_overlay_window()

    def check_stdin():
        try:
            if sys.stdin.read(1) == '':
                AppHelper.stopEventLoop()
        except:
            AppHelper.stopEventLoop()

    import threading
    threading.Thread(target=check_stdin, daemon=True).start()
    AppHelper.runEventLoop()
