use eyre::Result;

#[cfg(target_os = "macos")]
use cocoa::base::{id, nil};
#[cfg(target_os = "macos")]
use cocoa::foundation::{NSAutoreleasePool, NSPoint, NSRect, NSSize, NSString};
#[cfg(target_os = "macos")]
use objc::runtime::{Class, Object};
#[cfg(target_os = "macos")]
use objc::{msg_send, sel, sel_impl};
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex};

#[cfg(target_os = "macos")]
pub struct ProgressWindow {
    window: Arc<Mutex<id>>,
    text_field: Arc<Mutex<id>>,
    progress_bar: Arc<Mutex<id>>,
}

#[cfg(target_os = "macos")]
impl ProgressWindow {
    pub fn new(title: &str, total_files: usize) -> Result<Self> {
        unsafe {
            let _pool = NSAutoreleasePool::new(nil);

            // Create window
            let window_class =
                Class::get("NSPanel").ok_or_else(|| eyre::eyre!("NSPanel class not found"))?;
            let window: id = msg_send![window_class, alloc];

            let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(500.0, 120.0));

            let style_mask = 1 << 0 | 1 << 1 | 1 << 3; // Titled | Closable | Utility window
            let window: id = msg_send![window, initWithContentRect:frame
                                               styleMask:style_mask
                                               backing:2 // NSBackingStoreBuffered
                                               defer:0];

            // Set window properties
            let ns_title = NSString::alloc(nil).init_str(title);
            let _: () = msg_send![window, setTitle: ns_title];
            let _: () = msg_send![window, center];
            let _: () = msg_send![window, setLevel: 3]; // NSFloatingWindowLevel
            let _: () = msg_send![window, setReleasedWhenClosed: 0];

            // Create progress bar
            let progress_class = Class::get("NSProgressIndicator")
                .ok_or_else(|| eyre::eyre!("NSProgressIndicator not found"))?;
            let progress_bar: id = msg_send![progress_class, alloc];
            let progress_frame = NSRect::new(NSPoint::new(20.0, 20.0), NSSize::new(460.0, 20.0));
            let progress_bar: id = msg_send![progress_bar, initWithFrame: progress_frame];
            let _: () = msg_send![progress_bar, setIndeterminate: 0];
            let _: () = msg_send![progress_bar, setMinValue: 0.0];
            let _: () = msg_send![progress_bar, setMaxValue: total_files as f64];
            let _: () = msg_send![progress_bar, setDoubleValue: 0.0];

            // Create text field
            let textfield_class =
                Class::get("NSTextField").ok_or_else(|| eyre::eyre!("NSTextField not found"))?;
            let text_field: id = msg_send![textfield_class, alloc];
            let text_frame = NSRect::new(NSPoint::new(20.0, 50.0), NSSize::new(460.0, 50.0));
            let text_field: id = msg_send![text_field, initWithFrame: text_frame];
            let _: () = msg_send![text_field, setBordered: 0];
            let _: () = msg_send![text_field, setDrawsBackground: 0];
            let _: () = msg_send![text_field, setEditable: 0];
            let _: () = msg_send![text_field, setSelectable: 0];

            let initial_text = NSString::alloc(nil).init_str("Initializing download...");
            let _: () = msg_send![text_field, setStringValue: initial_text];

            // Add views to window
            let content_view: id = msg_send![window, contentView];
            let _: () = msg_send![content_view, addSubview: progress_bar];
            let _: () = msg_send![content_view, addSubview: text_field];

            // Show window
            let _: () = msg_send![window, makeKeyAndOrderFront: nil];

            Ok(Self {
                window: Arc::new(Mutex::new(window)),
                text_field: Arc::new(Mutex::new(text_field)),
                progress_bar: Arc::new(Mutex::new(progress_bar)),
            })
        }
    }

    pub fn update(
        &self,
        file_index: usize,
        total_files: usize,
        file_name: &str,
        progress_text: String,
    ) {
        unsafe {
            let message = format!(
                "File {}/{}:  {}\n{}",
                file_index + 1,
                total_files,
                file_name,
                progress_text
            );

            if let (Ok(text_field), Ok(progress_bar)) =
                (self.text_field.lock(), self.progress_bar.lock())
            {
                let ns_message = NSString::alloc(nil).init_str(&message);
                let _: () = msg_send![*text_field, setStringValue: ns_message];
                let _: () = msg_send![*progress_bar, setDoubleValue: (file_index + 1) as f64];

                // Force update
                if let Ok(window) = self.window.lock() {
                    let _: () = msg_send![*window, display];
                }
            }
        }
    }

    pub fn close(self) {
        unsafe {
            if let Ok(window) = self.window.lock() {
                let _: () = msg_send![*window, close];
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl Drop for ProgressWindow {
    fn drop(&mut self) {
        unsafe {
            if let Ok(window) = self.window.lock() {
                let _: () = msg_send![*window, close];
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct ProgressWindow;

#[cfg(not(target_os = "macos"))]
impl ProgressWindow {
    pub fn new(_title: &str, _total_files: usize) -> Result<Self> {
        Ok(Self)
    }

    pub fn update(
        &self,
        _file_index: usize,
        _total_files: usize,
        _file_name: &str,
        _progress_text: String,
    ) {
    }

    pub fn close(self) {}
}
