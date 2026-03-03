/// Recording overlay using a Wayland layer-shell surface.
///
/// Uses `wlr-layer-shell` via smithay-client-toolkit to create a surface that:
/// - Anchors to the top-right corner with exact margins
/// - Never steals keyboard focus (`KeyboardInteractivity::None`)
/// - Has a fully transparent background
/// - Shows a pulsing red dot while recording, then a green checkmark after
///   transcription completes (auto-closes after 1.5 s)
///
/// Runs on its own thread — no main-thread requirement.
#[cfg(target_os = "linux")]
mod imp {
    use smithay_client_toolkit::{
        compositor::{CompositorHandler, CompositorState},
        delegate_compositor, delegate_layer, delegate_output, delegate_registry, delegate_shm,
        output::{OutputHandler, OutputState},
        registry::{ProvidesRegistryState, RegistryState},
        registry_handlers,
        shell::{
            wlr_layer::{
                Anchor, KeyboardInteractivity, Layer, LayerShell, LayerShellHandler, LayerSurface,
                LayerSurfaceConfigure,
            },
            WaylandSurface,
        },
        shm::{slot::SlotPool, Shm, ShmHandler},
    };
    use wayland_client::{
        globals::registry_queue_init,
        protocol::{wl_output, wl_shm, wl_surface},
        Connection, QueueHandle,
    };

    use std::sync::{mpsc, Arc, Mutex, OnceLock};
    use std::time::Instant;

    // ── Commands ──────────────────────────────────────────────────────────────

    #[derive(Debug)]
    pub enum Cmd {
        TranscriptionDone,
        Close,
    }

    // ── Phase ─────────────────────────────────────────────────────────────────

    #[derive(Clone, Copy, PartialEq)]
    enum Phase {
        Recording,
        Done,
    }

    // ── Wayland app state ─────────────────────────────────────────────────────

    struct OverlayApp {
        registry_state: RegistryState,
        output_state: OutputState,
        shm: Shm,
        layer: LayerSurface,
        pool: SlotPool,

        width: u32,
        height: u32,
        configured: bool,

        phase: Phase,
        start_time: Instant,
        done_at: Option<Instant>,
        cmd_rx: Arc<Mutex<mpsc::Receiver<Cmd>>>,
        exit: bool,
    }

    const SIZE: u32 = 48;
    const MARGIN: i32 = 16;

    impl OverlayApp {
        fn draw(&mut self, qh: &QueueHandle<Self>) {
            let w = self.width;
            let h = self.height;
            let stride = w as i32 * 4;

            let (buffer, canvas) =
                match self
                    .pool
                    .create_buffer(w as i32, h as i32, stride, wl_shm::Format::Argb8888)
                {
                    Ok(v) => v,
                    Err(e) => {
                        log::error!("overlay: create_buffer: {e}");
                        return;
                    }
                };

            // Render with tiny-skia into the shared-memory buffer
            let Some(mut pixmap) = tiny_skia::Pixmap::from_vec(
                canvas.to_vec(),
                tiny_skia::IntSize::from_wh(w, h).unwrap(),
            ) else {
                return;
            };
            // Fallback: just use a local pixmap and copy
            pixmap.fill(tiny_skia::Color::TRANSPARENT);

            let cx = w as f32 / 2.0;
            let cy = h as f32 / 2.0;
            let max_r = (w.min(h) as f32 / 2.0) - 8.0;

            let t = self.start_time.elapsed().as_secs_f32();
            let pulse = (t * std::f32::consts::PI).sin().abs();

            match self.phase {
                Phase::Recording => {
                    // Outer glow
                    let glow_r = max_r + pulse * 8.0;
                    let mut paint = tiny_skia::Paint::default();
                    paint.anti_alias = true;
                    paint.set_color(
                        tiny_skia::Color::from_rgba(0.9, 0.1, 0.1, 0.35 * pulse).unwrap(),
                    );
                    if let Some(path) = tiny_skia::PathBuilder::from_circle(cx, cy, glow_r) {
                        pixmap.fill_path(
                            &path,
                            &paint,
                            tiny_skia::FillRule::Winding,
                            tiny_skia::Transform::identity(),
                            None,
                        );
                    }

                    // Solid dot
                    let dot_r = max_r * (0.80 + 0.20 * pulse);
                    paint.set_color(tiny_skia::Color::from_rgba(0.9, 0.1, 0.1, 1.0).unwrap());
                    if let Some(path) = tiny_skia::PathBuilder::from_circle(cx, cy, dot_r) {
                        pixmap.fill_path(
                            &path,
                            &paint,
                            tiny_skia::FillRule::Winding,
                            tiny_skia::Transform::identity(),
                            None,
                        );
                    }
                }
                Phase::Done => {
                    // Green circle
                    let mut paint = tiny_skia::Paint::default();
                    paint.anti_alias = true;
                    paint.set_color(tiny_skia::Color::from_rgba(0.15, 0.75, 0.3, 1.0).unwrap());
                    if let Some(path) = tiny_skia::PathBuilder::from_circle(cx, cy, max_r) {
                        pixmap.fill_path(
                            &path,
                            &paint,
                            tiny_skia::FillRule::Winding,
                            tiny_skia::Transform::identity(),
                            None,
                        );
                    }

                    // Checkmark stroke
                    let s = max_r * 0.55;
                    let mut pb = tiny_skia::PathBuilder::new();
                    pb.move_to(cx - s * 0.6, cy);
                    pb.line_to(cx - s * 0.1, cy + s * 0.5);
                    pb.line_to(cx + s * 0.65, cy - s * 0.55);
                    if let Some(path) = pb.finish() {
                        let mut stroke = tiny_skia::Stroke::default();
                        stroke.width = max_r * 0.18;
                        stroke.line_cap = tiny_skia::LineCap::Round;
                        stroke.line_join = tiny_skia::LineJoin::Round;
                        paint.set_color(tiny_skia::Color::WHITE);
                        pixmap.stroke_path(
                            &path,
                            &paint,
                            &stroke,
                            tiny_skia::Transform::identity(),
                            None,
                        );
                    }
                }
            }

            // Copy pixmap pixels (pre-multiplied RGBA) → ARGB8888 canvas
            // tiny-skia uses RGBA unpremultiplied; Wayland ARGB8888 is BGRA in memory
            // We need to premultiply and swap R/B.
            let pixels = pixmap.pixels();
            for (dst, src) in canvas.chunks_exact_mut(4).zip(pixels.iter()) {
                let a = src.alpha();
                let r = src.red();
                let g = src.green();
                let b = src.blue();
                // ARGB8888 in memory (little-endian): B G R A
                dst[0] = b;
                dst[1] = g;
                dst[2] = r;
                dst[3] = a;
            }

            self.layer
                .wl_surface()
                .damage_buffer(0, 0, w as i32, h as i32);
            // Request next frame callback for animation
            self.layer
                .wl_surface()
                .frame(qh, self.layer.wl_surface().clone());
            buffer
                .attach_to(self.layer.wl_surface())
                .expect("attach buffer");
            self.layer.commit();
        }
    }

    // ── SCTK trait impls ──────────────────────────────────────────────────────

    impl CompositorHandler for OverlayApp {
        fn scale_factor_changed(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: &wl_surface::WlSurface,
            _: i32,
        ) {
        }
        fn transform_changed(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: &wl_surface::WlSurface,
            _: wl_output::Transform,
        ) {
        }
        fn surface_enter(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: &wl_surface::WlSurface,
            _: &wl_output::WlOutput,
        ) {
        }
        fn surface_leave(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: &wl_surface::WlSurface,
            _: &wl_output::WlOutput,
        ) {
        }
        fn frame(
            &mut self,
            _: &Connection,
            qh: &QueueHandle<Self>,
            _: &wl_surface::WlSurface,
            _: u32,
        ) {
            // Check commands
            let cmd = self.cmd_rx.lock().unwrap().try_recv().ok();
            match cmd {
                Some(Cmd::TranscriptionDone) => {
                    self.phase = Phase::Done;
                    self.done_at = Some(Instant::now());
                }
                Some(Cmd::Close) => {
                    self.exit = true;
                    return;
                }
                None => {}
            }

            // Auto-close 1.5 s after Done
            if self.phase == Phase::Done {
                if self
                    .done_at
                    .map_or(false, |t| t.elapsed().as_millis() >= 1500)
                {
                    self.exit = true;
                    return;
                }
            }

            self.draw(qh);
        }
    }

    impl OutputHandler for OverlayApp {
        fn output_state(&mut self) -> &mut OutputState {
            &mut self.output_state
        }
        fn new_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {}
        fn update_output(&mut self, _: &Connection, _: &QueueHandle<Self>, _: wl_output::WlOutput) {
        }
        fn output_destroyed(
            &mut self,
            _: &Connection,
            _: &QueueHandle<Self>,
            _: wl_output::WlOutput,
        ) {
        }
    }

    impl LayerShellHandler for OverlayApp {
        fn closed(&mut self, _: &Connection, _: &QueueHandle<Self>, _: &LayerSurface) {
            self.exit = true;
        }
        fn configure(
            &mut self,
            _: &Connection,
            qh: &QueueHandle<Self>,
            _: &LayerSurface,
            configure: LayerSurfaceConfigure,
            _: u32,
        ) {
            // Use compositor-suggested size, or our requested size
            let (cw, ch) = configure.new_size;
            self.width = if cw == 0 { SIZE } else { cw };
            self.height = if ch == 0 { SIZE } else { ch };

            if !self.configured {
                self.configured = true;
                self.draw(qh);
            }
        }
    }

    impl ShmHandler for OverlayApp {
        fn shm_state(&mut self) -> &mut Shm {
            &mut self.shm
        }
    }

    delegate_compositor!(OverlayApp);
    delegate_output!(OverlayApp);
    delegate_shm!(OverlayApp);
    delegate_layer!(OverlayApp);
    delegate_registry!(OverlayApp);

    impl ProvidesRegistryState for OverlayApp {
        fn registry(&mut self) -> &mut RegistryState {
            &mut self.registry_state
        }
        registry_handlers![OutputState];
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub struct WaylandHandle {
        tx: mpsc::Sender<Cmd>,
    }

    impl WaylandHandle {
        pub fn signal_done(&self) {
            let _ = self.tx.send(Cmd::TranscriptionDone);
        }
    }

    impl Drop for WaylandHandle {
        fn drop(&mut self) {
            let _ = self.tx.send(Cmd::Close);
        }
    }

    pub fn launch() -> WaylandHandle {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let rx = Arc::new(Mutex::new(rx));

        std::thread::spawn(move || {
            if let Err(e) = run(rx) {
                log::error!("overlay thread: {e}");
            }
        });

        WaylandHandle { tx }
    }

    fn run(cmd_rx: Arc<Mutex<mpsc::Receiver<Cmd>>>) -> Result<(), Box<dyn std::error::Error>> {
        let conn = Connection::connect_to_env()?;
        let (globals, mut event_queue) = registry_queue_init(&conn)?;
        let qh = event_queue.handle();

        let compositor = CompositorState::bind(&globals, &qh)?;
        let layer_shell = LayerShell::bind(&globals, &qh)?;
        let shm = Shm::bind(&globals, &qh)?;

        let surface = compositor.create_surface(&qh);
        let layer = layer_shell.create_layer_surface(
            &qh,
            surface,
            Layer::Overlay,
            Some("clevernote-overlay"),
            None,
        );

        layer.set_anchor(Anchor::TOP | Anchor::RIGHT);
        layer.set_size(SIZE, SIZE);
        layer.set_margin(MARGIN, MARGIN, 0, 0);
        layer.set_keyboard_interactivity(KeyboardInteractivity::None);
        layer.commit();

        let pool = SlotPool::new((SIZE * SIZE * 4) as usize, &shm)?;

        let mut app = OverlayApp {
            registry_state: RegistryState::new(&globals),
            output_state: OutputState::new(&globals, &qh),
            shm,
            layer,
            pool,
            width: SIZE,
            height: SIZE,
            configured: false,
            phase: Phase::Recording,
            start_time: Instant::now(),
            done_at: None,
            cmd_rx,
            exit: false,
        };

        loop {
            event_queue.blocking_dispatch(&mut app)?;
            if app.exit {
                break;
            }
        }

        Ok(())
    }
}

// ── Public façade ─────────────────────────────────────────────────────────────

use std::sync::{atomic::AtomicBool, Arc};

/// Returns true when running under a Wayland compositor.
fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

// On Wayland we keep a handle to the live overlay so we can signal it.
#[cfg(target_os = "linux")]
use std::sync::{Mutex, OnceLock};

#[cfg(target_os = "linux")]
static HANDLE: OnceLock<Mutex<Option<imp::WaylandHandle>>> = OnceLock::new();

#[cfg(target_os = "linux")]
fn handle_store() -> &'static Mutex<Option<imp::WaylandHandle>> {
    HANDLE.get_or_init(|| Mutex::new(None))
}

pub fn show_recording_overlay() -> Arc<AtomicBool> {
    #[cfg(target_os = "linux")]
    if is_wayland() {
        let handle = imp::launch();
        *handle_store().lock().unwrap() = Some(handle);
    } else {
        // X11 / unknown: notify-send "recording started" indicator
        let _ = std::process::Command::new("notify-send")
            .args([
                "--urgency=low",
                "--expire-time=0",
                "clevernote",
                "Recording…",
            ])
            .spawn();
    }
    Arc::new(AtomicBool::new(false))
}

pub fn show_transcription_complete() {
    #[cfg(target_os = "linux")]
    if is_wayland() {
        if let Some(handle) = handle_store().lock().unwrap().as_ref() {
            handle.signal_done();
        }
    } else {
        // X11 / unknown: replace any lingering notification with a done message
        let _ = std::process::Command::new("notify-send")
            .args([
                "--urgency=low",
                "--expire-time=2000",
                "clevernote",
                "Transcription complete",
            ])
            .spawn();
    }
}

pub fn hide_recording_overlay(_should_close: Arc<AtomicBool>) {}
