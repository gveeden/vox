use eyre::{Result, WrapErr};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(target_os = "macos")]
use crate::progress_window::ProgressWindow;

#[cfg(target_os = "macos")]
fn show_download_dialog(message: &str) -> i32 {
    use std::process::Command;

    let script = format!(
        "display dialog \"{}\" with title \"Download Model\" buttons {{\"Download\", \"Cancel\"}} default button \"Download\"",
        message.replace("\"", "\\\"")
    );

    let output = Command::new("osascript").arg("-e").arg(&script).output();

    match output {
        Ok(result) => {
            if result.status.success() {
                let output_str = String::from_utf8_lossy(&result.stdout);
                if output_str.contains("Cancel") {
                    return 1;
                }
                return 0;
            }
            1
        }
        Err(_) => 1,
    }
}

#[cfg(target_os = "macos")]
fn show_info_dialog(title: &str, message: &str) {
    use std::process::Command;

    let script = format!(
        "display dialog \"{}\" with title \"{}\" buttons {{\"OK\"}} default button \"OK\"",
        message.replace("\"", "\\\""),
        title.replace("\"", "\\\"")
    );

    let _ = Command::new("osascript").arg("-e").arg(&script).output();
}

// ProgressWindow is now defined in progress_window.rs module

const HUGGINGFACE_BASE: &str =
    "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main";

const TDT_FILES: &[(&str, &str)] = &[
    ("encoder.onnx", "encoder-model.onnx"),
    ("encoder-model.onnx.data", "encoder-model.onnx.data"),
    ("decoder_joint.onnx", "decoder_joint-model.onnx"),
    ("vocab.txt", "vocab.txt"),
];

pub fn ensure_model_exists(model_path: &str, is_tdt: bool) -> Result<PathBuf> {
    let path = PathBuf::from(model_path);

    // If path exists and has required files, return it
    if path.exists() {
        if is_tdt && check_tdt_model_complete(&path) {
            println!("✅ TDT model found and complete");
            return Ok(path);
        } else if !is_tdt {
            // For CTC, just check if the path exists
            return Ok(path);
        }
    }

    // If TDT model is incomplete or missing, offer to download
    if is_tdt {
        println!("⚠️  TDT model not found or incomplete at: {}", model_path);

        // Use GUI dialog on macOS
        #[cfg(target_os = "macos")]
        {
            let message = format!(
                "Vox needs to download the speech recognition model.\n\n\
                 Model: parakeet-tdt-0.6b-v3 (~3GB)\n\
                 Location: {}\n\n\
                 This will take a few minutes depending on your connection.\n\n\
                 Download now?",
                model_path
            );

            let result = show_download_dialog(&message);

            if result == 0 {
                download_tdt_model(&path)?;

                let success_msg = format!(
                    "Model downloaded successfully!\n\n\
                     Location: {}\n\n\
                     Vox is ready to use!",
                    model_path
                );
                show_info_dialog("Download Complete", &success_msg);

                return Ok(path);
            } else {
                return Err(eyre::eyre!(
                    "Model download cancelled. The app cannot run without a model."
                ));
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            println!("📦 Would you like to download the TDT model from Hugging Face?");
            println!("   Model: parakeet-tdt-0.6b-v3-onnx (~3GB)");
            println!("   This will take a few minutes depending on your connection.");
            println!();
            print!("Download now? [Y/n]: ");
            std::io::stdout().flush()?;

            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let input = input.trim().to_lowercase();

            if input.is_empty() || input == "y" || input == "yes" {
                download_tdt_model(&path)?;
                println!("✅ TDT model downloaded successfully!");
                return Ok(path);
            } else {
                return Err(eyre::eyre!(
                    "Model download cancelled. Please provide a valid model path."
                ));
            }
        }
    }

    Ok(path)
}

fn check_tdt_model_complete(path: &Path) -> bool {
    for (local_name, _) in TDT_FILES {
        let file_path = path.join(local_name);
        if !file_path.exists() {
            println!("   Missing: {}", local_name);
            return false;
        }
    }
    true
}

fn download_tdt_model(dest_dir: &Path) -> Result<()> {
    // Create directory if it doesn't exist
    fs::create_dir_all(dest_dir).wrap_err("Failed to create model directory")?;

    println!("📥 Downloading TDT model files to: {}", dest_dir.display());
    println!();

    // Create GUI progress window on macOS
    #[cfg(target_os = "macos")]
    let progress_window = ProgressWindow::new("Downloading Vox Models", TDT_FILES.len())?;

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()?;

    let mut file_index = 0;

    for (local_name, remote_name) in TDT_FILES {
        let url = format!("{}/{}", HUGGINGFACE_BASE, remote_name);
        let dest_path = dest_dir.join(local_name);

        // Skip if file already exists
        if dest_path.exists() {
            println!("✓ {} (already exists)", local_name);
            #[cfg(target_os = "macos")]
            progress_window.update(
                file_index,
                TDT_FILES.len(),
                local_name,
                "Already exists".to_string(),
            );
            file_index += 1;
            continue;
        }

        println!("📦 Downloading {}...", local_name);

        // Get file size for progress bar
        let response = client
            .get(&url)
            .send()
            .wrap_err_with(|| format!("Failed to download {}", remote_name))?;

        let total_size = response.content_length().unwrap_or(0);

        // Create terminal progress bar (still useful for debugging)
        let pb = if total_size > 0 {
            let pb = ProgressBar::new(total_size);
            pb.set_style(ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"));
            pb.set_message(format!("Downloading {}", local_name));
            Some(pb)
        } else {
            None
        };

        // Download and write file
        let mut file = File::create(&dest_path)
            .wrap_err_with(|| format!("Failed to create file: {}", dest_path.display()))?;

        let mut downloaded: u64 = 0;
        let mut buffer = vec![0; 8192];
        let mut reader = response;
        let mut last_gui_update = std::time::Instant::now();

        loop {
            use std::io::Read;
            let bytes_read = reader.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;

            // Update terminal progress bar
            if let Some(pb) = &pb {
                pb.set_position(downloaded);
            }

            // Update GUI progress window (throttle to every 100ms)
            #[cfg(target_os = "macos")]
            {
                if last_gui_update.elapsed().as_millis() > 100 {
                    let mb_downloaded = downloaded as f64 / 1024.0 / 1024.0;
                    let mb_total = total_size as f64 / 1024.0 / 1024.0;
                    let progress_pct = if total_size > 0 {
                        (downloaded as f64 / total_size as f64 * 100.0) as usize
                    } else {
                        0
                    };

                    let progress_text = if total_size > 0 {
                        format!(
                            "{:.1}/{:.1} MB ({}%)",
                            mb_downloaded, mb_total, progress_pct
                        )
                    } else {
                        format!("{:.1} MB", mb_downloaded)
                    };

                    progress_window.update(file_index, TDT_FILES.len(), local_name, progress_text);
                    last_gui_update = std::time::Instant::now();
                }
            }
        }

        // Final update for this file
        #[cfg(target_os = "macos")]
        {
            let mb_total = total_size as f64 / 1024.0 / 1024.0;
            progress_window.update(
                file_index,
                TDT_FILES.len(),
                local_name,
                format!("{:.1} MB - Complete", mb_total),
            );
        }

        if let Some(pb) = pb {
            pb.finish_with_message(format!("✓ {} downloaded", local_name));
        }

        println!("✓ {} downloaded successfully", local_name);
        println!();

        file_index += 1;
    }

    // Close progress window
    #[cfg(target_os = "macos")]
    progress_window.close();

    // Create symlink for encoder.onnx.data if needed
    let encoder_data = dest_dir.join("encoder.onnx.data");
    let encoder_model_data = dest_dir.join("encoder-model.onnx.data");

    if !encoder_data.exists() && encoder_model_data.exists() {
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("encoder-model.onnx.data", &encoder_data)?;
            println!("✓ Created symlink for encoder.onnx.data");
        }

        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(&encoder_model_data, &encoder_data)?;
            println!("✓ Created symlink for encoder.onnx.data");
        }
    }

    Ok(())
}

pub fn get_default_model_path() -> PathBuf {
    // Use ~/.config/vox/models/parakeet-tdt
    // Inline the config dir logic to avoid import issues
    let config_home = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join(".config")
            .to_string_lossy()
            .to_string()
    });
    PathBuf::from(config_home)
        .join("vox")
        .join("models")
        .join("parakeet-tdt")
}
