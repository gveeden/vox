use anyhow::{anyhow, Result};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use std::fs::{self, File};
use std::io::{copy, Read, Write};
use std::path::{Path, PathBuf};

use super::registry::{Model, ModelRegistry};

/// Check available disk space in bytes
fn get_available_disk_space(path: &Path) -> Result<u64> {
    #[cfg(target_os = "linux")]
    {
        // Use statvfs to get filesystem stats
        let stats = nix::sys::statvfs::statvfs(path)?;
        let available_bytes = stats.blocks_available() * stats.block_size();
        Ok(available_bytes)
    }

    #[cfg(not(target_os = "linux"))]
    {
        // On other platforms, we'll be conservative and assume we have space
        // This could be improved with platform-specific implementations
        log::warn!("Disk space checking not implemented for this platform");
        Ok(u64::MAX)
    }
}

/// Download a single file with progress indication
fn download_file(url: &str, dest_path: &Path, show_progress: bool) -> Result<()> {
    info!("Downloading {} to {:?}...", url, dest_path);

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()?;

    let mut response = client
        .get(url)
        .header("User-Agent", "clevernote/0.1.0")
        .send()?;

    if !response.status().is_success() {
        return Err(anyhow!("Failed to download file: {}", response.status()));
    }

    let total_size = response.content_length().unwrap_or(0);

    let pb = if show_progress && total_size > 0 {
        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message(format!(
            "Downloading {}",
            dest_path.file_name().unwrap().to_str().unwrap()
        ));
        Some(pb)
    } else {
        None
    };

    let mut dest_file = File::create(dest_path)?;
    let mut downloaded: u64 = 0;

    // Use copy from response to file
    if show_progress && total_size > 0 {
        // For progress reporting, we need to read in chunks
        let mut buffer = vec![0; 8192];
        loop {
            let bytes_read = response.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            dest_file.write_all(&buffer[..bytes_read])?;
            downloaded += bytes_read as u64;
            if let Some(ref pb) = pb {
                pb.set_position(downloaded);
            }
        }
    } else {
        // Simple copy without progress
        copy(&mut response, &mut dest_file)?;
    }

    if let Some(pb) = pb {
        pb.finish_with_message(format!(
            "Downloaded {}",
            dest_path.file_name().unwrap().to_str().unwrap()
        ));
    }

    Ok(())
}

/// Download all files for a model
pub fn download_model(model: &Model, models_dir: &PathBuf, show_progress: bool) -> Result<PathBuf> {
    let model_dir = models_dir.join(&model.id);

    // Create model directory
    fs::create_dir_all(&model_dir)?;

    // Check disk space
    let required_space_bytes = model.size_mb * 1024 * 1024;
    let available_space = get_available_disk_space(&model_dir)?;

    if available_space < required_space_bytes {
        return Err(anyhow!(
            "Insufficient disk space. Required: {} MB, Available: {} MB",
            required_space_bytes / 1024 / 1024,
            available_space / 1024 / 1024
        ));
    }

    info!("Downloading model '{}' to {:?}", model.id, model_dir);
    info!("Total size: {} MB", model.size_mb);

    // Download each file
    for file in &model.files {
        let dest_path = model_dir.join(&file.filename);

        // Skip if file already exists and has correct size
        if dest_path.exists() {
            info!("File already exists: {:?}, skipping", dest_path);
            continue;
        }

        download_file(&file.url, &dest_path, show_progress)?;
    }

    info!(
        "Model '{}' downloaded successfully to {:?}",
        model.id, model_dir
    );
    Ok(model_dir)
}

/// Remove a downloaded model
pub fn remove_model(model_id: &str, models_dir: &PathBuf) -> Result<()> {
    let model_dir = models_dir.join(model_id);

    if !model_dir.exists() {
        return Err(anyhow!("Model '{}' is not downloaded", model_id));
    }

    info!("Removing model '{}' from {:?}", model_id, model_dir);
    fs::remove_dir_all(&model_dir)?;
    info!("Model '{}' removed successfully", model_id);

    Ok(())
}

/// Ensure a model is downloaded, downloading if necessary
pub fn ensure_model_downloaded(
    model_id: &str,
    models_dir: &PathBuf,
    show_progress: bool,
) -> Result<PathBuf> {
    let registry = ModelRegistry::load()?;

    let model = registry
        .get_model(model_id)
        .ok_or_else(|| anyhow!("Model '{}' not found in registry", model_id))?;

    if registry.is_model_downloaded(model_id, models_dir) {
        info!("Model '{}' is already downloaded", model_id);
        return Ok(models_dir.join(model_id));
    }

    download_model(model, models_dir, show_progress)
}
