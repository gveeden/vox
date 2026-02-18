use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Represents a single file to download for a model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub url: String,
    pub filename: String,
}

/// Represents a model in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: String,
    pub repository: String,
    pub quantization: String,
    pub model_type: String,
    pub size_mb: u64,
    #[serde(default)]
    pub default: bool,
    #[serde(default = "default_backend")]
    pub backend: String,
    pub files: Vec<ModelFile>,
}

fn default_backend() -> String {
    "parakeet".to_string()
}

/// The model registry loaded from models.json
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub models: Vec<Model>,
}

impl ModelRegistry {
    /// Load the model registry from the embedded models.json
    /// Merges with user config at ~/.config/clevernote/models.json if present
    pub fn load() -> Result<Self> {
        // Always load embedded defaults first
        let embedded_json = include_str!("../../models.json");
        let mut registry: ModelRegistry = serde_json::from_str(embedded_json)?;

        // Try to load user overrides/additions
        let config_dir = crate::get_config_dir();
        let user_registry_path = config_dir.join("models.json");

        if user_registry_path.exists() {
            let user_json = fs::read_to_string(&user_registry_path)?;
            let user_registry: ModelRegistry = serde_json::from_str(&user_json)?;

            // Merge: user models override or add to embedded models
            for user_model in user_registry.models {
                // Check if model with this ID already exists
                if let Some(existing_idx) =
                    registry.models.iter().position(|m| m.id == user_model.id)
                {
                    // Override existing model
                    registry.models[existing_idx] = user_model;
                } else {
                    // Add new model
                    registry.models.push(user_model);
                }
            }
        }

        Ok(registry)
    }

    /// Copy the default registry to the config directory so users can edit it
    pub fn copy_default_to_config() -> Result<PathBuf> {
        let config_dir = crate::get_config_dir();
        fs::create_dir_all(&config_dir)?;
        let registry_path = config_dir.join("models.json");

        if !registry_path.exists() {
            let registry_json = include_str!("../../models.json");
            fs::write(&registry_path, registry_json)?;
        }

        Ok(registry_path)
    }

    /// Get a model by ID
    pub fn get_model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Get the default model
    pub fn get_default_model(&self) -> Option<&Model> {
        self.models.iter().find(|m| m.default)
    }

    /// List all available models
    pub fn list_models(&self) -> &[Model] {
        &self.models
    }

    /// Check if a model exists locally
    pub fn is_model_downloaded(&self, model_id: &str, models_dir: &PathBuf) -> bool {
        if let Some(model) = self.get_model(model_id) {
            let model_dir = models_dir.join(model_id);
            if !model_dir.exists() {
                return false;
            }

            // Check if all required files exist
            for file in &model.files {
                let file_path = model_dir.join(&file.filename);
                if !file_path.exists() {
                    return false;
                }
            }
            true
        } else {
            false
        }
    }

    /// Get the path to a downloaded model
    pub fn get_model_path(&self, model_id: &str, models_dir: &PathBuf) -> Result<PathBuf> {
        if !self.is_model_downloaded(model_id, models_dir) {
            return Err(anyhow!("Model '{}' is not downloaded", model_id));
        }
        Ok(models_dir.join(model_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_registry() {
        let registry = ModelRegistry::load().unwrap();
        assert!(!registry.models.is_empty());
    }

    #[test]
    fn test_get_default_model() {
        let registry = ModelRegistry::load().unwrap();
        let default_model = registry.get_default_model();
        assert!(default_model.is_some());
        assert_eq!(default_model.unwrap().id, "parakeet-tdt-0.6b-v3-int8");
    }
}
