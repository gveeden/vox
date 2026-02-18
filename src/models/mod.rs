pub mod backends;
pub mod downloader;
pub mod registry;

pub use downloader::{download_model, ensure_model_downloaded, remove_model};
pub use registry::{Model, ModelFile, ModelRegistry};
