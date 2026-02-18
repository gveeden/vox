use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Trait for tokenizers
pub trait Tokenizer: Send {
    /// Decode token IDs to text
    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String>;

    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get the tokenizer type
    fn tokenizer_type(&self) -> &'static str;
}

/// Simple vocabulary tokenizer (vocab.txt format)
pub struct VocabTxtTokenizer {
    token_to_id: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    pad_token: i64,
    unk_token: i64,
}

impl VocabTxtTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (idx, line) in content.lines().enumerate() {
            let token = line.trim().to_string();
            let id = idx as i64;
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }

        let pad_token = *token_to_id.get("<pad>").unwrap_or(&0);
        let unk_token = *token_to_id.get("<unk>").unwrap_or(&0);

        Ok(Self {
            token_to_id,
            id_to_token,
            pad_token,
            unk_token,
        })
    }
}

impl Tokenizer for VocabTxtTokenizer {
    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String> {
        let mut text = String::new();

        for &token in tokens {
            if skip_special_tokens && (token == self.pad_token || token == self.unk_token) {
                continue;
            }

            if let Some(token_str) = self.id_to_token.get(&token) {
                // Remove BPE prefix if present
                let clean_token = if token_str.starts_with("##") {
                    &token_str[2..]
                } else {
                    token_str
                };

                if !text.is_empty()
                    && !clean_token.starts_with("##")
                    && !clean_token.starts_with("'")
                {
                    text.push(' ');
                }
                text.push_str(clean_token);
            }
        }

        Ok(text)
    }

    fn encode(&self, text: &str, _add_special_tokens: bool) -> Result<Vec<i64>> {
        // Simple word-based tokenization
        let tokens: Vec<i64> = text
            .split_whitespace()
            .map(|word| *self.token_to_id.get(word).unwrap_or(&self.unk_token))
            .collect();
        Ok(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn tokenizer_type(&self) -> &'static str {
        "vocab_txt"
    }
}

/// JSON vocabulary tokenizer (vocab.json format, BPE)
pub struct VocabJsonTokenizer {
    token_to_id: HashMap<String, i64>,
    id_to_token: HashMap<i64, String>,
    pad_token: i64,
    unk_token: i64,
    bos_token: i64,
    eos_token: i64,
}

impl VocabJsonTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let vocab: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&content)?;

        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (token, id_val) in vocab {
            if let Some(id) = id_val.as_i64() {
                token_to_id.insert(token.clone(), id);
                id_to_token.insert(id, token);
            }
        }

        let pad_token = *token_to_id.get("<|endoftext|>").unwrap_or(&0);
        let unk_token = *token_to_id.get("<|endoftext|>").unwrap_or(&0);
        let bos_token = *token_to_id.get("<|startoftranscript|>").unwrap_or(&50258);
        let eos_token = *token_to_id.get("<|endoftext|>").unwrap_or(&50256);

        Ok(Self {
            token_to_id,
            id_to_token,
            pad_token,
            unk_token,
            bos_token,
            eos_token,
        })
    }

    /// Get the BOS token ID
    pub fn bos_token(&self) -> i64 {
        self.bos_token
    }

    /// Get the EOS token ID
    pub fn eos_token(&self) -> i64 {
        self.eos_token
    }
}

impl Tokenizer for VocabJsonTokenizer {
    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String> {
        let mut text = String::new();

        for &token in tokens {
            if skip_special_tokens
                && (token == self.pad_token
                    || token == self.bos_token
                    || token == self.eos_token
                    || token == self.unk_token)
            {
                continue;
            }

            if let Some(token_str) = self.id_to_token.get(&token) {
                // Handle Whisper special tokens
                if token_str.starts_with("<|") && token_str.ends_with("|>") {
                    if !skip_special_tokens {
                        text.push_str(token_str);
                    }
                    continue;
                }

                // Handle BPE tokens (replace "Ġ" with space)
                let clean_token = token_str.replace("Ġ", " ");
                text.push_str(&clean_token);
            }
        }

        Ok(text.trim().to_string())
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        // Simple BPE encoding (simplified - production would use full BPE)
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(self.bos_token);
        }

        // Tokenize by spaces and look up each word
        for word in text.split_whitespace() {
            let token_str = format!("Ġ{}", word);
            if let Some(&id) = self.token_to_id.get(&token_str) {
                tokens.push(id);
            } else {
                tokens.push(self.unk_token);
            }
        }

        if add_special_tokens {
            tokens.push(self.eos_token);
        }

        Ok(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    fn tokenizer_type(&self) -> &'static str {
        "vocab_json"
    }
}

/// HuggingFace tokenizer wrapper
pub struct HFTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HFTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("Failed to load HF tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }
}

impl Tokenizer for HFTokenizer {
    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(|e| anyhow!("Decode error: {}", e))?;
        Ok(text)
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow!("Encode error: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        Ok(ids)
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn tokenizer_type(&self) -> &'static str {
        "hf_tokenizer"
    }
}

/// Factory function to create appropriate tokenizer
pub fn create_tokenizer<P: AsRef<Path>>(tokenizer_path: P) -> Result<Box<dyn Tokenizer>> {
    let path = tokenizer_path.as_ref();

    if let Some(ext) = path.extension() {
        match ext.to_str() {
            Some("txt") => Ok(Box::new(VocabTxtTokenizer::from_file(path)?)),
            Some("json") => Ok(Box::new(VocabJsonTokenizer::from_file(path)?)),
            Some("model") => {
                // SentencePiece - would need sentencepiece crate
                // For now, return error
                Err(anyhow!(
                    "SentencePiece tokenizer not yet implemented. File: {:?}",
                    path
                ))
            }
            _ => {
                // Try HF tokenizer
                if path.file_name().map_or(false, |n| n == "tokenizer.json") {
                    Ok(Box::new(HFTokenizer::from_file(path)?))
                } else {
                    Err(anyhow!("Unknown tokenizer format: {:?}", path))
                }
            }
        }
    } else {
        Err(anyhow!("Tokenizer file has no extension: {:?}", path))
    }
}

/// Create tokenizer by model type and directory
pub fn create_tokenizer_for_model(
    model_path: &Path,
    model_type: &str,
) -> Result<Box<dyn Tokenizer>> {
    match model_type {
        "parakeet-tdt" => {
            let vocab_path = model_path.join("vocab.txt");
            if vocab_path.exists() {
                create_tokenizer(vocab_path)
            } else {
                Err(anyhow!("vocab.txt not found in {:?}", model_path))
            }
        }
        "whisper" => {
            let vocab_path = model_path.join("vocab.json");
            if vocab_path.exists() {
                create_tokenizer(vocab_path)
            } else {
                Err(anyhow!("vocab.json not found in {:?}", model_path))
            }
        }
        "moonshine" => {
            let tokenizer_path = model_path.join("tokenizer.json");
            if tokenizer_path.exists() {
                create_tokenizer(tokenizer_path)
            } else {
                Err(anyhow!("tokenizer.json not found in {:?}", model_path))
            }
        }
        "sensevoice" => {
            let tokenizer_path = model_path.join("tokenizer.model");
            if tokenizer_path.exists() {
                create_tokenizer(tokenizer_path)
            } else {
                Err(anyhow!("tokenizer.model not found in {:?}", model_path))
            }
        }
        _ => Err(anyhow!("Unknown model type: {}", model_type)),
    }
}
