use anyhow::{anyhow, Result};
use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel, Special},
    token::LlamaToken,
};
use std::path::Path;

pub struct GgufBackend {
    backend: LlamaBackend,
    model: LlamaModel,
    n_threads: i32,
}

impl GgufBackend {
    pub fn new<P: AsRef<Path>>(model_path: P, n_threads: usize) -> Result<Self> {
        let path = model_path.as_ref();

        let gguf_path = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().and_then(|e| e.to_str()) == Some("gguf"))
            .ok_or_else(|| anyhow!("No .gguf file found in {:?}", path))?;

        log::info!(
            "Loading GGUF model: {}",
            gguf_path.file_name().unwrap_or_default().to_string_lossy()
        );

        let mut backend = LlamaBackend::init()
            .map_err(|e| anyhow!("Failed to init llama backend: {}", e))?;
        backend.void_logs();

        let model_params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(&backend, &gguf_path, &model_params)
            .map_err(|e| anyhow!("Failed to load GGUF model: {}", e))?;

        log::info!("GGUF model loaded and ready");

        Ok(Self {
            backend,
            model,
            n_threads: n_threads.max(1) as i32,
        })
    }

    pub fn process(&mut self, text: &str, prompt_template: &str) -> Result<String> {
        let prompt = prompt_template.replace("{text}", text);
        self.generate(&prompt)
    }

    fn generate(&mut self, prompt: &str) -> Result<String> {
        // ChatML format — compatible with most modern instruct-tuned models
        // (Qwen, Mistral, many others). Models with a different embedded chat
        // template will still work because the tokenizer sees the raw text.
        let formatted = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            prompt
        );

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZeroU32::new(2048))
            .with_n_threads(self.n_threads)
            .with_n_threads_batch(self.n_threads);

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| anyhow!("Failed to create GGUF context: {}", e))?;

        let tokens = self
            .model
            .str_to_token(&formatted, AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let n_prompt = tokens.len();
        if n_prompt == 0 {
            return Ok(String::new());
        }

        // Prefill: process all prompt tokens, requesting logits only for the last
        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let need_logits = i == n_prompt - 1;
            batch
                .add(token, i as i32, &[0], need_logits)
                .map_err(|e| anyhow!("Batch add failed: {}", e))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Prompt decode failed: {}", e))?;

        // Greedy autoregressive decode
        let eos = self.model.token_eos();
        let mut output = String::new();

        for step in 0..512_usize {
            // Step 0: the last prompt token is at index (n_prompt-1) in the batch.
            // Step 1+: the batch was cleared and contains exactly one token at index 0.
            let logit_idx = if step == 0 { n_prompt as i32 - 1 } else { 0 };
            let logits = ctx.get_logits_ith(logit_idx);
            let next_token = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(id, _)| LlamaToken(id as i32))
                .ok_or_else(|| anyhow!("Empty logits at step {}", step))?;

            if next_token == eos {
                break;
            }

            let piece = self
                .model
                .token_to_str(next_token, Special::Plaintext)
                .map_err(|e| anyhow!("Detokenize failed: {}", e))?;
            output.push_str(&piece);

            batch.clear();
            batch
                .add(next_token, (n_prompt + step) as i32, &[0], true)
                .map_err(|e| anyhow!("Batch add failed at step {}: {}", step, e))?;
            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Decode failed at step {}: {}", step, e))?;
        }

        Ok(output.trim().to_string())
    }
}
