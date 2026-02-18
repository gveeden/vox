use anyhow::anyhow;
use anyhow::Result;
use ort::value::Value;

use crate::models::backends::whisper::WhisperBackend;

impl WhisperBackend {
    /// Helper to run decoder with 12 layers (Whisper Small)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decoder_with_layers_12(
        &mut self,
        input_ids_value: Value,
        encoder_value: Value,
        use_cache: Value,
        batch_size: usize,
        num_heads: usize,
        past_seq_len: usize,
        head_dim: usize,
        encoder_seq_len: usize,
        decoder_kv_size: usize,
        encoder_kv_size: usize,
    ) -> Result<ort::session::SessionOutputs> {
        macro_rules! kv {
            () => {
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size],
                ))?
                .into()
            };
            (enc) => {
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size],
                ))?
                .into()
            };
        }

        let (k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11): (
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
        ) = (
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
        );
        let (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11): (
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
        ) = (
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
            kv!(),
        );
        let (ek0, ek1, ek2, ek3, ek4, ek5, ek6, ek7, ek8, ek9, ek10, ek11): (
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
        ) = (
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
        );
        let (ev0, ev1, ev2, ev3, ev4, ev5, ev6, ev7, ev8, ev9, ev10, ev11): (
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
            Value,
        ) = (
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
            kv!(enc),
        );

        self.decoder
            .run(ort::inputs![
                input_ids_value,
                encoder_value,
                k0,
                v0,
                ek0,
                ev0,
                k1,
                v1,
                ek1,
                ev1,
                k2,
                v2,
                ek2,
                ev2,
                k3,
                v3,
                ek3,
                ev3,
                k4,
                v4,
                ek4,
                ev4,
                k5,
                v5,
                ek5,
                ev5,
                k6,
                v6,
                ek6,
                ev6,
                k7,
                v7,
                ek7,
                ev7,
                k8,
                v8,
                ek8,
                ev8,
                k9,
                v9,
                ek9,
                ev9,
                k10,
                v10,
                ek10,
                ev10,
                k11,
                v11,
                ek11,
                ev11,
                use_cache
            ])
            .map_err(|e| anyhow!("Decoder failed: {}", e))
    }

    /// Helper to run decoder with 24 layers (Whisper Medium)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decoder_with_layers_24(
        &mut self,
        input_ids_value: Value,
        encoder_value: Value,
        use_cache: Value,
        batch_size: usize,
        num_heads: usize,
        past_seq_len: usize,
        head_dim: usize,
        encoder_seq_len: usize,
        decoder_kv_size: usize,
        encoder_kv_size: usize,
    ) -> Result<ort::session::SessionOutputs> {
        // Generate all 96 KV tensors inline (24 layers * 4 tensors per layer)
        // Value doesn't implement Clone, so we must create each inline
        self.decoder
            .run(ort::inputs![
                input_ids_value,
                encoder_value,
                // Layer 0
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 1
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 2
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 3
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 4
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 5
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 6
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 7
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 8
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 9
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 10
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 11
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 12
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 13
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 14
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 15
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 16
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 17
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 18
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 19
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 20
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 21
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 22
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 23
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                use_cache
            ])
            .map_err(|e| anyhow!("Decoder failed: {}", e))
    }

    /// Helper to run decoder with 32 layers (Whisper Large)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_decoder_with_layers_32(
        &mut self,
        input_ids_value: Value,
        encoder_value: Value,
        use_cache: Value,
        batch_size: usize,
        num_heads: usize,
        past_seq_len: usize,
        head_dim: usize,
        encoder_seq_len: usize,
        decoder_kv_size: usize,
        encoder_kv_size: usize,
    ) -> Result<ort::session::SessionOutputs> {
        self.decoder
            .run(ort::inputs![
                input_ids_value,
                encoder_value,
                // Layer 0
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 1
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 2
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 3
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 4
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 5
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 6
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 7
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 8
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 9
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 10
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 11
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 12
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 13
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 14
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 15
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 16
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 17
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 18
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 19
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 20
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 21
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 22
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 23
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 24
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 25
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 26
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 27
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 28
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 29
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 30
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                // Layer 31
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, past_seq_len, head_dim],
                    vec![0.0f32; decoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                Value::from_array((
                    [batch_size, num_heads, encoder_seq_len, head_dim],
                    vec![0.0f32; encoder_kv_size]
                ))?,
                use_cache
            ])
            .map_err(|e| anyhow!("Decoder failed: {}", e))
    }
}
