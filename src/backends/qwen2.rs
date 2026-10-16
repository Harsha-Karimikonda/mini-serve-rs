//! Optimized Metal/CUDA/CPU native Qwen2 model implementation.
//! Replaces unsupported custom-op RMSNorm with native tensor operations for full Metal GPU acceleration.

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, linear_no_bias, Activation, Linear, VarBuilder};
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: Option<usize>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub attention_bias: Option<bool>,
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn default_rope_theta() -> f64 {
    1000000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = (variance + self.eps)?.sqrt()?;
        let x_normed = x_f32.broadcast_div(&norm)?.to_dtype(dtype)?;
        x_normed.broadcast_mul(&self.weight)
    }
}

impl Module for RmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    pub sin: Tensor,
    pub cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = apply_rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = apply_rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    let x_cos = x.broadcast_mul(&cos)?;
    let x_sin = rotate_half(x)?.broadcast_mul(&sin)?;
    x_cos.broadcast_add(&x_sin)
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

/// Computes single-token (seq_len=1) attention using grouped-query attention without copying or repeating KV cache.
/// `q`: shape `[num_heads, head_dim]`
/// `k`: shape `[num_kv_heads, kv_len, head_dim]`
/// `v`: shape `[num_kv_heads, kv_len, head_dim]`
/// Returns: shape `[num_heads, head_dim]`
pub fn decode_attention_gqa_zero_copy(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_kv_heads: usize,
    num_kv_groups: usize,
    scale: f64,
) -> Result<Tensor> {
    let mut head_outputs = Vec::with_capacity(num_kv_heads);
    for h in 0..num_kv_heads {
        let q_group = q.narrow(0, h * num_kv_groups, num_kv_groups)?; // [num_kv_groups, head_dim]
        let k_mat = k.narrow(0, h, 1)?.squeeze(0)?; // [kv_len, head_dim]
        let v_mat = v.narrow(0, h, 1)?.squeeze(0)?; // [kv_len, head_dim]

        let k_t = k_mat.t()?; // [head_dim, kv_len]
        let scores = (q_group.matmul(&k_t)? * scale)?; // [num_kv_groups, kv_len]
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out_group = probs.matmul(&v_mat)?; // [num_kv_groups, head_dim]
        head_outputs.push(out_group);
    }
    Tensor::cat(&head_outputs, 0)
}

#[derive(Debug, Clone)]
pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_kv_groups: usize,
    pub head_dim: usize,
    pub rotary_emb: Arc<RotaryEmbedding>,
    pub max_seq_len: usize,
    pub kv_cache: Option<candle_nn::kv_cache::KvCache>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings.max(4096);

        let has_bias =
            vb.pp("q_proj").contains_tensor("bias") || cfg.attention_bias.unwrap_or(false);

        let q_proj = if has_bias {
            linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?
        } else {
            linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?
        };
        let k_proj = if has_bias {
            linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?
        } else {
            linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?
        };
        let v_proj = if has_bias {
            linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?
        } else {
            linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?
        };
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        let q_norm = if vb.pp("q_norm").contains_tensor("weight") {
            Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?)
        } else {
            None
        };
        let k_norm = if vb.pp("k_norm").contains_tensor("weight") {
            Some(RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            max_seq_len,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let query_states = match &self.q_norm {
            Some(norm) => norm.forward(&query_states)?,
            None => query_states,
        };
        let key_states = match &self.k_norm {
            Some(norm) => norm.forward(&key_states)?,
            None => key_states,
        };

        let query_states = query_states.transpose(1, 2)?;
        let key_states = key_states.transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = self.append_kv(&key_states, &value_states)?;

        let attn_output = if b_sz == 1 && q_len == 1 && attention_mask.is_none() {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let q_b = query_states.squeeze(0)?.squeeze(1)?;
            let k_b = key_states.squeeze(0)?;
            let v_b = value_states.squeeze(0)?;
            let out_2d = decode_attention_gqa_zero_copy(
                &q_b,
                &k_b,
                &v_b,
                self.num_kv_heads,
                self.num_kv_groups,
                scale,
            )?;
            out_2d.reshape((1, 1, self.num_heads * self.head_dim))?
        } else {
            let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
            let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
            attn_weights
                .matmul(&value_states)?
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
        };
        attn_output.apply(&self.o_proj)
    }

    pub fn append_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        match &mut self.kv_cache {
            None => {
                let mut cache = candle_nn::kv_cache::KvCache::new(2, self.max_seq_len);
                let res = cache.append(&k, &v)?;
                self.kv_cache = Some(cache);
                Ok(res)
            }
            Some(cache) => cache.append(&k, &v),
        }
    }

    pub fn get_kv_snapshot(&self) -> Option<(Tensor, Tensor)> {
        if let Some(cache) = &self.kv_cache {
            if let (Ok(Some(k)), Ok(Some(v))) = (cache.k(), cache.v()) {
                return Some((k, v));
            }
        }
        None
    }

    pub fn set_kv_snapshot(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        let mut cache = candle_nn::kv_cache::KvCache::new(2, self.max_seq_len);
        cache.append(&k.contiguous()?, &v.contiguous()?)?;
        self.kv_cache = Some(cache);
        Ok(())
    }

    fn clear_kv_cache(&mut self) {
        if let Some(cache) = &mut self.kv_cache {
            cache.reset();
        }
        self.kv_cache = None;
    }
}

#[derive(Debug, Clone)]
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }

    pub fn get_kv_snapshot(&self) -> Option<(Tensor, Tensor)> {
        self.self_attn.get_kv_snapshot()
    }

    pub fn set_kv_snapshot(&mut self, k: &Tensor, v: &Tensor) -> Result<()> {
        self.self_attn.set_kv_snapshot(k, v)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<DecoderLayer>,
    pub norm: RmsNorm,
    pub sliding_window: Option<usize>,
    pub device: Device,
    pub dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let sliding_window = self.sliding_window;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || sliding_window.is_some_and(|w| j + w < i) {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask =
            Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?.to_dtype(self.dtype)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        self.norm.forward(&xs)
    }

    pub fn get_kv_snapshot(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers.iter().map(|l| l.get_kv_snapshot()).collect()
    }

    pub fn set_kv_snapshot(&mut self, snapshot: &[(Tensor, Tensor)]) -> Result<()> {
        if snapshot.len() != self.layers.len() {
            candle_core::bail!("KV snapshot layer count mismatch")
        }
        let mut expected_len = None;
        // Validate the entire snapshot before changing any layer.
        for (layer, (k, v)) in self.layers.iter().zip(snapshot) {
            let (batch, heads, len, dim) = k.dims4()?;
            let attn = &layer.self_attn;
            if v.dims() != k.dims()
                || batch != 1
                || heads != attn.num_kv_heads
                || dim != attn.head_dim
                || len > attn.max_seq_len
                || expected_len.is_some_and(|expected| expected != len)
            {
                candle_core::bail!("Invalid KV snapshot shape")
            }
            expected_len = Some(len);
        }
        // Build independent mutable buffers; cached tensor views remain read-only.
        let mut restored = self.clone();
        for (layer, (k, v)) in restored.layers.iter_mut().zip(snapshot) {
            layer.set_kv_snapshot(k, v)?;
        }
        *self = restored;
        Ok(())
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    pub base_model: Model,
    pub lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder<'_>) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            Linear::new(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    /// Set the physical contiguous KV allocation before this request's first forward.
    pub fn set_kv_capacity(&mut self, tokens: usize) -> Result<()> {
        if tokens == 0
            || self
                .base_model
                .layers
                .iter()
                .any(|layer| layer.self_attn.kv_cache.is_some())
        {
            candle_core::bail!("KV capacity must be positive and set before initialization")
        }
        for layer in &mut self.base_model.layers {
            layer.self_attn.max_seq_len = tokens;
        }
        Ok(())
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        self.base_model
            .forward(input_ids, seqlen_offset, None)?
            .narrow(1, seq_len - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Single batched forward pass for concurrent sequence decode steps.
    /// Runs all heavy weight projections (MLP, QKV, LM Head) batched on the GPU.
    pub fn step_decode_batch(
        models: &mut [&mut Self],
        input_ids: &[u32],
        positions: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let b_sz = models.len();
        if b_sz == 0 {
            return Err(candle_core::Error::Msg(
                "Empty batch provided to step_decode_batch".to_string(),
            ));
        }
        if b_sz == 1 {
            let input_tensor = Tensor::new(&[input_ids[0]], device)?.unsqueeze(0)?;
            let logits = models[0].forward(&input_tensor, positions[0])?;
            let last_logits = logits.squeeze(1)?;
            return Ok(last_logits);
        }

        // 1. Batched Token Embeddings: [b_sz, 1] -> [b_sz, 1, hidden_size]
        let input_tensor = Tensor::new(input_ids, device)?.reshape((b_sz, 1))?;
        let mut xs = models[0].base_model.embed_tokens.forward(&input_tensor)?;

        // Convert positions to Tensor for batched RoPE index_select on Metal
        let pos_u32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let pos_tensor = Tensor::new(pos_u32.as_slice(), device)?;

        let num_heads = models[0].base_model.layers[0].self_attn.num_heads;
        let num_kv_heads = models[0].base_model.layers[0].self_attn.num_kv_heads;
        let num_kv_groups = models[0].base_model.layers[0].self_attn.num_kv_groups;
        let head_dim = models[0].base_model.layers[0].self_attn.head_dim;
        let scale = 1f64 / f64::sqrt(head_dim as f64);

        for l in 0..models[0].base_model.layers.len() {
            let residual = xs.clone();
            let norm_xs = models[0].base_model.layers[l]
                .input_layernorm
                .forward(&xs)?;

            // Batched Q, K, V Projections across all sequences
            let q = models[0].base_model.layers[l]
                .self_attn
                .q_proj
                .forward(&norm_xs)?;
            let k = models[0].base_model.layers[l]
                .self_attn
                .k_proj
                .forward(&norm_xs)?;
            let v = models[0].base_model.layers[l]
                .self_attn
                .v_proj
                .forward(&norm_xs)?;

            let q = q.reshape((b_sz, 1, num_heads, head_dim))?;
            let k = k.reshape((b_sz, 1, num_kv_heads, head_dim))?;
            let v = v
                .reshape((b_sz, 1, num_kv_heads, head_dim))?
                .transpose(1, 2)?;

            let q = match &models[0].base_model.layers[l].self_attn.q_norm {
                Some(norm) => norm.forward(&q)?,
                None => q,
            };
            let k = match &models[0].base_model.layers[l].self_attn.k_norm {
                Some(norm) => norm.forward(&k)?,
                None => k,
            };

            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;

            // Batched RoPE across all sequences with their individual positions
            let (q, k) = {
                let rot = &models[0].base_model.layers[l].self_attn.rotary_emb;
                let cos = rot.cos.index_select(&pos_tensor, 0)?;
                let sin = rot.sin.index_select(&pos_tensor, 0)?;
                let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?
                    .unsqueeze(1)?
                    .unsqueeze(1)?;
                let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?
                    .unsqueeze(1)?
                    .unsqueeze(1)?;
                let q_embed = q
                    .broadcast_mul(&cos)?
                    .broadcast_add(&rotate_half(&q.contiguous()?)?.broadcast_mul(&sin)?)?;
                let k_embed = k
                    .broadcast_mul(&cos)?
                    .broadcast_add(&rotate_half(&k.contiguous()?)?.broadcast_mul(&sin)?)?;
                (q_embed, k_embed)
            };

            // Attention per sequence using each sequence's own pre-allocated KV cache with zero-copy GQA
            let mut attn_outputs = Vec::with_capacity(b_sz);
            for (b, model_b) in models.iter_mut().enumerate() {
                let q_b = q.narrow(0, b, 1)?.squeeze(0)?.squeeze(1)?; // [num_heads, head_dim]
                let k_b = k.narrow(0, b, 1)?;
                let v_b = v.narrow(0, b, 1)?;

                let (past_k, past_v) = model_b.base_model.layers[l]
                    .self_attn
                    .append_kv(&k_b, &v_b)?;

                let past_k = past_k.squeeze(0)?; // [num_kv_heads, kv_len, head_dim]
                let past_v = past_v.squeeze(0)?; // [num_kv_heads, kv_len, head_dim]

                let out_2d = decode_attention_gqa_zero_copy(
                    &q_b,
                    &past_k,
                    &past_v,
                    num_kv_heads,
                    num_kv_groups,
                    scale,
                )?;
                attn_outputs.push(out_2d.unsqueeze(0)?); // [1, num_heads, head_dim]
            }

            let attn_output = Tensor::cat(&attn_outputs, 0)?; // [b_sz, num_heads, head_dim]
            let attn_proj = attn_output
                .reshape((b_sz, 1, num_heads * head_dim))?
                .apply(&models[0].base_model.layers[l].self_attn.o_proj)?;

            let xs_mid = (residual + attn_proj)?;
            let norm_mid = models[0].base_model.layers[l]
                .post_attention_layernorm
                .forward(&xs_mid)?;
            let mlp_out = models[0].base_model.layers[l].mlp.forward(&norm_mid)?;
            xs = (xs_mid + mlp_out)?;
        }

        let xs = models[0].base_model.norm.forward(&xs)?;
        let logits = xs.narrow(1, 0, 1)?.apply(&models[0].lm_head)?;

        let last_logits = logits.squeeze(1)?;
        Ok(last_logits)
    }

    pub fn get_kv_snapshot(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.base_model.get_kv_snapshot()
    }

    pub fn set_kv_snapshot(&mut self, snapshot: &[(Tensor, Tensor)]) -> Result<()> {
        self.base_model.set_kv_snapshot(snapshot)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_decode_batch_execution() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 100,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            sliding_window: None,
            max_window_layers: None,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
            head_dim: Some(16),
            attention_bias: Some(false),
        };

        let vb = VarBuilder::zeros(DType::F32, &dev);
        let mut model1 = ModelForCausalLM::new(&cfg, vb.clone())?;
        let mut model2 = ModelForCausalLM::new(&cfg, vb.clone())?;
        let mut model3 = ModelForCausalLM::new(&cfg, vb)?;

        // Test batched decode step with 3 sequences at different positions
        let input_ids = [42u32, 17u32, 88u32];
        let positions = [0usize, 5usize, 12usize];

        let mut models = [&mut model1, &mut model2, &mut model3];
        let logits =
            ModelForCausalLM::step_decode_batch(&mut models, &input_ids, &positions, &dev)?;

        assert_eq!(logits.dims2()?, (3, 100));
        Ok(())
    }

    #[test]
    fn test_gqa_broadcast_matmul() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let b_sz = 2;
        let num_kv_heads = 2;
        let num_kv_groups = 7;
        let num_heads = 14;
        let head_dim = 64;
        let kv_len = 100;

        let q = Tensor::randn(0.0f32, 1.0f32, (b_sz, num_heads, 1, head_dim), &dev)?;
        let k = Tensor::randn(0.0f32, 1.0f32, (b_sz, num_kv_heads, kv_len, head_dim), &dev)?;
        let v = Tensor::randn(0.0f32, 1.0f32, (b_sz, num_kv_heads, kv_len, head_dim), &dev)?;

        // 1. Traditional repeat_kv method:
        let scale = 1f64 / f64::sqrt(head_dim as f64);
        let k_rep = repeat_kv(k.clone(), num_kv_groups)?.contiguous()?;
        let v_rep = repeat_kv(v.clone(), num_kv_groups)?.contiguous()?;
        let scores_rep = (q.matmul(&k_rep.transpose(2, 3)?)? * scale)?;
        let probs_rep = candle_nn::ops::softmax(&scores_rep, candle_core::D::Minus1)?;
        let expected_out =
            probs_rep
                .matmul(&v_rep)?
                .transpose(1, 2)?
                .reshape((b_sz, 1, num_heads * head_dim))?;

        // 2. Zero-copy GQA method:
        let q_b = q.narrow(0, 0, 1)?.squeeze(0)?.squeeze(1)?; // [14, 64]
        let k_b = k.narrow(0, 0, 1)?.squeeze(0)?; // [2, 100, 64]
        let v_b = v.narrow(0, 0, 1)?.squeeze(0)?; // [2, 100, 64]

        let mut head_outputs = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            let q_group = q_b.narrow(0, h * num_kv_groups, num_kv_groups)?; // [7, 64]
            let k_mat = k_b.narrow(0, h, 1)?.squeeze(0)?; // [100, 64] (2D)
            let v_mat = v_b.narrow(0, h, 1)?.squeeze(0)?; // [100, 64] (2D)

            let k_t = k_mat.t()?; // [64, 100] (2D)
            let scores = (q_group.matmul(&k_t)? * scale)?; // [7, 100]
            let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
            let out_group = probs.matmul(&v_mat)?; // [7, 64]
            head_outputs.push(out_group);
        }
        let full_out = Tensor::cat(&head_outputs, 0)?; // [14, 64]
        let actual_out = full_out.reshape((1, 1, num_heads * head_dim))?;

        // Verify batch 0 slice matches expected
        let expected_b0 = expected_out.narrow(0, 0, 1)?;
        let diff = (expected_b0 - actual_out)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "Max diff between repeat_kv and zero-copy GQA is {diff}"
        );
        Ok(())
    }

    #[test]
    fn test_kv_snapshot_save_and_restore() -> candle_core::Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            hidden_size: 64,
            intermediate_size: 128,
            vocab_size: 100,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 512,
            sliding_window: None,
            max_window_layers: None,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
            head_dim: None,
            attention_bias: None,
        };

        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &dev);

        let mut model1 = ModelForCausalLM::new(&cfg, vb.clone())?;
        let mut model2 = ModelForCausalLM::new(&cfg, vb)?;

        // 1. Model 1 prefills 4 tokens: [10, 20, 30, 40]
        let input1 = Tensor::new(&[[10u32, 20, 30, 40]], &dev)?;
        let _ = model1.forward(&input1, 0)?;

        // 2. Extract snapshot from Model 1
        let snapshot = model1.get_kv_snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot.iter().all(|s| s.is_some()));

        let pairs: Vec<(Tensor, Tensor)> = snapshot.into_iter().flatten().collect();
        assert_eq!(pairs[0].0.dims4()?, (1, 2, 4, 16)); // [1, num_kv_heads, 4, head_dim]

        // 3. Restore snapshot into Model 2
        model2.set_kv_snapshot(&pairs)?;

        // 4. Both models now forward token [50] with seqlen_offset = 4
        let next_input = Tensor::new(&[[50u32]], &dev)?;
        let logits1 = model1.forward(&next_input, 4)?;
        let logits2 = model2.forward(&next_input, 4)?;

        // Verify logits are identical
        let diff = (logits1 - logits2)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "Max diff between original and restored model logits: {diff}"
        );

        Ok(())
    }
}
