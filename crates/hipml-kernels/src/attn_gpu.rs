//! The full-attention block on the device: resident weights, one layer call,
//! and the small kernels that sit between the projections and the attention.
//!
//! Layout is the CPU reference's (`attention::KvState`): the cache is
//! position-major, `[n_ctx][n_kv * head_dim]`, appended one row per token. The
//! decode kernel takes `head_stride`/`row_stride`, so the same entry point
//! serves that layout and the kv-major parity case.

#[cfg(feature = "hip")]
pub use imp::AttnLayer;

#[cfg(feature = "hip")]
mod imp {
    use crate::launch::device::Launcher;
    use hipml_core::attention::FullAttnW;
    use hipml_core::config::Config;
    use hipml_core::mat::{Kind, Mat};
    use hipml_hip::{Buffer, ByteBuffer, LaunchConfig};
    use std::ffi::c_void;

    /// Upper bound on sequence splits; the partial buffers are sized for it.
    const MAX_SPLITS: usize = 64;
    /// Must match SPLIT_TILE in attn_decode_split_f32.hip.
    const SPLIT_TILE: usize = 8;

    /// The prefill-chunk launches (child so they see the layer's private
    /// scratch and weight handles).
    #[path = "attn_chunk.rs"]
    mod chunk;

    /// A weight payload that is either owned (the standalone parity path) or
    /// borrowed from the resident loader, so the graph never uploads the same
    /// bytes twice.
    enum WeightBuf<'a> {
        Owned(ByteBuffer),
        Borrowed(&'a ByteBuffer),
    }

    impl WeightBuf<'_> {
        fn as_ptr(&self) -> *mut c_void {
            match self {
                WeightBuf::Owned(b) => b.as_ptr(),
                WeightBuf::Borrowed(b) => b.as_ptr(),
            }
        }

        fn len(&self) -> usize {
            match self {
                WeightBuf::Owned(b) => b.len(),
                WeightBuf::Borrowed(b) => b.len(),
            }
        }

        fn bytes(&self) -> &ByteBuffer {
            match self {
                WeightBuf::Owned(b) => b,
                WeightBuf::Borrowed(b) => b,
            }
        }
    }

    /// One weight tensor resident in VRAM, in its GGUF block form.
    struct ResidentMat<'a> {
        data: WeightBuf<'a>,
        rows: usize,
        cols: usize,
        kind: Kind,
    }

    impl ResidentMat<'_> {
        fn upload(m: &Mat) -> Result<ResidentMat<'static>, String> {
            Ok(ResidentMat {
                data: WeightBuf::Owned(ByteBuffer::from_host(m.data.as_slice())?),
                rows: m.rows,
                cols: m.cols,
                kind: m.kind,
            })
        }

        fn borrow(data: &ByteBuffer, rows: usize, cols: usize, kind: Kind) -> ResidentMat<'_> {
            ResidentMat {
                data: WeightBuf::Borrowed(data),
                rows,
                cols,
                kind,
            }
        }

        fn entry(&self) -> Result<&'static str, String> {
            Ok(match self.kind {
                Kind::F32 => "gemv_f32",
                Kind::Q4K => "gemv_q4_k",
                Kind::Q6K => "gemv_q6_k",
                Kind::Q8_0 => "gemv_q8_0",
                Kind::Q5K => return Err("attention weight is Q5_K: no GEMV kernel".to_string()),
            })
        }

        fn data_bytes(&self) -> &ByteBuffer {
            self.data.bytes()
        }

        /// The tiled prefill GEMM for the same payload (`tokens` rows of x).
        fn prefill_entry(&self) -> Result<&'static str, String> {
            Ok(match self.kind {
                Kind::Q4K => "prefill_q4_k",
                Kind::Q6K => "prefill_q6_k",
                Kind::Q8_0 => "prefill_q8_0",
                other => {
                    return Err(format!(
                        "attention weight {other:?}: no tiled prefill kernel"
                    ))
                }
            })
        }
    }

    /// Shape and rope parameters one layer needs.
    #[derive(Clone, Copy, Debug)]
    pub struct LayerDims {
        pub n_head: usize,
        pub n_kv: usize,
        pub head_dim: usize,
        pub n_rot: usize,
        pub sections: [usize; 4],
        pub eps: f32,
        pub rope_base: f32,
        pub width: usize,
        pub n_ctx: usize,
        pub n_embd: usize,
    }

    /// One attention layer whose weights and KV cache stay on the device.
    pub struct AttnLayer<'a> {
        wq: ResidentMat<'a>,
        wk: ResidentMat<'a>,
        wv: ResidentMat<'a>,
        wo: ResidentMat<'a>,
        q_norm: Buffer,
        k_norm: Buffer,
        qg: Buffer,
        q: Buffer,
        gate: Buffer,
        k: Buffer,
        v: Buffer,
        out: Buffer,
        y: Buffer,
        kv_k: Buffer,
        kv_v: Buffer,
        part_acc: Buffer,
        part_ml: Buffer,
        pub dims: LayerDims,
    }

    impl<'a> AttnLayer<'a> {
        pub fn new(
            l: &Launcher,
            w: &FullAttnW,
            cfg: &Config,
            n_ctx: usize,
        ) -> Result<AttnLayer<'static>, String> {
            let (n_head, n_kv, hd) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
            let width = n_kv * hd;
            if w.wq.rows != n_head * hd * 2 || w.wq.cols != cfg.n_embd {
                return Err(format!(
                    "attn_q is {}x{}, expected {}x{}",
                    w.wq.rows,
                    w.wq.cols,
                    n_head * hd * 2,
                    cfg.n_embd
                ));
            }
            for (name, mat, rows, cols) in [
                ("attn_k", &w.wk, n_kv * hd, cfg.n_embd),
                ("attn_v", &w.wv, n_kv * hd, cfg.n_embd),
                ("attn_output", &w.wo, cfg.n_embd, n_head * hd),
            ] {
                if mat.rows != rows || mat.cols != cols {
                    return Err(format!(
                        "{name} is {}x{}, expected {rows}x{cols}",
                        mat.rows, mat.cols
                    ));
                }
            }
            let dims = LayerDims {
                n_head,
                n_kv,
                head_dim: hd,
                n_rot: cfg.n_rot,
                sections: cfg.rope_sections,
                eps: cfg.eps,
                rope_base: cfg.rope_base,
                width,
                n_ctx,
                n_embd: cfg.n_embd,
            };
            let _ = l;
            Ok(AttnLayer {
                wq: ResidentMat::upload(&w.wq)?,
                wk: ResidentMat::upload(&w.wk)?,
                wv: ResidentMat::upload(&w.wv)?,
                wo: ResidentMat::upload(&w.wo)?,
                q_norm: Buffer::from_host(&w.q_norm)?,
                k_norm: Buffer::from_host(&w.k_norm)?,
                qg: Buffer::alloc(2 * n_head * hd)?,
                q: Buffer::alloc(n_head * hd)?,
                gate: Buffer::alloc(n_head * hd)?,
                k: Buffer::alloc(width)?,
                v: Buffer::alloc(width)?,
                out: Buffer::alloc(n_head * hd)?,
                y: Buffer::alloc(cfg.n_embd)?,
                kv_k: Buffer::alloc(n_ctx * width)?,
                kv_v: Buffer::alloc(n_ctx * width)?,
                part_acc: Buffer::alloc(n_head * MAX_SPLITS * hd)?,
                part_ml: Buffer::alloc(n_head * MAX_SPLITS * 2)?,
                dims,
            })
        }

        /// A layer whose four projections are already resident in the loader.
        #[allow(clippy::too_many_arguments)]
        pub fn new_resident(
            l: &Launcher,
            wq: (&'a ByteBuffer, usize, usize, Kind),
            wk: (&'a ByteBuffer, usize, usize, Kind),
            wv: (&'a ByteBuffer, usize, usize, Kind),
            wo: (&'a ByteBuffer, usize, usize, Kind),
            q_norm: &[f32],
            k_norm: &[f32],
            cfg: &Config,
            n_ctx: usize,
        ) -> Result<AttnLayer<'a>, String> {
            let (n_head, n_kv, hd) = (cfg.n_head, cfg.n_head_kv, cfg.head_dim);
            let width = n_kv * hd;
            if wq.1 != n_head * hd * 2 || wq.2 != cfg.n_embd {
                return Err(format!(
                    "attn_q is {}x{}, expected {}x{}",
                    wq.1,
                    wq.2,
                    n_head * hd * 2,
                    cfg.n_embd
                ));
            }
            for (name, m, rows, cols) in [
                ("attn_k", wk, n_kv * hd, cfg.n_embd),
                ("attn_v", wv, n_kv * hd, cfg.n_embd),
                ("attn_output", wo, cfg.n_embd, n_head * hd),
            ] {
                if m.1 != rows || m.2 != cols {
                    return Err(format!("{name} is {}x{}, expected {rows}x{cols}", m.1, m.2));
                }
            }
            let dims = LayerDims {
                n_head,
                n_kv,
                head_dim: hd,
                n_rot: cfg.n_rot,
                sections: cfg.rope_sections,
                eps: cfg.eps,
                rope_base: cfg.rope_base,
                width,
                n_ctx,
                n_embd: cfg.n_embd,
            };
            let _ = l;
            Ok(AttnLayer {
                wq: ResidentMat::borrow(wq.0, wq.1, wq.2, wq.3),
                wk: ResidentMat::borrow(wk.0, wk.1, wk.2, wk.3),
                wv: ResidentMat::borrow(wv.0, wv.1, wv.2, wv.3),
                wo: ResidentMat::borrow(wo.0, wo.1, wo.2, wo.3),
                q_norm: Buffer::from_host(q_norm)?,
                k_norm: Buffer::from_host(k_norm)?,
                qg: Buffer::alloc(2 * n_head * hd)?,
                q: Buffer::alloc(n_head * hd)?,
                gate: Buffer::alloc(n_head * hd)?,
                k: Buffer::alloc(width)?,
                v: Buffer::alloc(width)?,
                out: Buffer::alloc(n_head * hd)?,
                y: Buffer::alloc(cfg.n_embd)?,
                kv_k: Buffer::alloc(n_ctx * width)?,
                kv_v: Buffer::alloc(n_ctx * width)?,
                part_acc: Buffer::alloc(n_head * MAX_SPLITS * hd)?,
                part_ml: Buffer::alloc(n_head * MAX_SPLITS * 2)?,
                dims,
            })
        }

        /// The layer output buffer (`attn_output` + residual is the caller's).
        pub fn output(&self) -> &Buffer {
            &self.y
        }

        /// Bytes the KV cache occupies in VRAM.
        pub fn kv_bytes(&self) -> usize {
            2 * self.dims.n_ctx * self.dims.width * 4
        }

        /// Bytes of the four projected weights, as streamed per token.
        pub fn weight_bytes(&self) -> usize {
            self.wq.data.len() + self.wk.data.len() + self.wv.data.len() + self.wo.data.len()
        }

        pub fn zero_kv(&self) -> Result<(), String> {
            self.kv_k.zero()?;
            self.kv_v.zero()
        }

        /// Run q/k RMS norm and M-RoPE alone (the standalone parity target).
        pub fn norm_rope(
            &self,
            l: &Launcher,
            q_raw: &[f32],
            k_raw: &[f32],
            pos: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), String> {
            self.q.write(q_raw)?;
            self.k.write(k_raw)?;
            self.norm_and_rope(l, pos)?;
            Ok((self.q.to_host()?, self.k.to_host()?))
        }

        fn norm_and_rope(&self, l: &Launcher, pos: usize) -> Result<(), String> {
            self.norm_rope_at(l, self.q.as_ptr(), self.k.as_ptr(), pos)
        }

        /// Head RMS norm and M-RoPE on explicit q/k pointers, so the prefill
        /// path can run the same kernels on row `t` of a chunk buffer.
        fn norm_rope_at(
            &self,
            l: &Launcher,
            q: *mut c_void,
            k: *mut c_void,
            pos: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            let norm = self.module_kernel(l, "head_rmsnorm_f32")?;
            let mut hd = d.head_dim as i32;
            let mut eps = d.eps;

            // q heads
            let mut p_x = q;
            let mut p_w = self.q_norm.as_ptr();
            let mut p_y = q;
            let mut nh = d.n_head as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ];
            norm.launch(LaunchConfig::per_row(d.n_head, 256), &l.stream, &mut params)?;

            // k heads
            let mut p_x = k;
            let mut p_w = self.k_norm.as_ptr();
            let mut p_y = k;
            let mut nkv = d.n_kv as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut nkv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ];
            norm.launch(LaunchConfig::per_row(d.n_kv, 256), &l.stream, &mut params)?;

            self.rope_launch(l, q, d.n_head, pos)?;
            self.rope_launch(l, k, d.n_kv, pos)
        }

        fn rope_launch(
            &self,
            l: &Launcher,
            x: *mut c_void,
            n_heads: usize,
            pos: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            let rope = self.module_kernel(l, "mrope_f32")?;
            let mut p_in = x;
            let mut p_out = x;
            let mut nh = n_heads as i32;
            let mut hd = d.head_dim as i32;
            let mut nrot = d.n_rot as i32;
            let mut s0 = d.sections[0] as i32;
            let mut s1 = d.sections[1] as i32;
            let mut s2 = d.sections[2] as i32;
            let mut s3 = d.sections[3] as i32;
            let mut p0 = pos as i32;
            let mut base = d.rope_base;
            let mut params: [*mut c_void; 14] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut s0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut p0 as *mut _ as *mut c_void,
                &mut p0 as *mut _ as *mut c_void,
                &mut p0 as *mut _ as *mut c_void,
                &mut p0 as *mut _ as *mut c_void,
                &mut base as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (n_heads as u32, 1, 1),
                block: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            rope.launch(cfg, &l.stream, &mut params)
        }

        /// One layer for one token at `pos`. `x` is already the attention input
        /// (after `attn_norm`). Downloads the layer output.
        pub fn forward(&self, l: &Launcher, x: &[f32], pos: usize) -> Result<Vec<f32>, String> {
            let xb = Buffer::from_host(x)?;
            self.forward_buf(l, &xb, pos)?;
            self.y.to_host()
        }

        /// Q/K/V projections, q-gate split, head norm and rope. Leaves `q` and
        /// the K/V rows in their buffers; attention and the cache append are the
        /// caller's choice.
        pub fn prepare_qkv(&self, l: &Launcher, x: &Buffer, pos: usize) -> Result<(), String> {
            let d = self.dims;
            self.gemv(l, &self.wq, x, &self.qg, 2 * d.n_head * d.head_dim)?;
            self.gemv(l, &self.wk, x, &self.k, d.width)?;
            self.gemv(l, &self.wv, x, &self.v, d.width)?;

            let split = self.module_kernel(l, "qg_split_f32")?;
            let mut p_qg = self.qg.as_ptr();
            let mut p_q = self.q.as_ptr();
            let mut p_gate = self.gate.as_ptr();
            let mut nh = d.n_head as i32;
            let mut hd = d.head_dim as i32;
            let mut params: [*mut c_void; 5] = [
                &mut p_qg as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_gate as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_head as u32, 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            split.launch(cfg, &l.stream, &mut params)?;
            self.norm_and_rope(l, pos)
        }

        /// `out = W_o (attn_out * gate)` — the tail shared by both KV paths.
        pub fn apply_output(&self, l: &Launcher) -> Result<(), String> {
            let d = self.dims;
            let gate = self.module_kernel(l, "q_gate_f32")?;
            let mut p_out = self.out.as_ptr();
            let mut p_gate = self.gate.as_ptr();
            let mut p_y = self.out.as_ptr();
            let mut n = (d.n_head * d.head_dim) as i32;
            let mut params: [*mut c_void; 4] = [
                &mut p_out as *mut _ as *mut c_void,
                &mut p_gate as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
            ];
            gate.launch(
                LaunchConfig::linear(d.n_head * d.head_dim, 256),
                &l.stream,
                &mut params,
            )?;
            self.gemv(l, &self.wo, &self.out, &self.y, d.n_embd)
        }

        /// No-attention accessors for the quantized-KV path.
        pub fn q_buf(&self) -> &Buffer {
            &self.q
        }
        pub fn k_buf(&self) -> &Buffer {
            &self.k
        }
        pub fn v_buf(&self) -> &Buffer {
            &self.v
        }
        pub fn attn_out(&self) -> &Buffer {
            &self.out
        }

        /// The position-major KV cache this layer appends to, `[n_ctx][width]`.
        pub fn kv_k_buf(&self) -> &Buffer {
            &self.kv_k
        }

        pub fn kv_v_buf(&self) -> &Buffer {
            &self.kv_v
        }

        /// Same, with `x` resident on the device. Does not synchronize.
        pub fn forward_buf(&self, l: &Launcher, x: &Buffer, pos: usize) -> Result<(), String> {
            let d = self.dims;
            if pos >= d.n_ctx {
                return Err(format!("position {pos} beyond KV cache {}", d.n_ctx));
            }
            self.prepare_qkv(l, x, pos)?;

            let append = self.module_kernel(l, "kv_append_f32")?;
            let mut p_kc = self.kv_k.as_ptr();
            let mut p_vc = self.kv_v.as_ptr();
            let mut p_k = self.k.as_ptr();
            let mut p_v = self.v.as_ptr();
            let mut off = pos as i32;
            let mut width = d.width as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_kc as *mut _ as *mut c_void,
                &mut p_vc as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut off as *mut _ as *mut c_void,
                &mut width as *mut _ as *mut c_void,
            ];
            append.launch(
                LaunchConfig::linear(2 * d.width, 256),
                &l.stream,
                &mut params,
            )?;

            self.attention(l, pos)?;
            self.apply_output(l)
        }

        /// Causal attention for this token. Uses the split-K kernel when the
        /// head shape allows it, otherwise the one-block-per-head kernel.
        fn attention(&self, l: &Launcher, pos: usize) -> Result<(), String> {
            self.attention_at(l, self.q.as_ptr(), self.out.as_ptr(), pos + 1)
        }

        /// Same launch over explicit q/out rows and a given sequence length, so
        /// a prefill chunk can attend from row `t` at `seq = pos + 1`.
        fn attention_at(
            &self,
            l: &Launcher,
            q: *mut c_void,
            out: *mut c_void,
            seq: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            let group = d.n_head / d.n_kv;
            let dpl = d.head_dim / 32;
            let shared = 2 * SPLIT_TILE * d.head_dim * 4;
            let split_ok = d.head_dim % 32 == 0
                && dpl <= 8
                && group * 32 <= 1024
                && shared <= 64 * 1024;

            if split_ok {
                let splits = (seq / 128).clamp(1, MAX_SPLITS);
                let mut p_q = q;
                let mut p_k = self.kv_k.as_ptr();
                let mut p_v = self.kv_v.as_ptr();
                let mut p_acc = self.part_acc.as_ptr();
                let mut p_ml = self.part_ml.as_ptr();
                let mut n_heads = d.n_head as i32;
                let mut n_kv = d.n_kv as i32;
                let mut hd = d.head_dim as i32;
                let mut v_seq = seq as i32;
                let mut scale = 1.0f32 / (d.head_dim as f32).sqrt();
                let mut v_splits = splits as i32;
                let mut width = d.width as i32;
                let mut params: [*mut c_void; 12] = [
                    &mut p_q as *mut _ as *mut c_void,
                    &mut p_k as *mut _ as *mut c_void,
                    &mut p_v as *mut _ as *mut c_void,
                    &mut p_acc as *mut _ as *mut c_void,
                    &mut p_ml as *mut _ as *mut c_void,
                    &mut n_heads as *mut _ as *mut c_void,
                    &mut n_kv as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut v_seq as *mut _ as *mut c_void,
                    &mut scale as *mut _ as *mut c_void,
                    &mut v_splits as *mut _ as *mut c_void,
                    &mut width as *mut _ as *mut c_void,
                ];
                let cfg = LaunchConfig {
                    grid: ((d.n_kv * splits) as u32, 1, 1),
                    block: ((group * 32) as u32, 1, 1),
                    shared_mem_bytes: shared as u32,
                };
                self.module_kernel(l, "attn_decode_split_f32")?
                    .launch(cfg, &l.stream, &mut params)?;

                let reduce = self.module_kernel(l, "attn_reduce_f32")?;
                let mut p_acc = self.part_acc.as_ptr();
                let mut p_ml = self.part_ml.as_ptr();
                let mut p_out = out;
                let mut params: [*mut c_void; 5] = [
                    &mut p_acc as *mut _ as *mut c_void,
                    &mut p_ml as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                    &mut hd as *mut _ as *mut c_void,
                    &mut v_splits as *mut _ as *mut c_void,
                ];
                let cfg = LaunchConfig {
                    grid: (d.n_head as u32, 1, 1),
                    block: (d.head_dim as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                return reduce.launch(cfg, &l.stream, &mut params);
            }

            let attn = self.module_kernel(l, "attn_decode_f32")?;
            let mut p_q = q;
            let mut p_k = self.kv_k.as_ptr();
            let mut p_v = self.kv_v.as_ptr();
            let mut p_out = out;
            let mut n_heads = d.n_head as i32;
            let mut n_kv = d.n_kv as i32;
            let mut hd = d.head_dim as i32;
            let mut v_seq = seq as i32;
            let mut scale = 1.0f32 / (d.head_dim as f32).sqrt();
            let mut head_stride = d.head_dim as i32;
            let mut row_stride = d.width as i32;
            let mut params: [*mut c_void; 11] = [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut n_heads as *mut _ as *mut c_void,
                &mut n_kv as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut v_seq as *mut _ as *mut c_void,
                &mut scale as *mut _ as *mut c_void,
                &mut head_stride as *mut _ as *mut c_void,
                &mut row_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_head as u32, 1, 1),
                block: (d.head_dim as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            attn.launch(cfg, &l.stream, &mut params)
        }

        fn module_kernel<'b>(
            &self,
            l: &'b Launcher,
            name: &str,
        ) -> Result<hipml_hip::Kernel<'b>, String> {
            l.module.kernel(name)
        }

        fn gemv(
            &self,
            l: &Launcher,
            w: &ResidentMat,
            x: &Buffer,
            y: &Buffer,
            m: usize,
        ) -> Result<(), String> {
            if w.rows != m {
                return Err(format!("gemv: {} rows, launch says {m}", w.rows));
            }
            let entry = w.entry()?;
            let spec = crate::spec(entry).ok_or_else(|| format!("{entry} not in manifest"))?;
            let kernel = l.module.kernel(entry)?;
            let mut p_w = w.data.as_ptr();
            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_m = m as i32;
            let mut v_k = w.cols as i32;
            let mut params: [*mut c_void; 5] = [
                &mut p_w as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_m as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
            ];
            let cfg = match spec.kind {
                crate::KernelKind::RowTile { rows, block } => {
                    LaunchConfig::row_tile(m, rows, block)
                }
                crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
                _ => return Err(format!("{entry}: unsupported grid")),
            };
            kernel.launch(cfg, &l.stream, &mut params)
        }
    }

}

#[cfg(not(feature = "hip"))]
pub struct AttnLayer;

#[cfg(not(feature = "hip"))]
impl AttnLayer {
    pub fn new(
        _l: &crate::Launcher,
        _w: &hipml_core::attention::FullAttnW,
        _cfg: &hipml_core::config::Config,
        _n_ctx: usize,
    ) -> Result<AttnLayer, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn forward(
        &self,
        _l: &crate::Launcher,
        _x: &[f32],
        _pos: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn forward_buf(
        &self,
        _l: &crate::Launcher,
        _x: &hipml_hip::Buffer,
        _pos: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn norm_rope(
        &self,
        _l: &crate::Launcher,
        _q: &[f32],
        _k: &[f32],
        _pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn prepare_qkv(
        &self,
        _l: &crate::Launcher,
        _x: &hipml_hip::Buffer,
        _pos: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn apply_output(&self, _l: &crate::Launcher) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn q_buf(&self) -> &hipml_hip::Buffer {
        unreachable!()
    }

    pub fn k_buf(&self) -> &hipml_hip::Buffer {
        unreachable!()
    }

    pub fn v_buf(&self) -> &hipml_hip::Buffer {
        unreachable!()
    }

    pub fn attn_out(&self) -> &hipml_hip::Buffer {
        unreachable!()
    }

    pub fn zero_kv(&self) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn kv_bytes(&self) -> usize {
        0
    }

    pub fn weight_bytes(&self) -> usize {
        0
    }
}
