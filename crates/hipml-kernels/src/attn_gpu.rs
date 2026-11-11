// SPDX-License-Identifier: Apache-2.0
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

    /// Device pointer of row `row` of a `[rows][width]` f32 buffer.
    fn row_ptr(buf: &Buffer, row: usize, width: usize) -> *mut c_void {
        unsafe { (buf.as_ptr() as *mut u8).add(row * width * 4) as *mut c_void }
    }

    /// Upper bound on sequence splits; the partial buffers are sized for it.
    const MAX_SPLITS: usize = 64;
    /// Must match SPLIT_TILE in attn_decode_split_f32.hip.
    const SPLIT_TILE: usize = 8;

    /// The prefill-chunk launches (child so they see the layer's private
    /// scratch and weight handles).
    #[path = "attn_chunk.rs"]
    mod chunk;
    #[path = "attn_chunk_kv.rs"]
    mod chunk_kv;
    #[path = "attn_cols.rs"]
    mod cols;
    #[path = "attn_rows.rs"]
    mod rows;

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
            Ok(crate::stream_entry(match self.kind {
                Kind::F32 => "gemv_f32",
                Kind::Q4K => "gemv_q4_k",
                Kind::Q6K => "gemv_q6_k",
                Kind::Q8_0 => "gemv_q8_0",
                // Unsloth's UD dynamic quants store some attention projections
                // as Q5_K / IQ4; the GEMV entries read those blocks.
                Kind::Q5K => "gemv_q5_k",
                Kind::IQ4XS => "gemv_iq4_xs",
                Kind::IQ4NL => "gemv_iq4_nl",
                Kind::Q3K | Kind::IQ3S => {
                    return Err(format!(
                        "attention weight {:?} is expanded to F32 at load",
                        self.kind
                    ))
                }
            }))
        }

        fn data_bytes(&self) -> &ByteBuffer {
            self.data.bytes()
        }

        /// The tiled prefill GEMM for the same payload (`tokens` rows of x).
        fn prefill_entry(&self) -> Result<&'static str, String> {
            Ok(match self.kind {
                Kind::Q4K => "prefill_q4_k",
                Kind::Q6K => "prefill_q6_k",
                Kind::Q5K => "prefill_q5_k",
                Kind::Q8_0 => "prefill_q8_0",
                Kind::IQ4XS => "prefill_iq4_xs",
                Kind::IQ4NL => "prefill_iq4_nl",
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
            l.set_step(pos, pos + 1, 0)?;
            self.norm_and_rope(l)?;
            Ok((self.q.to_host()?, self.k.to_host()?))
        }

        fn norm_and_rope(&self, l: &Launcher) -> Result<(), String> {
            self.norm_rope_at(l, self.q.as_ptr(), self.k.as_ptr())
        }

        /// Head RMS norm and M-RoPE on explicit q/k pointers, so the prefill
        /// path can run the same kernels on row `t` of a chunk buffer.
        pub(crate) fn norm_rope_at(
            &self,
            l: &Launcher,
            q: *mut c_void,
            k: *mut c_void,
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

            self.rope_launch(l, q, d.n_head)?;
            self.rope_launch(l, k, d.n_kv)
        }

        fn rope_launch(&self, l: &Launcher, x: *mut c_void, n_heads: usize) -> Result<(), String> {
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
            let mut base = d.rope_base;
            // All four position ids are the token position for text; the array
            // lives in the step buffer so the launch survives capture.
            let mut p_pos = l.mrope_pos();
            let mut params: [*mut c_void; 11] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut nrot as *mut _ as *mut c_void,
                &mut s0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut p_pos as *mut _ as *mut c_void,
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
            // `forward_buf` reads the position and the sequence length from the
            // step buffer; this host entry point is the one that owns them.
            // (The engine sets the step itself, with the token id, so the
            // capture-safe `forward_buf` stays free of the upload.)
            l.set_step(pos, pos + 1, 0)?;
            self.forward_buf(l, &xb, pos)?;
            self.y.to_host()
        }

        /// Q/K/V projections, q-gate split, head norm and rope. Leaves `q` and
        /// the K/V rows in their buffers; attention and the cache append are the
        /// caller's choice.
        pub fn prepare_qkv(&self, l: &Launcher, x: &Buffer) -> Result<(), String> {
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
            self.norm_and_rope(l)
        }

        /// Batched Q/K/V projections for `rows` x-vectors: one pass over each
        /// weight. `qg` is `[rows][2*n_head*head_dim]`, `k`/`v` are
        /// `[rows][width]`. The split, head norm and rope are per row because
        /// every row has its own position (see `split_rope_rows`).
        pub fn qkv_rows(
            &self,
            l: &Launcher,
            x: &Buffer,
            qg: &Buffer,
            k: &Buffer,
            v: &Buffer,
            rows: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            self.gemv_batched(l, &self.wq, x, qg, 2 * d.n_head * d.head_dim, rows)?;
            self.gemv_batched(l, &self.wk, x, k, d.width, rows)?;
            self.gemv_batched(l, &self.wv, x, v, d.width, rows)
        }

        /// Per-row q-gate split, head RMS norm and M-RoPE at each row's own
        /// position. `positions[b]` is row `b`'s token position.
        #[allow(clippy::too_many_arguments)]
        pub fn split_rope_rows(
            &self,
            l: &Launcher,
            qg: &Buffer,
            k: &Buffer,
            q: &Buffer,
            gate: &Buffer,
            positions: &[usize],
            pos_buf: &Buffer,
        ) -> Result<(), String> {
            let d = self.dims;
            let qw = d.n_head * d.head_dim;
            let rows = positions.len();
            // The device `pos_buf` is `[rows][2]` (position, seq), written once
            // per step; these launches read it and are slot-indexed on grid.y.
            let split = self.module_kernel(l, "qg_split_b")?;
            let mut p_qg = qg.as_ptr();
            let mut p_q = q.as_ptr();
            let mut p_gate = gate.as_ptr();
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
                grid: (d.n_head as u32, rows as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            split.launch(cfg, &l.stream, &mut params)?;

            // Head RMS norm: q then k.
            let norm = self.module_kernel(l, "head_rmsnorm_b")?;
            let mut eps = d.eps;
            let mut p_x = q.as_ptr();
            let mut p_w = self.q_norm.as_ptr();
            let mut p_y = q.as_ptr();
            let mut v_nh = d.n_head as i32;
            let mut v_stride = qw as i32;
            let mut params: [*mut c_void; 7] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_head as u32, rows as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            norm.launch(cfg, &l.stream, &mut params)?;

            let mut p_x = k.as_ptr();
            let mut p_w = self.k_norm.as_ptr();
            let mut p_y = k.as_ptr();
            let mut v_nh = d.n_kv as i32;
            let mut v_stride = d.width as i32;
            let mut params: [*mut c_void; 7] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
                &mut v_stride as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_kv as u32, rows as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            norm.launch(cfg, &l.stream, &mut params)?;

            // M-RoPE at each row's own position.
            let rope = self.module_kernel(l, "mrope_b")?;
            let mut p_pos = pos_buf.as_ptr();
            let mut v_nrot = d.n_rot as i32;
            let mut s0 = d.sections[0] as i32;
            let mut s1 = d.sections[1] as i32;
            let mut s2 = d.sections[2] as i32;
            let mut s3 = d.sections[3] as i32;
            let mut base = d.rope_base;
            let mut p_in = q.as_ptr();
            let mut p_out = q.as_ptr();
            let mut v_nh = d.n_head as i32;
            let mut hd = d.head_dim as i32;
            let mut params: [*mut c_void; 11] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut v_nrot as *mut _ as *mut c_void,
                &mut s0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut p_pos as *mut _ as *mut c_void,
                &mut base as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_head as u32, rows as u32, 1),
                block: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            rope.launch(cfg, &l.stream, &mut params)?;

            let mut p_in = k.as_ptr();
            let mut p_out = k.as_ptr();
            let mut v_nh = d.n_kv as i32;
            let mut params: [*mut c_void; 11] = [
                &mut p_in as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut hd as *mut _ as *mut c_void,
                &mut v_nrot as *mut _ as *mut c_void,
                &mut s0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut p_pos as *mut _ as *mut c_void,
                &mut base as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (d.n_kv as u32, rows as u32, 1),
                block: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            rope.launch(cfg, &l.stream, &mut params)
        }

        /// Batched attention-output projection: gate each row of `out` and run
        /// one pass over `W_o` into `y` (`[rows][n_embd]`).
        pub fn out_rows(
            &self,
            l: &Launcher,
            out: &Buffer,
            gate: &Buffer,
            y: &Buffer,
            rows: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            let qw = d.n_head * d.head_dim;
            let gate_k = self.module_kernel(l, "q_gate_f32")?;
            // `out` and `gate` are `[rows][qw]` row-major and the op is pure
            // elementwise, so one launch covers every row.
            let mut n = (rows * qw) as i32;
            let mut p_out = out.as_ptr();
            let mut p_gate = gate.as_ptr();
            let mut p_y = p_out;
            let mut params: [*mut c_void; 4] = [
                &mut p_out as *mut _ as *mut c_void,
                &mut p_gate as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
            ];
            gate_k.launch(LaunchConfig::linear(rows * qw, 256), &l.stream, &mut params)?;
            self.gemv_batched(l, &self.wo, out, y, d.n_embd, rows)
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
            self.prepare_qkv(l, x)?;
            self.append_at(l, self.k.as_ptr(), self.v.as_ptr())?;
            self.attention(l, pos)?;
            self.apply_output(l)
        }

        /// Append explicit K/V rows at `step[0]` (the position lives in the
        /// device step buffer, as in `forward_buf`).
        pub fn append_at(
            &self,
            l: &Launcher,
            k: *mut c_void,
            v: *mut c_void,
        ) -> Result<(), String> {
            let d = self.dims;
            let append = self.module_kernel(l, "kv_append_f32")?;
            let mut p_kc = self.kv_k.as_ptr();
            let mut p_vc = self.kv_v.as_ptr();
            let mut p_k = k;
            let mut p_v = v;
            let mut p_step = l.step_buf().as_ptr();
            let mut width = d.width as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_kc as *mut _ as *mut c_void,
                &mut p_vc as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_step as *mut _ as *mut c_void,
                &mut width as *mut _ as *mut c_void,
            ];
            append.launch(
                LaunchConfig::linear(2 * d.width, 256),
                &l.stream,
                &mut params,
            )
        }

        /// The K/V projections, k norm+rope and cache append for one row, with
        /// no query: the MTP drafter's prompt prefill only needs the KV rows.
        pub fn append_kv_only(&self, l: &Launcher, x: &Buffer, pos: usize) -> Result<(), String> {
            let d = self.dims;
            if pos >= d.n_ctx {
                return Err(format!("position {pos} beyond KV cache {}", d.n_ctx));
            }
            self.gemv(l, &self.wk, x, &self.k, d.width)?;
            self.gemv(l, &self.wv, x, &self.v, d.width)?;
            self.norm_rope_at(l, self.q.as_ptr(), self.k.as_ptr())?;
            self.append_at(l, self.k.as_ptr(), self.v.as_ptr())
        }

        /// Batched Q/K/V for `rows` columns of `x`, then the per-row split,
        /// head norm and rope. `pos0 + c` is row `c`'s absolute position.
        ///
        /// One weight pass for all rows: the `_b` GEMVs are bit-identical per
        /// row to the single-sequence GEMV, so the verify columns match a plain
        /// decode.
        #[allow(clippy::too_many_arguments)]
        pub fn prepare_rows(
            &self,
            l: &Launcher,
            x: &Buffer,
            rows: usize,
            qg: &Buffer,
            q: &Buffer,
            gate: &Buffer,
            k: &Buffer,
            v: &Buffer,
            pos0: usize,
        ) -> Result<(), String> {
            self.gemv_rows(l, &self.wq, x, qg, rows)?;
            self.gemv_rows(l, &self.wk, x, k, rows)?;
            self.gemv_rows(l, &self.wv, x, v, rows)?;
            self.split_norm_rope(l, rows, qg, q, gate, k, pos0)
        }

        /// Append row `c`'s K/V at `pos` and run this row's causal attention.
        /// `q_row`/`out_row` are device pointers to the batch's rows.
        pub fn attend_row_at(
            &self,
            l: &Launcher,
            q_row: *mut c_void,
            out_row: *mut c_void,
            pos: usize,
        ) -> Result<(), String> {
            self.attention_at(l, q_row, out_row, pos)
        }

        /// `y = Wo (aout * sigmoid(gate))` for `rows` rows: the gate in one
        /// elementwise pass (the rows are contiguous), then one batched weight
        /// pass for the output projection.
        pub fn output_rows(
            &self,
            l: &Launcher,
            aout: &Buffer,
            gate: &Buffer,
            y: &Buffer,
            rows: usize,
        ) -> Result<(), String> {
            let d = self.dims;
            let qgate = self.module_kernel(l, "q_gate_f32")?;
            let n = d.n_head * d.head_dim;
            let all = n * rows;
            let mut p_out = aout.as_ptr();
            let mut p_gate = gate.as_ptr();
            let mut p_y = aout.as_ptr();
            let mut v_n = all as i32;
            let mut params: [*mut c_void; 4] = [
                &mut p_out as *mut _ as *mut c_void,
                &mut p_gate as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
            ];
            qgate.launch(LaunchConfig::linear(all, 256), &l.stream, &mut params)?;
            self.gemv_rows(l, &self.wo, aout, y, rows)
        }

        /// One projection over `rows` x-rows with a single weight pass, or
        /// per-row GEMVs when the format has no `_b` kernel (F32 weights).
        fn gemv_rows(
            &self,
            l: &Launcher,
            w: &ResidentMat,
            x: &Buffer,
            y: &Buffer,
            rows: usize,
        ) -> Result<(), String> {
            if crate::skinny_on() {
                if let Some(name) = crate::skinny_entry(w.entry()?) {
                    return l.run_skinny_resident(name, w.data_bytes(), x, y, w.rows, w.cols, rows);
                }
            }
            if let Some(kind) = crate::batch_kind_for(w.kind) {
                return l.gemv_batched(kind, w.data_bytes(), x, y, w.rows, w.cols, rows);
            }
            for c in 0..rows {
                let entry = w.entry()?;
                l.run_quant_resident_ptr(
                    entry,
                    w.data_bytes(),
                    row_ptr(x, c, w.cols),
                    row_ptr(y, c, w.rows),
                    w.rows,
                    w.cols,
                )?;
            }
            Ok(())
        }

        /// One projection over a prefill tile whose block format has no tiled
        /// GEMM: the `_b` GEMV, one weight pass per 128 rows.
        fn proj_rows(
            &self,
            l: &Launcher,
            w: &ResidentMat,
            x: &Buffer,
            y: &Buffer,
            rows: usize,
        ) -> Result<(), String> {
            let kind = crate::batch_kind_for(w.kind)
                .ok_or_else(|| format!("attention weight {:?}: no batched GEMV", w.kind))?;
            l.gemv_batched_rows(kind, w.data.bytes(), x, y, w.rows, w.cols, rows)
        }

        /// Causal attention for this token. Uses the split-K kernel when the
        /// head shape allows it, otherwise the one-block-per-head kernel.
        fn attention(&self, l: &Launcher, pos: usize) -> Result<(), String> {
            self.attention_at(l, self.q.as_ptr(), self.out.as_ptr(), pos)
        }

        /// Same launch over explicit q/out rows, at absolute position `pos`.
        /// One column: the verify pass calls `attention_cols` directly.
        pub fn attention_at(
            &self,
            l: &Launcher,
            q: *mut c_void,
            out: *mut c_void,
            pos: usize,
        ) -> Result<(), String> {
            self.attention_cols(l, q, out, &self.part_acc, &self.part_ml, pos, 1)
        }

        fn module_kernel<'b>(
            &self,
            l: &'b Launcher,
            name: &str,
        ) -> Result<hipml_hip::Kernel<'b>, String> {
            l.module.kernel(name)
        }

        fn gemv_batched(
            &self,
            l: &Launcher,
            w: &ResidentMat,
            x: &Buffer,
            y: &Buffer,
            m: usize,
            rows: usize,
        ) -> Result<(), String> {
            if w.rows != m {
                return Err(format!("gemv_batched: {} rows, launch says {m}", w.rows));
            }
            let kind = crate::batch_kind_for(w.kind)
                .ok_or_else(|| format!("attention weight {:?}: no batched GEMV", w.kind))?;
            l.gemv_batched(kind, w.data.bytes(), x, y, m, w.cols, rows)
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

    pub fn prepare_qkv(&self, _l: &crate::Launcher, _x: &hipml_hip::Buffer) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn apply_output(&self, _l: &crate::Launcher) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention(
        &self,
        _l: &crate::Launcher,
        _q: &hipml_hip::Buffer,
        _kc: &hipml_hip::Buffer,
        _vc: &hipml_hip::Buffer,
        _out: &hipml_hip::Buffer,
        _pos0: usize,
        _tokens: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chunk_front(
        &self,
        _l: &crate::Launcher,
        _qg: &hipml_hip::Buffer,
        _k: &hipml_hip::Buffer,
        _q: &hipml_hip::Buffer,
        _gate: &hipml_hip::Buffer,
        _pos0: usize,
        _tokens: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn q8_chunk_supported(&self) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention_q8(
        &self,
        _l: &crate::Launcher,
        _q: &hipml_hip::Buffer,
        _kc: &hipml_hip::Buffer,
        _vc: &hipml_hip::Buffer,
        _out: &hipml_hip::Buffer,
        _pos0: usize,
        _tokens: usize,
        _dtype: usize,
        _stride: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chunk_attention_q8_split(
        &self,
        _l: &crate::Launcher,
        _q: &hipml_hip::Buffer,
        _kc: &hipml_hip::Buffer,
        _vc: &hipml_hip::Buffer,
        _out: &hipml_hip::Buffer,
        _pos0: usize,
        _tokens: usize,
        _dtype: usize,
        _stride: usize,
        _part_acc: &hipml_hip::Buffer,
        _part_ml: &hipml_hip::Buffer,
        _splits: usize,
    ) -> Result<(), String> {
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

    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rows(
        &self,
        _l: &crate::Launcher,
        _x: &hipml_hip::Buffer,
        _qg: &hipml_hip::Buffer,
        _k: &hipml_hip::Buffer,
        _v: &hipml_hip::Buffer,
        _rows: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn split_rope_rows(
        &self,
        _l: &crate::Launcher,
        _qg: &hipml_hip::Buffer,
        _k: &hipml_hip::Buffer,
        _q: &hipml_hip::Buffer,
        _gate: &hipml_hip::Buffer,
        _positions: &[usize],
        _pos_buf: &hipml_hip::Buffer,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn out_rows(
        &self,
        _l: &crate::Launcher,
        _out: &hipml_hip::Buffer,
        _gate: &hipml_hip::Buffer,
        _y: &hipml_hip::Buffer,
        _rows: usize,
    ) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
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
