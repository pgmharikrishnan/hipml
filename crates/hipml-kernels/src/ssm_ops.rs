// SPDX-License-Identifier: Apache-2.0
//! Kernel launches for one GDN layer, split from `ssm.rs` for size.
//!
//! Every launch writes its arguments out explicitly, matching the existing
//! launchers: `hipModuleLaunchKernel` takes pointers to the arguments, so each
//! value needs a stable home in a local.
//!
//! The `*_at` forms take raw pointers so the prefill chunk path
//! (`ssm_chunk.rs`) can run the same kernels on row `t` of a chunk buffer. The
//! single-token `step` chain is these same launches over the layer's own
//! scratch.

use crate::ssm::{SsmGpuLayer, SsmWeight};
use hipml_hip::{Buffer, LaunchConfig};
use crate::Launcher;
use std::ffi::c_void;
use std::time::Instant;

impl<'a> SsmGpuLayer<'a> {
    /// The four input projections.
    pub fn gemvs(&self) -> Result<(), String> {
        let d = &self.dims;
        self.gemv(&self.wqkv, &self.x, &self.qkv, d.conv_ch, d.n_embd)?;
        self.gemv(&self.wgate, &self.x, &self.z, d.inner(), d.n_embd)?;
        self.gemv(&self.alpha_w, &self.x, &self.alpha_raw, d.n_v, d.n_embd)?;
        self.gemv(&self.beta_w, &self.x, &self.beta_raw, d.n_v, d.n_embd)?;
        Ok(())
    }

    /// The four input projections for `rows` columns, one weight pass each.
    /// `x`, `qkv`, `z`, `alpha` and `beta` are `[rows][..]` buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn project_rows(
        &self,
        l: &crate::Launcher,
        x: &Buffer,
        rows: usize,
        qkv: &Buffer,
        z: &Buffer,
        alpha: &Buffer,
        beta: &Buffer,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.gemv_rows(l, &self.wqkv, x, qkv, d.conv_ch, d.n_embd, rows)?;
        self.gemv_rows(l, &self.wgate, x, z, d.inner(), d.n_embd, rows)?;
        self.gemv_rows(l, &self.alpha_w, x, alpha, d.n_v, d.n_embd, rows)?;
        self.gemv_rows(l, &self.beta_w, x, beta, d.n_v, d.n_embd, rows)
    }

    /// The output projection for `rows` columns.
    pub fn out_rows(
        &self,
        l: &crate::Launcher,
        gated: &Buffer,
        out: &Buffer,
        rows: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.gemv_rows(l, &self.wout, gated, out, d.n_embd, d.inner(), rows)
    }

    /// One projection over `rows` x-rows: the `_b` kernel when the format has
    /// one, otherwise a GEMV per row (f32 alpha/beta are tiny).
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows(
        &self,
        l: &crate::Launcher,
        w: &SsmWeight<'_>,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        rows: usize,
    ) -> Result<(), String> {
        match w {
            SsmWeight::Quant(b, entry) => {
                if crate::skinny_on() {
                    if let Some(name) = crate::skinny_entry(entry) {
                        return l.run_skinny_resident(name, b, x, y, m, k, rows);
                    }
                }
                if let Some(kind) = crate::batch_kind_entry(entry) {
                    return l.gemv_batched(kind, b, x, y, m, k, rows);
                }
                for c in 0..rows {
                    l.run_quant_resident_ptr(
                        entry,
                        b,
                        offset(x.as_ptr(), c * k),
                        offset(y.as_ptr(), c * m),
                        m,
                        k,
                    )?;
                }
                Ok(())
            }
            SsmWeight::F32(b) => {
                for c in 0..rows {
                    let mut p_w = b.as_ptr();
                    let mut p_x = offset(x.as_ptr(), c * k);
                    let mut p_y = offset(y.as_ptr(), c * m);
                    let mut v_m = m as i32;
                    let mut v_k = k as i32;
                    let mut params: [*mut c_void; 5] = [
                        &mut p_w as *mut _ as *mut c_void,
                        &mut p_x as *mut _ as *mut c_void,
                        &mut p_y as *mut _ as *mut c_void,
                        &mut v_m as *mut _ as *mut c_void,
                        &mut v_k as *mut _ as *mut c_void,
                    ];
                    self.k_gemv
                        .launch(LaunchConfig::per_row(m, 256), self.stream, &mut params)?;
                }
                Ok(())
            }
        }
    }

    /// Everything except the projections: gates, conv, norm, recurrence,
    /// gated RMS norm. This is the compute the recurrent block adds on top of
    /// streaming the weights.
    pub fn recurrent(&self) -> Result<(), String> {
        self.gates()?;
        self.conv()?;
        self.l2norms()?;
        self.recurrence()?;
        self.gate_norm()
    }

    pub fn step_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.step())
    }

    pub fn gemv_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.gemvs())
    }

    pub fn recurrent_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.recurrent())
    }

    pub fn output_gemv_reps(&self, reps: usize) -> Result<f64, String> {
        self.time_reps(reps, || self.output_gemv())
    }

    fn time_reps(&self, reps: usize, f: impl Fn() -> Result<(), String>) -> Result<f64, String> {
        if reps == 0 {
            return Err("reps must be positive".to_string());
        }
        let start = Instant::now();
        for _ in 0..reps {
            f()?;
        }
        self.stream.synchronize()?;
        Ok(start.elapsed().as_secs_f64() / reps as f64)
    }

    fn gemv(
        &self,
        w: &SsmWeight<'_>,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
    ) -> Result<(), String> {
        // The argument array is the ABI and is the same for both paths; only the
        // weight pointer and the grid differ.
        let (mut p_w, entry) = match w {
            SsmWeight::F32(b) => (b.as_ptr(), None),
            SsmWeight::Quant(b, e) => (b.as_ptr(), Some(*e)),
        };
        let mut p_x = x.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        match entry {
            None => self
                .k_gemv
                .launch(LaunchConfig::per_row(m, 256), self.stream, &mut params),
            Some(entry) => {
                let spec = crate::spec(entry)
                    .ok_or_else(|| format!("{entry} is not in the manifest"))?;
                let cfg = match spec.kind {
                    crate::KernelKind::RowTile { rows, block } => {
                        LaunchConfig::row_tile(m, rows, block)
                    }
                    crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
                    _ => return Err(format!("{entry}: unsupported grid")),
                };
                self.module
                    .kernel(entry)?
                    .launch(cfg, self.stream, &mut params)
            }
        }
    }

    pub fn output_gemv(&self) -> Result<(), String> {
        let d = &self.dims;
        self.gemv(&self.wout, &self.gated, &self.out, d.n_embd, d.inner())
    }

    fn gates(&self) -> Result<(), String> {
        self.gates_at(
            self.alpha_raw.as_ptr(),
            self.beta_raw.as_ptr(),
            self.g.as_ptr(),
            self.beta_sig.as_ptr(),
        )
    }

    /// `g = softplus(alpha + dt_bias) * a`, `beta_sig = sigmoid(beta)`.
    pub fn gates_at(
        &self,
        alpha: *mut c_void,
        beta: *mut c_void,
        g: *mut c_void,
        beta_sig: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        let mut p_alpha = alpha;
        let mut p_beta = beta;
        let mut p_dt = self.dt_bias.as_ptr();
        let mut p_a = self.a.as_ptr();
        let mut p_g = g;
        let mut p_out = beta_sig;
        let mut v_n = d.n_v as i32;
        let mut params: [*mut c_void; 7] = [
            &mut p_alpha as *mut _ as *mut c_void,
            &mut p_beta as *mut _ as *mut c_void,
            &mut p_dt as *mut _ as *mut c_void,
            &mut p_a as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        self.k_gates
            .launch(LaunchConfig::linear(d.n_v, 64), self.stream, &mut params)
    }

    fn conv(&self) -> Result<(), String> {
        self.conv_at(self.qkv.as_ptr(), self.conv_out.as_ptr())
    }

    /// `conv_out = silu(conv1d(qkv, state))` in two launches, mirroring the
    /// reference's conv-then-activate pair.
    pub fn conv_at(&self, qkv: *mut c_void, conv_out: *mut c_void) -> Result<(), String> {
        let d = &self.dims;
        let mut p_x = qkv;
        let mut p_w = self.conv_w.as_ptr();
        let mut p_state = self.conv_state.as_ptr();
        let mut p_y = conv_out;
        let mut v_c = d.conv_ch as i32;
        let mut v_k = d.d_conv as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_state as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_c as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
        ];
        self.k_conv.launch(
            LaunchConfig::linear(d.conv_ch, 256),
            self.stream,
            &mut params,
        )?;
        self.silu_at(conv_out)
    }

    pub(crate) fn silu_at(&self, x: *mut c_void) -> Result<(), String> {
        let d = &self.dims;
        let mut p_x = x;
        let mut v_n = d.conv_ch as i32;
        let mut params: [*mut c_void; 2] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        self.k_silu
            .launch(LaunchConfig::linear(d.conv_ch, 256), self.stream, &mut params)
    }

    fn l2norms(&self) -> Result<(), String> {
        let d = &self.dims;
        let base = self.conv_out.as_ptr();
        self.l2norm_at_ptr(base, 0, d.n_k, d.head_k)?;
        self.l2norm_at_ptr(base, d.key_dim(), d.n_k, d.head_k)
    }

    /// In place: the kernel drops `__restrict__` on its two pointers.
    pub fn l2norm_at_ptr(
        &self,
        base: *mut c_void,
        off: usize,
        heads: usize,
        dim: usize,
    ) -> Result<(), String> {
        let mut p_x = offset(base, off);
        let mut p_y = p_x;
        let mut v_h = heads as i32;
        let mut v_d = dim as i32;
        let mut v_eps = self.dims.eps;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (heads as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_l2norm.launch(cfg, self.stream, &mut params)
    }

    fn recurrence(&self) -> Result<(), String> {
        self.gdn_at(
            self.conv_out.as_ptr(),
            self.g.as_ptr(),
            self.beta_sig.as_ptr(),
            self.o.as_ptr(),
        )
    }

    /// One gated-delta-net step over the q|k|v rows of `conv_out`.
    pub fn gdn_at(
        &self,
        conv_out: *mut c_void,
        g: *mut c_void,
        beta: *mut c_void,
        o: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        let mut p_s = self.s_state.as_ptr();
        let mut p_q = conv_out;
        let mut p_k = offset(conv_out, d.key_dim());
        let mut p_v = offset(conv_out, 2 * d.key_dim());
        let mut p_g = g;
        let mut p_beta = beta;
        let mut p_out = o;
        let mut v_nv = d.n_v as i32;
        let mut v_nk = d.n_k as i32;
        let mut v_sd = d.head_v as i32;
        let mut params: [*mut c_void; 10] = [
            &mut p_s as *mut _ as *mut c_void,
            &mut p_q as *mut _ as *mut c_void,
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_g as *mut _ as *mut c_void,
            &mut p_beta as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_nv as *mut _ as *mut c_void,
            &mut v_nk as *mut _ as *mut c_void,
            &mut v_sd as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, 1, 1),
            block: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_gdn.launch(cfg, self.stream, &mut params)
    }

    fn gate_norm(&self) -> Result<(), String> {
        self.gate_norm_at(
            self.o.as_ptr(),
            self.z.as_ptr(),
            self.gated.as_ptr(),
        )
    }

    /// `gated = rmsnorm(o, norm) * silu(z)`.
    pub fn gate_norm_at(
        &self,
        o: *mut c_void,
        z: *mut c_void,
        gated: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        let mut p_o = o;
        let mut p_norm = self.norm.as_ptr();
        let mut p_z = z;
        let mut p_y = gated;
        let mut v_h = d.n_v as i32;
        let mut v_d = d.head_v as i32;
        let mut v_eps = d.eps;
        let mut params: [*mut c_void; 7] = [
            &mut p_o as *mut _ as *mut c_void,
            &mut p_norm as *mut _ as *mut c_void,
            &mut p_z as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_h as *mut _ as *mut c_void,
            &mut v_d as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (d.n_v as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        self.k_gate_norm.launch(cfg, self.stream, &mut params)
    }
}

/// Device pointer advanced by `floats` f32s, for the q|k|v slices.
fn offset(p: *mut c_void, floats: usize) -> *mut c_void {
    unsafe { (p as *mut u8).add(floats * 4) as *mut c_void }
}

// ---------------------------------------------------------------------------
// Batched decode: one weight pass for B sequences.
//
// The projections read the same resident weights as the single path; only the
// x/y buffers become `[B][..]`. Everything that touches recurrent state stays
// per slot (one `SsmGpuLayer` per slot owns its conv window and delta-net
// matrix), driven by `recurrent_at` over row `b` of the shared scratch.
// ---------------------------------------------------------------------------

/// The batched-GEMV kind for a resident projection entry.
pub(crate) fn batch_kind_for_entry(entry: &str) -> Option<i32> {
    use crate::launch::{GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K, GEMV_B_Q80};
    Some(match entry {
        "gemv_q4_k" => GEMV_B_Q4K,
        "gemv_q6_k" => GEMV_B_Q6K,
        "gemv_q5_k" => GEMV_B_Q5K,
        "gemv_q8_0" => GEMV_B_Q80,
        _ => return None,
    })
}

impl<'a> SsmGpuLayer<'a> {
    /// The four input projections for `rows` x-vectors: Q8_0/Q6_K/Q5_K/Q4_K
    /// run one batched pass, the tiny F32 alpha/beta run per slot.
    #[allow(clippy::too_many_arguments)]
    pub fn gemvs_batched(
        &self,
        l: &Launcher,
        x: &Buffer,
        qkv: &Buffer,
        z: &Buffer,
        alpha: &Buffer,
        beta: &Buffer,
        rows: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.proj_rows(l, &self.wqkv, x, qkv, d.conv_ch, d.n_embd, rows)?;
        self.proj_rows(l, &self.wgate, x, z, d.inner(), d.n_embd, rows)?;
        self.proj_rows(l, &self.alpha_w, x, alpha, d.n_v, d.n_embd, rows)?;
        self.proj_rows(l, &self.beta_w, x, beta, d.n_v, d.n_embd, rows)
    }

    /// `out = W_out gated` for `rows` rows.
    pub fn output_gemv_batched(
        &self,
        l: &Launcher,
        gated: &Buffer,
        out: &Buffer,
        rows: usize,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.proj_rows(l, &self.wout, gated, out, d.n_embd, d.inner(), rows)
    }

    /// One projection over `rows` rows of `x` (`[rows][k]`) into `y`
    /// (`[rows][m]`). Quantized weights use the batched kernel; the F32
    /// alpha/beta vectors are small enough to run per row.
    #[allow(clippy::too_many_arguments)]
    fn proj_rows(
        &self,
        l: &Launcher,
        w: &SsmWeight<'_>,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        rows: usize,
    ) -> Result<(), String> {
        match w {
            // `alpha`/`beta` are F32 payloads stored in the Quant variant with
            // entry `gemv_f32`; they are 48x5120 and stay per row.
            SsmWeight::Quant(b, entry) if *entry == "gemv_f32" => {
                self.f32_rows(b.as_ptr(), x, y, m, k, rows, l)
            }
            SsmWeight::Quant(b, entry) => {
                let kind = batch_kind_for_entry(entry)
                    .ok_or_else(|| format!("{entry}: no batched GEMV"))?;
                l.gemv_batched(kind, b, x, y, m, k, rows)
            }
            SsmWeight::F32(b) => self.f32_rows(b.as_ptr(), x, y, m, k, rows, l),
        }
    }

    /// Per-row f32 GEMV for the small wide-and-short projections. The batched
    /// engine runs one launch for all rows; the per-row loop below remains for
    /// the single-row case (parity).
    #[allow(clippy::too_many_arguments)]
    fn f32_rows(
        &self,
        w: *mut c_void,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        rows: usize,
        l: &Launcher,
    ) -> Result<(), String> {
        if rows > 1 && std::env::var_os("HIPML_BATCH_F32_LEGACY").is_none() {
            let kernel = self.module.kernel("gemv_f32_b")?;
            let mut p_w = w;
            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_m = m as i32;
            let mut v_k = k as i32;
            let mut v_xs = k as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_w as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_m as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
                &mut v_xs as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (m as u32, rows as u32, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            return kernel.launch(cfg, l.stream(), &mut params);
        }
        for r in 0..rows {
            let mut p_w = w;
            let mut p_x = row_ptr(x, r, k);
            let mut p_y = row_ptr(y, r, m);
            let mut v_m = m as i32;
            let mut v_k = k as i32;
            let mut params: [*mut c_void; 5] = [
                &mut p_w as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_m as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
            ];
            self.k_gemv
                .launch(LaunchConfig::per_row(m, 256), l.stream(), &mut params)?;
        }
        Ok(())
    }

    /// Everything after the projections for slot `row` of the batched scratch:
    /// gates, conv, l2norm, the delta-net recurrence and the gated RMS norm.
    /// This instance's `conv_state`/`s_state` are that slot's recurrent state.
    #[allow(clippy::too_many_arguments)]
    pub fn recurrent_at(
        &self,
        qkv: *mut c_void,
        z: *mut c_void,
        alpha: *mut c_void,
        beta: *mut c_void,
        g: *mut c_void,
        beta_sig: *mut c_void,
        conv: *mut c_void,
        o: *mut c_void,
        gated: *mut c_void,
    ) -> Result<(), String> {
        let d = &self.dims;
        self.gates_at(alpha, beta, g, beta_sig)?;
        self.conv_at(qkv, conv)?;
        self.l2norm_at_ptr(conv, 0, d.n_k, d.head_k)?;
        self.l2norm_at_ptr(conv, d.key_dim(), d.n_k, d.head_k)?;
        self.gdn_at(conv, g, beta_sig, o)?;
        self.gate_norm_at(o, z, gated)
    }
}

/// Device pointer of row `row` of a `[rows][width]` f32 buffer.
fn row_ptr(buf: &Buffer, row: usize, width: usize) -> *mut c_void {
    unsafe { (buf.as_ptr() as *mut u8).add(row * width * 4) as *mut c_void }
}
