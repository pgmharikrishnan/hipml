// SPDX-License-Identifier: Apache-2.0
//! The elementwise launches the decode graph chains between its GEMVs.
//!
//! Every method here works on device buffers and does not synchronize; the
//! graph issues a whole token and syncs once. Argument arrays are written out
//! explicitly because they are the kernel ABI.

use super::*;

/// One entry of the `gemv_jobs` device job list. Mirrors `HipmlGemvJob` in
/// `gemv_jobs.hip`; pointers are device addresses.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvJob {
    pub w: usize,
    pub x: usize,
    pub y: usize,
    pub m: i32,
    pub k: i32,
    pub kind: i32,
    pub pad: i32,
}

/// `kind` values the job kernel dispatches on.
pub const GEMV_JOB_Q4K: i32 = 0;
pub const GEMV_JOB_Q6K: i32 = 1;
pub const GEMV_JOB_Q5K: i32 = 2;
pub const GEMV_JOB_Q80: i32 = 3;
pub const GEMV_JOB_IQ4XS: i32 = 4;

/// Tiles a job occupies: 8 rows a tile for the K-quants, 16 for IQ4_XS (its own
/// `GJ_IQ4X_ROWS_PER_TILE`), one row for Q8_0. Must match `gemv_jobs.hip`.
pub fn gemv_job_tiles(kind: i32, m: usize) -> usize {
    match kind {
        GEMV_JOB_Q80 => m,
        GEMV_JOB_IQ4XS => (m + 15) / 16,
        _ => (m + 7) / 8,
    }
}

/// The job kind for a resident GEMV entry, or `None` when the job kernel
/// carries no body for that format.
pub fn gemv_job_kind(entry: &str) -> Option<i32> {
    Some(match entry {
        "gemv_q4_k" | "gemv_q4_k_r1" => GEMV_JOB_Q4K,
        "gemv_q6_k" | "gemv_q6_k_r1" => GEMV_JOB_Q6K,
        "gemv_q5_k" => GEMV_JOB_Q5K,
        "gemv_q8_0" => GEMV_JOB_Q80,
        "gemv_iq4_xs" => GEMV_JOB_IQ4XS,
        _ => return None,
    })
}

/// The `kind` selector the grouped expert GEMM (`prefill_moe.hip`) decodes a
/// weight tile with. The ids are `pf_dequant8`'s (`prefill_common.inc`), which
/// is a different numbering from `gemv_job_kind`'s: the prefill tile has no
/// packed-fp16 (`halfw`) body because an expert bank is never fp16.
pub fn moe_prefill_kind(entry: &str) -> Option<i32> {
    Some(match entry {
        "gemv_q4_k" | "gemv_q4_k_r1" => 0,
        "gemv_q6_k" | "gemv_q6_k_r1" => 1,
        "gemv_q8_0" => 2,
        "gemv_q5_k" => 4,
        "gemv_iq4_xs" => 5,
        "gemv_iq4_nl" => 6,
        "gemv_q3_k" => 7,
        "gemv_iq3_s" => 8,
        _ => return None,
    })
}

/// The `kind` selector `dequant_rows.hip` decodes a row with.
///
/// Every format the engine can dequantise has a body in that kernel, so this is
/// a name lookup, not a capability list: a resident type that is not one of
/// these has no decoder anywhere in the engine and could not have been loaded.
/// It is one table because the previous shape — a kernel name per type — is
/// what made a Q8_0 `token_embd` fail to load on a file whose every other
/// tensor was fine.
pub fn row_kind(ty: hipml_core::quant::TensorType) -> Result<i32, String> {
    use hipml_core::quant::TensorType as T;
    Ok(match ty {
        T::F32 => 0,
        T::BF16 => 1,
        T::Q8_0 => 2,
        T::Q3K => 3,
        T::Q4K => 4,
        T::Q5K => 5,
        T::Q6K => 6,
        T::IQ4NL => 7,
        T::IQ4XS => 8,
        T::IQ3S => 9,
        other => return Err(format!("{}: no row-dequant kernel", other.name())),
    })
}

/// Blocks a row of `n_embd` values is decoded in, at
/// [`crate::ROW_BLOCK_VALUES`] values a block.
fn row_blocks(n_embd: usize) -> Result<usize, String> {
    if n_embd == 0 || n_embd % ROW_BLOCK_VALUES != 0 {
        return Err(format!(
            "embedding row of {n_embd} values is not a whole number of \
             {ROW_BLOCK_VALUES}-value blocks"
        ));
    }
    Ok(n_embd / ROW_BLOCK_VALUES)
}

/// Batched-GEMV kind ids: the resident weight formats whose decode kernels
/// have a `_b` entry (B x rows, one weight pass).
pub const GEMV_B_Q4K: i32 = 0;
pub const GEMV_B_Q6K: i32 = 1;
pub const GEMV_B_Q5K: i32 = 2;
pub const GEMV_B_Q80: i32 = 3;
pub const GEMV_B_IQ4XS: i32 = 4;
pub const GEMV_B_IQ4NL: i32 = 5;
pub const GEMV_B_Q3K: i32 = 6;
pub const GEMV_B_IQ3S: i32 = 7;

impl Launcher {
    /// The skinny-GEMM entry for a batched kind, when the format has one.
    ///
    /// This is the whole batch integration: `R <= 16` columns are padded to one
    /// 16-wide WMMA tile, so the cost is the weight stream, not R.
    fn skinny_for_kind(kind: i32) -> Option<&'static str> {
        crate::batch_kind_gemv(kind).and_then(crate::skinny_entry)
    }

    /// `HIPML_BATCH_TRACE=1`: name each batched projection route once, so an
    /// operator can see which kernel a slot count actually reaches. Once per
    /// `(entry, m, k, batch)` rather than per step, so a long server run prints
    /// a table and not a firehose.
    ///
    /// The line names the route as well as the entry. Both routes are traced,
    /// so "a line was printed" says nothing about which one ran: at drafts 3 the
    /// MTP verify is a `b=4` batch on the exact `gemv_*_b` kernels, and reading
    /// the bare line as "b>=4 therefore the approximate tile" is a mistake this
    /// word exists to prevent.
    fn trace_route(entry: &'static str, m: usize, k: usize, batch: usize, route: &str) {
        use std::sync::{Mutex, OnceLock};
        type Seen = Mutex<Vec<(&'static str, usize, usize, usize)>>;
        static SEEN: OnceLock<Seen> = OnceLock::new();
        if std::env::var_os("HIPML_BATCH_TRACE").is_none() {
            return;
        }
        let mut seen = SEEN
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .expect("batch trace table");
        let key = (entry, m, k, batch);
        if seen.contains(&key) {
            return;
        }
        seen.push(key);
        eprintln!("[batch] {entry} m={m} k={k} b={batch} route={route}");
    }

    /// `y[b][row] = W[row,:] . x[b][:]` for `batch` rows, one weight pass.
    ///
    /// `x` is `[batch][k]`, `y` is `[batch][m]`. Measured (docs/review.md,
    /// WP33): the f32-from-LDS `_b` kernels stream 381 GB/s at B=2, 272 at B=4
    /// and 154 at B=8; the WMMA skinny GEMM is flat at ~410 GB/s over the whole
    /// R range because R is padded to one 16-wide tile. The exact `_b` path is
    /// the default (bit-identical); `HIPML_SKINNY=1`/`HIPML_BATCH_SKINNY=1`
    /// opts into the faster WMMA tile, `HIPML_BATCH_LEGACY=1` forces `_b`;
    /// `HIPML_BATCH_SKINNY_TRACE=1` names every switched projection.
    ///
    /// Above 16 rows the tile runs once per 16-row group; each group is an
    /// independent weight pass, so cost scales with `ceil(batch / 16)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_batched(
        &self,
        kind: i32,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), String> {
        if batch == 0 || batch > 128 {
            return Err(format!("batched GEMV: batch {batch} outside 1..=128"));
        }
        self.gemv_batched_at(
            kind,
            w,
            x.as_ptr() as *mut u8,
            y.as_ptr() as *mut u8,
            m,
            k,
            batch,
        )
    }

    /// `gemv_batched` over any row count: one weight pass per 128-row window.
    /// A prefill tile of a format with no tiled GEMM can be 2048 rows, and the
    /// `_b` launch stops at 128.
    pub fn gemv_batched_rows(
        &self,
        kind: i32,
        w: &ByteBuffer,
        x: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        rows: usize,
    ) -> Result<(), String> {
        let mut base = 0usize;
        while base < rows {
            let r = (rows - base).min(128);
            let px = unsafe { (x.as_ptr() as *mut u8).add(base * k * 4) };
            let py = unsafe { (y.as_ptr() as *mut u8).add(base * m * 4) };
            self.gemv_batched_at(kind, w, px, py, m, k, r)?;
            base += r;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemv_batched_at(
        &self,
        kind: i32,
        w: &ByteBuffer,
        px: *mut u8,
        py: *mut u8,
        m: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), String> {
        // Opt-in only. The exact `_b` kernels keep each row's accumulation
        // order identical to the single-sequence GEMV, so the default batched
        // path is bit-identical (gemv-check, column-check). The WMMA tile is
        // measured faster from B=4 up but is not bit-identical, so it needs
        // `HIPML_SKINNY=1` (or `HIPML_BATCH_SKINNY=1`); `HIPML_BATCH_LEGACY=1`
        // forces the exact path even when one of those is set. Below B=4 the
        // tile pads R to 16 and loses, so it is only used where it wins.
        let skinny = batch >= 4
            && !crate::exact_forced()
            && std::env::var_os("HIPML_BATCH_LEGACY").is_none()
            && (std::env::var_os("HIPML_BATCH_SKINNY").is_some() || crate::skinny_on());
        if skinny {
            if let Some(entry) = Self::skinny_for_kind(kind) {
                Self::trace_route(entry, m, k, batch, "skinny");
                let mut base = 0usize;
                while base < batch {
                    let r = (batch - base).min(16);
                    if std::env::var_os("HIPML_BATCH_SKINNY_TRACE").is_some() {
                        eprintln!("[skinny] {entry} m={m} k={k} r={r} row0={base}");
                    }
                    let px = unsafe { px.add(base * k * 4) } as *const _;
                    let py = unsafe { py.add(base * m * 4) } as *const _;
                    super::skinny_launch::skinny_resident_rows(
                        &self.module,
                        &self.stream,
                        entry,
                        w,
                        px,
                        py,
                        m,
                        k,
                        r,
                    )?;
                    base += r;
                }
                return Ok(());
            }
        }
        let name = crate::batch_kind_gemv(kind)
            .ok_or_else(|| format!("unknown batched GEMV kind {kind}"))?;
        // One active slot is not a batch: the single-column decode kernel reads
        // the same bytes with a better geometry (one row per warp) and measured
        // 7.4% faster at B=1 on the UD file (28.87 vs 31.19 tok/s in-process,
        // batch-bench). The `_b` row tile stays the path from B=2 up, where its
        // sharing is what wins. The exact contract is unchanged -- this is the
        // kernel `batch-bench --check` already measures every slot against.
        if batch == 1 {
            let entry = crate::stream_entry(name);
            Self::trace_route(entry, m, k, batch, "exact");
            return self.run_quant_resident_ptr(
                entry,
                w,
                px as *mut c_void,
                py as *mut c_void,
                m,
                k,
            );
        }
        let (entry, rows, tile, tid) = match name {
            "gemv_q4_k" => ("gemv_q4_k_b", 16usize, 1024usize, 256u32),
            "gemv_q6_k" => ("gemv_q6_k_b", 16, 1024, 256),
            "gemv_q5_k" => ("gemv_q5_k_b", 16, 1024, 256),
            "gemv_q8_0" => ("gemv_q8_0_b", 1, 0, 256),
            "gemv_iq4_xs" => ("gemv_iq4_xs_b", 16, 1024, 256),
            "gemv_iq4_nl" => ("gemv_iq4_nl_b", 16, 1024, 256),
            "gemv_q3_k" => ("gemv_q3_k_b", 16, 1024, 256),
            "gemv_iq3_s" => ("gemv_iq3_s_b", 16, 1024, 256),
            other => return Err(format!("{other}: no batched GEMV kernel")),
        };
        // Q8_0's batched kernel is one block per output row and one thread per
        // 32-value weight block, so a 256-thread launch only fills `k / 32`
        // threads: at k = 2048 that is 64 of 256, three quarters of the block
        // idle. The extra rows then lengthen the per-thread FMA chain instead
        // of spreading over more parallel work, which is why B=4 used to cost
        // the same as four separate launches (measured: 137 GB/s against 587
        // at B=1, `gemv-bench output.weight`). Sizing the block to the work
        // lets four times as many blocks sit on each CU. Every thread still
        // owns exactly block `t`, so each row's sum order -- and its bits --
        // is unchanged; `column-check` and `gemv-check` prove it.
        let tid = match entry {
            "gemv_q8_0_b" => (((k / 32).max(1) + 31) / 32 * 32).clamp(32, 256) as u32,
            _ => tid,
        };
        Self::trace_route(entry, m, k, batch, "exact");
        let kernel = self.module.kernel(entry)?;
        // The `_b` kernels cover at most 8 rows, so the exact path takes one
        // pass per group (`HIPML_BATCH_LEGACY` at any B, or B < 4 where the
        // tile pads R to 16 and loses). Q8_0's row kernel is throughput-bound
        // above two rows -- it streams 569 GB/s at B=2, 400 at B=3 and 252 at
        // B=4, because the per-thread work grows with B while the block does
        // not -- so four rows are two two-row passes: that reads the weight
        // twice and is still 11% faster than one four-row pass. Three rows
        // stay one pass, which beats a 2+1 split by 39%. Each row's sum order
        // is independent of B, so either split is bit-identical.
        let max_rows = match (entry, batch) {
            ("gemv_q8_0_b", b) if b >= 4 => 2,
            _ => 8,
        };
        let mut base = 0usize;
        while base < batch {
            let r = (batch - base).min(max_rows);
            let tile = if tid == 256 {
                if r <= 4 {
                    1024
                } else {
                    512
                }
            } else {
                tile
            };
            // The Q5_K, Q6_K and IQ4_XS `_b` kernels read the x rows straight
            // from global, so they need no staged tile; only IQ4_XS keeps the
            // 16-float codebook in shared. Allocating the old `r * tile`
            // anyway cost L1 for nothing and made B=4 slower than B=2.
            let shared = match entry {
                "gemv_q5_k_b" | "gemv_q6_k_b" => 0,
                "gemv_iq4_xs_b" => 16 * 4,
                _ => r * tile * 4,
            };
            let mut p_w = w.as_ptr();
            let mut p_x = unsafe { px.add(base * k * 4) } as *mut c_void;
            let mut p_y = unsafe { py.add(base * m * 4) } as *mut c_void;
            let mut v_m = m as i32;
            let mut v_k = k as i32;
            let mut v_b = r as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_w as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_m as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
                &mut v_b as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (m.div_ceil(rows) as u32, 1, 1),
                block: (tid, 1, 1),
                shared_mem_bytes: shared as u32,
            };
            kernel.launch(cfg, &self.stream, &mut params)?;
            base += r;
        }
        Ok(())
    }

    /// Batched greedy argmax over `rows` logit vectors of `n` values each.
    /// `best_v`/`best_i` hold `rows * parts` entries, `out` holds `rows` ids.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_batched(
        &self,
        x: &Buffer,
        best_v: &Buffer,
        best_i: &Buffer,
        out: &Buffer,
        n: usize,
        parts: usize,
        rows: usize,
    ) -> Result<(), String> {
        let part = self.module.kernel("argmax_part_rows_f32")?;
        let mut p_x = x.as_ptr();
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut v_n = n as i32;
        let mut v_parts = parts as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (parts as u32, rows as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        part.launch(cfg, &self.stream, &mut params)?;

        let fin = self.module.kernel("argmax_final_rows_f32")?;
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut p_out = out.as_ptr();
        let mut params: [*mut c_void; 4] = [
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (rows as u32, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        fin.launch(cfg, &self.stream, &mut params)
    }

    /// Split-K Q4_K GEMV: row tiles x splits, then the ordered reduce.
    pub fn gemv_q4_k_split(
        &self,
        w: &ByteBuffer,
        x: &Buffer,
        part: &Buffer,
        y: &Buffer,
        m: usize,
        k: usize,
        splits: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("gemv_q4_k_split")?;
        let mut p_w = w.as_ptr();
        let mut p_x = x.as_ptr();
        let mut p_part = part.as_ptr();
        let mut v_m = m as i32;
        let mut v_k = k as i32;
        let mut v_splits = splits as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_x as *mut _ as *mut c_void,
            &mut p_part as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_k as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (((m + 7) / 8) as u32, splits as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)?;

        let reduce = self.module.kernel("gemv_split_reduce")?;
        let mut p_part = part.as_ptr();
        let mut p_y = y.as_ptr();
        let mut params: [*mut c_void; 4] = [
            &mut p_part as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_m as *mut _ as *mut c_void,
            &mut v_splits as *mut _ as *mut c_void,
        ];
        reduce.launch(LaunchConfig::linear(m, 256), &self.stream, &mut params)
    }

    /// Launch every job in `jobs` with one kernel: grid.y picks the job, grid.x
    /// the row tile inside it. Jobs must be independent of each other.
    pub fn gemv_jobs(&self, jobs: &Buffer, n_jobs: usize, tiles: usize) -> Result<(), String> {
        let kernel = self.module.kernel("gemv_jobs")?;
        let mut p_jobs = jobs.as_ptr();
        let mut v_n = n_jobs as i32;
        let mut params: [*mut c_void; 2] = [
            &mut p_jobs as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (tiles.max(1) as u32, n_jobs.max(1) as u32, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }

    /// `y = x / sqrt(mean(x^2) + eps) * w`, one block per row (row count 1 here).
    pub fn rmsnorm_buf(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        self.rmsnorm_rows(x, w, y, n, eps, 1)
    }

    /// `y[r][:] = rmsnorm(x[r][:], w)` for `rows` rows, one block per row. The
    /// prefill path runs a whole prompt tile through the same kernel.
    pub fn rmsnorm_rows(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        self.rmsnorm_ptr(x.as_ptr(), w, y.as_ptr(), n, eps, rows)
    }

    /// Same launch with raw pointers, so a chunk can normalise one row in place
    /// without a copy (`rmsnorm` reads `row * n` from the pointer).
    pub fn rmsnorm_ptr(
        &self,
        x: *mut c_void,
        w: &Buffer,
        y: *mut c_void,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("rmsnorm_f32")?;
        let mut p_x = x;
        let mut p_w = w.as_ptr();
        let mut p_y = y;
        let mut v_n = n as i32;
        let mut v_eps = eps;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::per_row(rows, 256), &self.stream, &mut params)
    }

    /// `y = a + b`, one thread per element.
    pub fn residual_buf(&self, a: &Buffer, b: &Buffer, y: &Buffer, n: usize) -> Result<(), String> {
        self.linear3("residual_add_f32", a, b, y, n)
    }

    /// `res = a + b` then `y = rmsnorm(res, w)` in one per-row pass: the fused
    /// kernel keeps the row in registers, so it is bit-identical to
    /// [`Self::residual_buf`] followed by [`Self::rmsnorm_ptr`] but skips one
    /// global read of the row. `n` must be at most `32 * 256` (the kernel's
    /// `MAXK * BLOCK`); callers fall back to the pair otherwise.
    pub fn residual_rmsnorm(
        &self,
        a: &Buffer,
        b: &Buffer,
        w: &Buffer,
        res: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel("residual_rmsnorm_f32")?;
        let mut p_a = a.as_ptr();
        let mut p_b = b.as_ptr();
        let mut p_w = w.as_ptr();
        let mut p_res = res.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_n = n as i32;
        let mut v_eps = eps;
        let mut params: [*mut c_void; 7] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_w as *mut _ as *mut c_void,
            &mut p_res as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_eps as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::per_row(rows, 256), &self.stream, &mut params)
    }

    /// [`Self::residual_rmsnorm`] when the row fits the fused kernel's register
    /// window, otherwise the two passes it replaces. Keeps the `MAXK * BLOCK`
    /// limit an implementation detail of the kernel.
    pub fn residual_rmsnorm_or_pair(
        &self,
        a: &Buffer,
        b: &Buffer,
        w: &Buffer,
        res: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), String> {
        if n <= 32 * 256 {
            self.residual_rmsnorm(a, b, w, res, y, n, eps, rows)
        } else {
            self.residual_buf(a, b, res, rows * n)?;
            self.rmsnorm_ptr(res.as_ptr(), w, y.as_ptr(), n, eps, rows)
        }
    }

    /// [`Launcher::residual_buf`] on raw pointers, for one row of a batch.
    pub fn residual_ptr(
        &self,
        a: *mut c_void,
        b: *mut c_void,
        y: *mut c_void,
        n: usize,
    ) -> Result<(), String> {
        self.linear3_ptr("residual_add_f32", a, b, y, n)
    }

    /// `y = silu(gate) * up`, one thread per element.
    pub fn silu_mul_buf(
        &self,
        gate: &Buffer,
        up: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<(), String> {
        self.linear3("silu_mul_f32", gate, up, y, n)
    }

    /// [`Launcher::silu_mul_buf`] on raw pointers, for one row of a batch.
    pub fn silu_mul_ptr(
        &self,
        gate: *mut c_void,
        up: *mut c_void,
        y: *mut c_void,
        n: usize,
    ) -> Result<(), String> {
        self.linear3_ptr("silu_mul_f32", gate, up, y, n)
    }

    fn linear3_ptr(
        &self,
        name: &str,
        a: *mut c_void,
        b: *mut c_void,
        y: *mut c_void,
        n: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel(name)?;
        let mut p_a = a;
        let mut p_b = b;
        let mut p_y = y;
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 4] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }

    /// `out = a * b` elementwise for the two linear kernels above.
    fn linear3(
        &self,
        name: &str,
        a: &Buffer,
        b: &Buffer,
        y: &Buffer,
        n: usize,
    ) -> Result<(), String> {
        let kernel = self.module.kernel(name)?;
        let mut p_a = a.as_ptr();
        let mut p_b = b.as_ptr();
        let mut p_y = y.as_ptr();
        let mut v_n = n as i32;
        let mut params: [*mut c_void; 4] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_y as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
        ];
        kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)
    }

    /// Dequantise the embedding row named by `step[2]` into `out`: the step's
    /// token id, so a captured graph is replayed for a new token without
    /// re-capturing. `ty` selects the block decoder.
    pub fn embed_row_dev(
        &self,
        ty: hipml_core::quant::TensorType,
        w: &ByteBuffer,
        row_bytes: usize,
        out: &Buffer,
        n_embd: usize,
    ) -> Result<(), String> {
        let nblocks = row_blocks(n_embd)?;
        let kernel = self.module.kernel("embed_row_any")?;
        let mut p_w = w.as_ptr();
        let mut p_out = out.as_ptr();
        let mut v_blocks = nblocks as i32;
        let mut v_row_bytes = row_bytes as i32;
        let mut p_step = self.step.as_ptr();
        let mut v_kind = row_kind(ty)?;
        let mut params: [*mut c_void; 6] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_blocks as *mut _ as *mut c_void,
            &mut v_row_bytes as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::per_row(nblocks, 256),
            &self.stream,
            &mut params,
        )
    }

    /// Dequantise one embedding row of a resident tensor into `out`, on the
    /// device: the graph's `embed` step, with no host round-trip.
    ///
    /// `row_bytes` is the tensor's own row stride, so this works for any
    /// `n_embd` that is a whole number of 256-value blocks.
    pub fn embed_row(
        &self,
        ty: hipml_core::quant::TensorType,
        w: &ByteBuffer,
        row: usize,
        row_bytes: usize,
        out: &Buffer,
    ) -> Result<(), String> {
        self.embed_row_at(ty, w, row, row_bytes, out, 0, out.len())
    }

    /// Same, writing the dequantized row at float offset `dst` of `out`: the
    /// chunked prefill embeds a whole tile into one `[tokens][n_embd]` buffer.
    pub fn embed_row_at(
        &self,
        ty: hipml_core::quant::TensorType,
        w: &ByteBuffer,
        row: usize,
        row_bytes: usize,
        out: &Buffer,
        dst: usize,
        n_embd: usize,
    ) -> Result<(), String> {
        let nblocks = row_blocks(n_embd)?;
        let kernel = self.module.kernel("dequant_row_any")?;
        let mut p_w = unsafe { (w.as_ptr() as *mut u8).add(row * row_bytes) } as *mut c_void;
        let mut p_out = unsafe { (out.as_ptr() as *mut u8).add(dst * 4) } as *mut c_void;
        let mut v_blocks = nblocks as i32;
        let mut v_kind = row_kind(ty)?;
        let mut params: [*mut c_void; 4] = [
            &mut p_w as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_blocks as *mut _ as *mut c_void,
            &mut v_kind as *mut _ as *mut c_void,
        ];
        kernel.launch(
            LaunchConfig::per_row(nblocks, 256),
            &self.stream,
            &mut params,
        )
    }

    /// Device argmax over `n` values, `parts` blocks then one final fold.
    ///
    /// `best_v` holds `parts` f32, `best_i` `parts` i32, `out` one i32; the
    /// caller reads `out` back with `download_bytes(0, 4)`.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_buf(
        &self,
        x: &Buffer,
        best_v: &Buffer,
        best_i: &Buffer,
        out: &Buffer,
        n: usize,
        parts: usize,
    ) -> Result<(), String> {
        let part = self.module.kernel("argmax_part_f32")?;
        let mut p_x = x.as_ptr();
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut v_n = n as i32;
        let mut v_parts = parts as i32;
        let mut params: [*mut c_void; 5] = [
            &mut p_x as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut v_n as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (parts as u32, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        part.launch(cfg, &self.stream, &mut params)?;

        let fin = self.module.kernel("argmax_final_f32")?;
        let mut p_v = best_v.as_ptr();
        let mut p_i = best_i.as_ptr();
        let mut p_out = out.as_ptr();
        let mut params: [*mut c_void; 4] = [
            &mut p_v as *mut _ as *mut c_void,
            &mut p_i as *mut _ as *mut c_void,
            &mut p_out as *mut _ as *mut c_void,
            &mut v_parts as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        fin.launch(cfg, &self.stream, &mut params)
    }

    /// Quantize one token's K and V rows into the cache at `step[0]`.
    ///
    /// One launch per layer replaces a readback, a host quantize and eight
    /// uploads. `dtype` is `KvDtype::id()`: 0 q8_0, 1 q4_0. `stride` is the
    /// cache's rows per KV head and stays fixed; the position comes from the
    /// step buffer so a captured graph replays at every step.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode(
        &self,
        dtype: usize,
        k_in: &Buffer,
        v_in: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<(), String> {
        self.kv_encode_ptr(
            dtype,
            k_in.as_ptr(),
            v_in.as_ptr(),
            k_cache,
            v_cache,
            n_kv,
            head_dim,
            stride,
        )
    }

    /// Quantize and append one K/V row whose source pointer is a row inside a
    /// larger chunk buffer. Used by the chunked q8 prefill. The position comes
    /// from the step buffer, exactly as it does for the whole-buffer entry.
    pub fn kv_encode_ptr(
        &self,
        dtype: usize,
        k_ptr: *mut std::ffi::c_void,
        v_ptr: *mut std::ffi::c_void,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<(), String> {
        self.kv_encode_ptr_on(
            &self.stream,
            &self.step,
            dtype,
            k_ptr,
            v_ptr,
            k_cache,
            v_cache,
            n_kv,
            head_dim,
            stride,
        )
    }

    /// Same append on a chosen stream with a chosen step buffer: the two-stream
    /// batch path gives each stream its own position.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_ptr_on(
        &self,
        stream: &Stream,
        step: &Buffer,
        dtype: usize,
        k_ptr: *mut std::ffi::c_void,
        v_ptr: *mut std::ffi::c_void,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_q8",
            1 => "kv_encode_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_ptr;
        let mut p_v = v_ptr;
        let mut p_kc = k_cache.as_ptr();
        let mut p_vc = v_cache.as_ptr();
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut p_step = step.as_ptr();
        let mut params: [*mut c_void; 7] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut p_step as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_kv as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, stream, &mut params)
    }

    /// Quantize `rows` positions of a position-major f32 buffer into the
    /// head-major q8/q4 cache. `start` is the destination position of the first
    /// row; `src0` is the source row of `k_f32`/`v_f32` that holds it (0 for a
    /// chunk-local tile, `start` for the full f32 mirror).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_encode_bulk(
        &self,
        dtype: usize,
        k_f32: &Buffer,
        v_f32: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        n_kv: usize,
        head_dim: usize,
        stride: usize,
        start: usize,
        src0: usize,
        rows: usize,
    ) -> Result<(), String> {
        let entry = match dtype {
            0 => "kv_encode_bulk_q8",
            1 => "kv_encode_bulk_q4",
            other => return Err(format!("unknown KV dtype {other}")),
        };
        let kernel = self.module.kernel(entry)?;
        let mut p_k = k_f32.as_ptr();
        let mut p_v = v_f32.as_ptr();
        let mut p_kc = k_cache.as_ptr();
        let mut p_vc = v_cache.as_ptr();
        let mut v_nkv = n_kv as i32;
        let mut v_hd = head_dim as i32;
        let mut v_stride = stride as i32;
        let mut v_start = start as i32;
        let mut v_src0 = src0 as i32;
        let mut v_rows = rows as i32;
        let mut params: [*mut c_void; 10] = [
            &mut p_k as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut p_kc as *mut _ as *mut c_void,
            &mut p_vc as *mut _ as *mut c_void,
            &mut v_nkv as *mut _ as *mut c_void,
            &mut v_hd as *mut _ as *mut c_void,
            &mut v_stride as *mut _ as *mut c_void,
            &mut v_start as *mut _ as *mut c_void,
            &mut v_rows as *mut _ as *mut c_void,
            &mut v_src0 as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: ((n_kv * rows) as u32, 1, 1),
            block: (head_dim.max(32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        kernel.launch(cfg, &self.stream, &mut params)
    }
}
