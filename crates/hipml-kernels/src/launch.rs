// SPDX-License-Identifier: Apache-2.0
//! Device-side dispatch: one match arm per kernel, argument lists written out.
//!
//! Shapes and scalars come from the parity case, so the launcher and the
//! reference are driven by exactly the same numbers. The argument arrays are
//! explicit because they are the kernel ABI: `hipModuleLaunchKernel` takes
//! pointers to the arguments, so every value needs a stable home in a local.

use hipml_core::parity::OpCase;
use hipml_core::tensor::Tensor;

/// Values one block of the embedding row-gather kernel covers: one K-quant
/// super-block, eight Q8_0/IQ4_NL blocks, or 1 KiB / 512 B of an F32 / BF16 row.
///
/// Mirrors `DR_VALUES` in `dequant_rows.hip`. Defined here rather than beside
/// the launcher so the host-side row arithmetic is the same number with and
/// without the `hip` feature.
pub const ROW_BLOCK_VALUES: usize = 256;

/// Grid for the warp-per-output `gemm_rows_f32` kernel: one warp per
/// `(token, output row)` pair, eight warps (256 threads) per block.
///
/// Kept outside the `hip` gate so the launch shape has a host-side regression
/// test. The first version passed the *block count* to `LaunchConfig::linear`,
/// which divides by the block size again: a 1-token chunk launched one block
/// instead of six and 40 of the 48 SSM alpha/beta rows were never written.
pub fn gemm_rows_grid(tokens: usize, m: usize) -> (usize, usize) {
    const WARPS: usize = 8;
    const THREADS: usize = WARPS * 32;
    let warps = tokens * m;
    ((warps + WARPS - 1) / WARPS, THREADS)
}

/// Number of k slices for a prefill GEMM whose row tile is only a few blocks.
///
/// The fp16 tile is 128 output rows x 256 tokens, so a 64-row SSM projection
/// over a 2048-token chunk is 8 blocks: on a 64-block card the matrix units
/// idle and the projection costs ~0.8 s of a 100K prefill. Slicing k across
/// `blockIdx.z` fills a wave instead. Only the alpha/beta path uses this, so
/// the rule can stay blunt: one wave, at most eight slices, each slice a whole
/// number of 64-value k-tiles.
pub fn prefill_slices(m: usize, k: usize, tokens: usize) -> usize {
    let blocks = m.div_ceil(128) * tokens.div_ceil(256);
    if blocks >= 32 {
        return 1;
    }
    let mut n = 1;
    while n < 8 && blocks * n < 64 {
        n *= 2;
    }
    while n > 1 && (k / n) % 64 != 0 {
        n /= 2;
    }
    n
}

#[cfg(feature = "hip")]
pub(crate) mod device {
    use super::*;
    use hipml_hip::{Buffer, ByteBuffer, Device, Event, LaunchConfig, Module, Stream};
    use std::ffi::c_void;

    #[path = "graph_launch.rs"]
    mod graph_launch;
    pub use graph_launch::{
        gemv_job_kind, gemv_job_tiles, moe_prefill_kind, row_kind, GemvJob, GEMV_B_IQ3S,
        GEMV_B_IQ4NL, GEMV_B_IQ4XS, GEMV_B_Q3K, GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K, GEMV_B_Q80,
        GEMV_JOB_IQ4XS, GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80,
    };
    #[path = "moe_launch.rs"]
    mod moe_launch;
    pub use moe_launch::{
        moe_prefill_padding, MOE_PREFILL_ROWS, MOE_PREFILL_THREADS, MOE_PREFILL_TILES,
        MOE_PREFILL_TOKENS, MOE_TOP_K_MAX,
    };
    #[path = "prefill_launch.rs"]
    mod prefill_launch;

    pub use prefill_launch::{
        gemm_rows_f32, prefill_entry_for_half, prefill_gu_entry, prefill_has_gu, prefill_has_silu,
        prefill_has_xhalf, prefill_resident, prefill_resident_gu, prefill_resident_half,
        prefill_resident_silu, prefill_resident_slices, prefill_resident_split_half,
        prefill_should_split, prefill_split_scratch_cap, prefill_xhalf,
    };
    #[path = "batch_launch.rs"]
    mod batch_launch;
    #[path = "quant_launch.rs"]
    mod quant_launch;
    #[path = "skinny_launch.rs"]
    mod skinny_launch;
    #[path = "stream_launch.rs"]
    mod stream_launch;
    pub use stream_launch::ReadJob;
    pub struct Launcher {
        pub(crate) module: Module,
        /// The graph's main stream: every op that has no declared partner.
        pub(crate) stream: Stream,
        /// The side stream. Independent projections run here so their blocks
        /// fill the main stream's waves instead of waiting for its tail.
        pub(crate) side: Stream,
        /// Reusable ordering events. `ev_in` gates the side stream on the main
        /// one, `ev_out` gates the main stream on the side one.
        pub(crate) ev_in: Event,
        pub(crate) ev_out: Event,
        /// The step scalars, in device memory:
        /// `[pos, seq, token, 0, p0, p1, p2, p3]`. The position, the sequence
        /// length and the token id are the only kernel arguments that change
        /// between decode steps, so keeping them here is what makes one captured
        /// graph replayable. The last four are the M-RoPE section positions
        /// (all the token position for text), read as `step + 16 bytes`.
        pub(crate) step: Buffer,
        /// The side stream's copy of the step scalars, so per-slot KV appends
        /// on the two streams do not race on one position buffer.
        pub(crate) step_side: Buffer,
        pub device_name: String,
    }

    impl Launcher {
        pub fn new() -> Result<Launcher, String> {
            let dir = crate::kernel_dir()
                .ok_or_else(|| "kernels were not compiled: set HIP_PATH and rebuild".to_string())?;
            let device = Device::first()?;
            let module = Module::load(&dir.join("hipml.co"))?;
            let stream = Stream::new()?;
            let side = Stream::new()?;
            let ev_in = Event::new()?;
            let ev_out = Event::new()?;
            let step = Buffer::alloc(8)?;
            let step_side = Buffer::alloc(8)?;
            Ok(Launcher {
                module,
                stream,
                side,
                ev_in,
                ev_out,
                step,
                step_side,
                device_name: device.name().to_string(),
            })
        }

        /// Write `[pos, seq, token]` for the next launches. Async on the main
        /// stream, so it is ordered ahead of them and never stalls the host.
        pub fn set_step(&self, pos: usize, seq: usize, token: usize) -> Result<(), String> {
            let p = pos as i32;
            let mut bytes = [0u8; 32];
            for (i, v) in [p, seq as i32, token as i32, 0, p, p, p, p]
                .iter()
                .enumerate()
            {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            self.step.upload_bytes_async(0, &bytes, &self.stream)
        }

        /// The device buffer the position-aware kernels read.
        pub fn step_buf(&self) -> &Buffer {
            &self.step
        }

        /// Same `[pos, seq, token, 0, p0..p3]` write for the side stream.
        pub fn set_step_side(&self, pos: usize, seq: usize, token: usize) -> Result<(), String> {
            let p = pos as i32;
            let mut bytes = [0u8; 32];
            for (i, v) in [p, seq as i32, token as i32, 0, p, p, p, p]
                .iter()
                .enumerate()
            {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            self.step_side.upload_bytes_async(0, &bytes, &self.side)
        }

        pub fn step_side_buf(&self) -> &Buffer {
            &self.step_side
        }

        /// The M-RoPE section positions: the four ints after `[pos, seq, token]`.
        pub fn mrope_pos(&self) -> *mut c_void {
            unsafe { (self.step.as_ptr() as *mut i32).add(4) as *mut c_void }
        }

        pub fn module(&self) -> &Module {
            &self.module
        }

        pub fn stream(&self) -> &Stream {
            &self.stream
        }

        pub fn stream2(&self) -> &Stream {
            &self.side
        }

        /// Start a parallel section: the side stream may run once everything
        /// already on the main stream has completed.
        pub fn fork(&self) -> Result<(), String> {
            self.ev_in.record(&self.stream)?;
            self.side.wait(&self.ev_in)
        }

        /// End it: the main stream continues only after the side stream does.
        pub fn join(&self) -> Result<(), String> {
            self.ev_out.record(&self.side)?;
            self.stream.wait(&self.ev_out)
        }

        /// Wait for both streams (a token boundary, or a timing readback).
        pub fn synchronize(&self) -> Result<(), String> {
            self.side.synchronize()?;
            self.stream.synchronize()
        }

        pub fn run(&self, case: &OpCase, inputs: &[Tensor]) -> Result<Vec<f32>, String> {
            self.run_reps(case, inputs, 1)
        }

        /// Launch the kernel `reps` times over one set of buffers.
        ///
        /// The copies happen once, so timing two different `reps` and taking the
        /// difference gives the launch cost without the transfer.
        pub fn run_reps(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            match case.name {
                "gemv_f32" => self.gemv(case, inputs, reps),
                "gemm_f32" => self.gemm(case, inputs, reps),
                "rmsnorm_f32" => self.rmsnorm(case, inputs, reps),
                "rope_neox_f32" => self.rope(case, inputs, reps),
                "silu_mul_f32" => self.linear2(case, inputs, reps),
                "residual_add_f32" => self.linear2(case, inputs, reps),
                "softmax_f32" => self.softmax(case, inputs, reps),
                "attn_decode_f32" | "attn_decode_pm_f32" => self.attention(case, inputs, reps),
                n if n.starts_with("attn_decode_kv") => self.attention_kv(case, inputs, reps),
                "qg_split_f32"
                | "head_rmsnorm_f32"
                | "mrope_f32"
                | "q_gate_f32"
                | "kv_append_f32"
                | "attn_decode_split_f32" => self.run_attn_case(case, inputs, reps),
                "kv_encode_q8" | "kv_encode_q4" => self.kv_encode_case(case, inputs, reps),
                "argmax_f32" => self.argmax_case(case, inputs, reps),
                "ssm_conv1d_f32" => {
                    crate::ssm_launch::conv(self.module(), self.stream(), case, inputs, reps)
                }
                "gdn_ar_f32" => {
                    crate::ssm_launch::gdn(self.module(), self.stream(), case, inputs, reps)
                }
                "ssm_l2norm_f32" => {
                    crate::ssm_launch::l2norm(self.module(), self.stream(), case, inputs, reps)
                }
                "ssm_gate_norm_f32" => {
                    crate::ssm_launch::gate_norm(self.module(), self.stream(), case, inputs, reps)
                }
                "ssm_layer_f32" => {
                    crate::ssm_launch::layer(self.module(), self.stream(), case, inputs, reps)
                }
                other => Err(format!("{other} has no device launcher")),
            }
        }

        fn gemv(&self, case: &OpCase, inputs: &[Tensor], reps: usize) -> Result<Vec<f32>, String> {
            let (m, k) = (case.scalar("m") as usize, case.scalar("k") as usize);
            let w = Buffer::from_host(inputs[0].data())?;
            let x = Buffer::from_host(inputs[1].data())?;
            let y = Buffer::alloc(m)?;
            let kernel = self.module.kernel("gemv_f32")?;

            let mut p_w = w.as_ptr();
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
            for _ in 0..reps {
                kernel.launch(LaunchConfig::per_row(m, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn gemm(&self, case: &OpCase, inputs: &[Tensor], reps: usize) -> Result<Vec<f32>, String> {
            let (r, m, k) = (
                case.scalar("r") as usize,
                case.scalar("m") as usize,
                case.scalar("k") as usize,
            );
            let w = Buffer::from_host(inputs[0].data())?;
            let x = Buffer::from_host(inputs[1].data())?;
            let y = Buffer::alloc(r * m)?;
            let kernel = self.module.kernel("gemm_f32")?;
            let mut p_w = w.as_ptr();
            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_r = r as i32;
            let mut v_m = m as i32;
            let mut v_k = k as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_w as *mut _ as *mut c_void,
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_r as *mut _ as *mut c_void,
                &mut v_m as *mut _ as *mut c_void,
                &mut v_k as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(LaunchConfig::per_row(m, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn rmsnorm(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n = case.scalar("n") as usize;
            let eps = case.scalar("eps");
            let x = Buffer::from_host(inputs[0].data())?;
            let w = Buffer::from_host(inputs[1].data())?;
            let y = Buffer::alloc(n)?;
            let kernel = self.module.kernel("rmsnorm_f32")?;

            let mut p_x = x.as_ptr();
            let mut p_w = w.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_n = n as i32;
            let mut v_eps = eps;
            let mut params: [*mut c_void; 5] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
            ];
            // The reference case is a single row; one block covers it.
            for _ in 0..reps {
                kernel.launch(LaunchConfig::per_row(1, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn rope(&self, case: &OpCase, inputs: &[Tensor], reps: usize) -> Result<Vec<f32>, String> {
            let heads = case.scalar("n_heads") as usize;
            let dim = case.scalar("head_dim") as usize;
            let pos = case.scalar("pos") as i32;
            let theta_base = case.scalar("theta_base");
            let x = Buffer::from_host(inputs[0].data())?;
            let y = Buffer::alloc(heads * dim)?;
            let kernel = self.module.kernel("rope_neox_f32")?;

            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_heads = heads as i32;
            let mut v_dim = dim as i32;
            let mut v_pos = pos;
            let mut v_base = theta_base;
            let mut params: [*mut c_void; 6] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_heads as *mut _ as *mut c_void,
                &mut v_dim as *mut _ as *mut c_void,
                &mut v_pos as *mut _ as *mut c_void,
                &mut v_base as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (heads as u32, 1, 1),
                block: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            for _ in 0..reps {
                kernel.launch(cfg, &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        /// Two same-length inputs, one output: `silu_mul` and `residual_add`.
        fn linear2(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n = inputs[0].numel();
            let a = Buffer::from_host(inputs[0].data())?;
            let b = Buffer::from_host(inputs[1].data())?;
            let out = Buffer::alloc(n)?;
            let kernel = self.module.kernel(case.name)?;

            let mut p_a = a.as_ptr();
            let mut p_b = b.as_ptr();
            let mut p_out = out.as_ptr();
            let mut v_n = n as i32;
            let mut params: [*mut c_void; 4] = [
                &mut p_a as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            out.to_host()
        }

        fn attention(
            &self,

            case: &OpCase,

            inputs: &[Tensor],

            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n_heads = case.scalar("n_heads") as usize;

            let n_kv = case.scalar("n_kv") as usize;

            let head_dim = case.scalar("head_dim") as usize;

            let seq = case.scalar("seq") as usize;

            let scale = case.scalar("scale");

            let q = Buffer::from_host(inputs[0].data())?;

            let k = Buffer::from_host(inputs[1].data())?;

            let v = Buffer::from_host(inputs[2].data())?;

            let out = Buffer::alloc(n_heads * head_dim)?;

            let kernel = self.module.kernel("attn_decode_f32")?;

            let mut p_q = q.as_ptr();

            let mut p_k = k.as_ptr();

            let mut p_v = v.as_ptr();

            let mut p_out = out.as_ptr();

            let mut v_heads = n_heads as i32;

            let mut v_kv = n_kv as i32;

            let mut v_dim = head_dim as i32;

            self.set_step(0, seq, 0)?;

            let mut p_step = self.step.as_ptr();

            let mut v_scale = scale;

            // The generic case stores the cache kv-major; the position-major
            // case matches attention::KvState.
            let pm = case.name == "attn_decode_pm_f32";
            let mut v_head_stride = if pm {
                head_dim as i32
            } else {
                (seq * head_dim) as i32
            };
            let mut v_row_stride = if pm {
                (n_kv * head_dim) as i32
            } else {
                head_dim as i32
            };

            let mut params: [*mut c_void; 11] = [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_heads as *mut _ as *mut c_void,
                &mut v_kv as *mut _ as *mut c_void,
                &mut v_dim as *mut _ as *mut c_void,
                &mut p_step as *mut _ as *mut c_void,
                &mut v_scale as *mut _ as *mut c_void,
                &mut v_head_stride as *mut _ as *mut c_void,
                &mut v_row_stride as *mut _ as *mut c_void,
            ];

            // One block per head; the block must be exactly head_dim wide because each

            // thread owns one component of the query and of the accumulator.

            let cfg = LaunchConfig {
                grid: (n_heads as u32, 1, 1),

                block: (head_dim as u32, 1, 1),

                shared_mem_bytes: 0,
            };

            for _ in 0..reps {
                kernel.launch(cfg, &self.stream, &mut params)?;
            }

            self.stream.synchronize()?;

            out.to_host()
        }

        /// Quantized-KV decode attention: the split pass and the reduce, so the
        /// timing covers both.
        fn attention_kv(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            crate::launch_kv::run(&self.module, &self.stream, &self.step, case, inputs, reps)
        }

        /// Encode one token's K/V rows, then decode the bytes that landed in the
        /// cache row so the comparison is against `kv::quantize`'s output.
        fn kv_encode_case(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n_kv = case.scalar("n_kv") as usize;
            let head_dim = case.scalar("head_dim") as usize;
            let seq = case.scalar("seq") as usize;
            let pos = case.scalar("pos") as usize;
            let dtype = case.scalar("dtype") as usize;
            let kv_dtype = if dtype == 0 {
                hipml_core::kv::KvDtype::Q8
            } else {
                hipml_core::kv::KvDtype::Q4
            };
            let k = Buffer::from_host(inputs[0].data())?;
            let v = Buffer::from_host(inputs[1].data())?;
            let row = kv_dtype.row_bytes(head_dim);
            let nbytes = n_kv * seq * row;
            let kc = Buffer::alloc(nbytes.div_ceil(4))?;
            let vc = Buffer::alloc(nbytes.div_ceil(4))?;
            self.set_step(pos, seq, 0)?;
            for _ in 0..reps {
                self.kv_encode(dtype, &k, &v, &kc, &vc, n_kv, head_dim, seq)?;
            }
            self.stream.synchronize()?;
            let kb = kc.download_bytes(0, nbytes)?;
            let vb = vc.download_bytes(0, nbytes)?;
            let mut out = Vec::with_capacity(2 * n_kv * head_dim);
            for bytes in [&kb, &vb] {
                for h in 0..n_kv {
                    let at = (h * seq + pos) * row;
                    out.extend_from_slice(&hipml_core::kv::dequantize(
                        kv_dtype,
                        &bytes[at..at + row],
                    ));
                }
            }
            Ok(out)
        }

        /// Device argmax against the host's, one index.
        fn argmax_case(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n = case.scalar("n") as usize;
            let x = Buffer::from_host(inputs[0].data())?;
            let parts = 8usize;
            let best_v = Buffer::alloc(parts)?;
            let best_i = Buffer::alloc(parts)?;
            let out = Buffer::alloc(1)?;
            for _ in 0..reps {
                self.argmax_buf(&x, &best_v, &best_i, &out, n, parts)?;
            }
            self.stream.synchronize()?;
            let bytes = out.download_bytes(0, 4)?;
            let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            Ok(vec![id as f32])
        }

        fn softmax(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n = case.scalar("n") as usize;
            let x = Buffer::from_host(inputs[0].data())?;
            let y = Buffer::alloc(n)?;
            let kernel = self.module.kernel("softmax_f32")?;

            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_n = n as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(LaunchConfig::per_row(1, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }
    }
}

#[cfg(feature = "hip")]
pub use device::{gemm_rows_f32, prefill_resident, prefill_resident_slices, prefill_xhalf};
#[cfg(feature = "hip")]
pub use device::{
    gemv_job_kind, gemv_job_tiles, moe_prefill_kind, moe_prefill_padding, GemvJob, Launcher,
    ReadJob, MOE_PREFILL_ROWS, MOE_PREFILL_THREADS, MOE_PREFILL_TILES, MOE_PREFILL_TOKENS,
};
#[cfg(feature = "hip")]
pub use device::{
    prefill_entry_for_half, prefill_gu_entry, prefill_has_gu, prefill_has_silu, prefill_has_xhalf,
    prefill_resident_gu, prefill_resident_half, prefill_resident_silu, prefill_resident_split_half,
    prefill_should_split, prefill_split_scratch_cap,
};
#[cfg(feature = "hip")]
pub use device::{row_kind, MOE_TOP_K_MAX};
#[cfg(feature = "hip")]
pub use device::{
    GEMV_B_IQ3S, GEMV_B_IQ4NL, GEMV_B_IQ4XS, GEMV_B_Q3K, GEMV_B_Q4K, GEMV_B_Q5K, GEMV_B_Q6K,
    GEMV_B_Q80,
};
#[cfg(feature = "hip")]
pub use device::{GEMV_JOB_IQ4XS, GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80};

/// Without the `hip` feature the launcher exists but refuses to run, so callers
/// need no conditional compilation of their own.
#[cfg(not(feature = "hip"))]
pub struct Launcher {
    pub device_name: String,
}

#[cfg(not(feature = "hip"))]
impl Launcher {
    pub fn new() -> Result<Launcher, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn run(&self, _case: &OpCase, _inputs: &[Tensor]) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn run_reps(
        &self,
        _case: &OpCase,
        _inputs: &[Tensor],
        _reps: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn run_quant(
        &self,
        _name: &str,
        _w: &[u8],
        _x: &[f32],
        _m: usize,
        _k: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn run_quant_reps(
        &self,
        _name: &str,
        _w: &[u8],
        _x: &[f32],
        _m: usize,
        _k: usize,
        _reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn run_quant_resident_reps(
        &self,
        _name: &str,
        _w: &hipml_hip::ByteBuffer,
        _x: &[f32],
        _m: usize,
        _k: usize,
        _reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_reps(
        &self,
        _name: &str,
        _w: &[u8],
        _x: &[f32],
        _m: usize,
        _k: usize,
        _tokens: usize,
        _reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    /// Fused gate+up parity launch. Same signature as the device one, so the
    /// `parity` command needs no conditional compilation.
    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_gu_reps(
        &self,
        _gate: &str,
        _up: &str,
        _wg: &[u8],
        _wu: &[u8],
        _x: &[f32],
        _m: usize,
        _k: usize,
        _tokens: usize,
        _reps: usize,
    ) -> Result<(Vec<f32>, Vec<f32>, f64), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_skinny_reps(
        &self,
        _name: &str,
        _w: &[u8],
        _x: &[f32],
        _n: usize,
        _k: usize,
        _r: usize,
        _reps: usize,
    ) -> Result<(Vec<f32>, f64), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn dequant_q4_k(
        &self,
        _w: &hipml_hip::ByteBuffer,
        _byte_offset: u64,
        _nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn dequant_q6_k(
        &self,
        _w: &hipml_hip::ByteBuffer,
        _byte_offset: u64,
        _nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn dequant_row_typed(
        &self,
        _ty: hipml_core::quant::TensorType,
        _w: &hipml_hip::ByteBuffer,
        _byte_offset: u64,
        _nblocks: usize,
    ) -> Result<Vec<f32>, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_prefill_time(
        &self,
        _name: &str,
        _w: &[u8],
        _x: &[f32],
        _m: usize,
        _k: usize,
        _tokens: usize,
        _reps: usize,
    ) -> Result<f64, String> {
        Err("hipml was built without the `hip` feature".to_string())
    }

    pub fn synchronize(&self) -> Result<(), String> {
        Err("hipml was built without the `hip` feature".to_string())
    }
}

/// Which inner loop the tiled prefill GEMM runs.
///
/// `F32` is the original kernel and the numerical reference. `Dot2` is the
/// packed-fp16 path (`v_dot2_f32_f16`; a register-only probe on this card puts
/// that instruction mix at 41.4 TFLOPS). `Wmma` is the matrix unit
/// (`wmma_f32_16x16x16_f16`, gfx12), which is what the quoted 95.7 TFLOPS
/// dense fp16 figure belongs to.
///
/// Lives outside the `hip` gate because the CLI and the parity runner pick an
/// entry point on both builds; without a device nothing is launched either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPath {
    F32,
    Dot2,
    Wmma,
}

impl PrefillPath {
    pub fn name(self) -> &'static str {
        match self {
            PrefillPath::F32 => "f32",
            PrefillPath::Dot2 => "fp16 dot2",
            PrefillPath::Wmma => "fp16 wmma",
        }
    }
}

/// Default is the matrix unit. The two environment switches are the
/// no-rebuild A/B hooks (`HIPML_PREFILL_DOT2`, `HIPML_PREFILL_F32`): all three
/// kernels stay in the code object, and the f32 one is the reference for a
/// divergence.
#[cfg(feature = "hip")]
static PREFILL_PATH: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(PrefillPath::Wmma as u8);

pub fn set_prefill_path(p: PrefillPath) {
    #[cfg(feature = "hip")]
    std::sync::atomic::AtomicU8::store(
        &PREFILL_PATH,
        p as u8,
        std::sync::atomic::Ordering::Relaxed,
    );
    #[cfg(not(feature = "hip"))]
    let _ = p;
}

pub fn prefill_path() -> PrefillPath {
    if std::env::var_os("HIPML_PREFILL_F32").is_some() {
        return PrefillPath::F32;
    }
    if std::env::var_os("HIPML_PREFILL_DOT2").is_some() {
        return PrefillPath::Dot2;
    }
    #[cfg(feature = "hip")]
    match std::sync::atomic::AtomicU8::load(&PREFILL_PATH, std::sync::atomic::Ordering::Relaxed) {
        0 => PrefillPath::F32,
        1 => PrefillPath::Dot2,
        _ => PrefillPath::Wmma,
    }
    #[cfg(not(feature = "hip"))]
    PrefillPath::Wmma
}

/// Kept for callers that only ask "is this an fp16 path" (the looser parity
/// budget): true unless the f32 kernel is selected.
pub fn set_prefill_half(on: bool) {
    set_prefill_path(if on {
        PrefillPath::Wmma
    } else {
        PrefillPath::F32
    });
}

pub fn prefill_half() -> bool {
    prefill_path() != PrefillPath::F32
}

/// Q5_K prefill route: the dequant-in-shared tile kernel (`prefill_q5_k`) is
/// the default. Setting `HIPML_Q5K_COPY_GB` to exactly `0` keeps the old
/// batched-GEMV fallback, so the two prefill routes can be measured without a
/// rebuild. The name is historical: there is no fp16 copy to budget any more.
pub fn q5_tile_prefill_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("HIPML_Q5K_COPY_GB")
            .ok()
            .map(|v| v.trim() != "0")
            .unwrap_or(true)
    })
}

/// The occupancy-shape prefill kernels (`prefill_wmma2.hip`). On by default on
/// the wmma path; `HIPML_PREFILL_V2=0` selects the shipped `prefill_wmma.hip`
/// shape. They keep the shipped kernel's k order, so switching between them is
/// a performance decision and not a numeric one.
pub fn prefill_v2() -> bool {
    static V2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V2.get_or_init(|| std::env::var_os("HIPML_PREFILL_V2").map_or(true, |v| v != "0"))
}

/// The entry point a base (f32) prefill kernel name maps to on the active path.
/// `None` means the f32 kernel is the one to launch.
pub fn prefill_entry_for(name: &str) -> Option<&'static str> {
    let w2 = prefill_v2();
    Some(match (prefill_path(), name) {
        (PrefillPath::F32, _) => return None,
        (PrefillPath::Dot2, "prefill_q4_k") => "prefill_q4_k_h",
        (PrefillPath::Dot2, "prefill_q6_k") => "prefill_q6_k_h",
        (PrefillPath::Dot2, "prefill_q8_0") => "prefill_q8_0_h",
        (PrefillPath::Dot2, "prefill_halfw") => "prefill_halfw_h",
        (PrefillPath::Wmma, "prefill_q4_k") if w2 => "prefill_q4_k_w2",
        (PrefillPath::Wmma, "prefill_q5_k") if w2 => "prefill_q5_k_w2",
        (PrefillPath::Wmma, "prefill_q6_k") if w2 => "prefill_q6_k_w2",
        (PrefillPath::Wmma, "prefill_q8_0") if w2 => "prefill_q8_0_w2",
        (PrefillPath::Wmma, "prefill_halfw") if w2 => "prefill_halfw_w2",
        (PrefillPath::Wmma, "prefill_q4_k") => "prefill_q4_k_w",
        (PrefillPath::Wmma, "prefill_q6_k") => "prefill_q6_k_w",
        (PrefillPath::Wmma, "prefill_q8_0") => "prefill_q8_0_w",
        (PrefillPath::Wmma, "prefill_halfw") => "prefill_halfw_w",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{gemm_rows_grid, prefill_slices};

    /// Every `(token, row)` output must get its own warp: the grid has to cover
    /// `tokens * m` warps. A launcher that treats the block count as a thread
    /// count (the WP42 bug) covers 256x fewer and leaves the tail of the SSM
    /// alpha/beta vectors unwritten.
    #[test]
    fn gemm_rows_grid_covers_every_output() {
        for (tokens, m) in [(1, 48), (1, 1), (4, 48), (169, 48), (2048, 48), (512, 5120)] {
            let (blocks, threads) = gemm_rows_grid(tokens, m);
            let warps = blocks * (threads / 32);
            assert!(
                warps >= tokens * m,
                "grid covers {warps} warps, needs {}",
                tokens * m
            );
            assert!(
                warps - tokens * m < 8,
                "grid over-launches {} warps for {} outputs",
                warps - tokens * m,
                tokens * m
            );
            assert_eq!(threads, 256);
        }
    }

    /// The SSM alpha/beta shape fills a wave; the ffn shapes that already have
    /// enough blocks must not be sliced (each slice would add a launch and a
    /// reduce for no work).
    #[test]
    fn prefill_slices_fills_a_wave_only_when_starved() {
        // 64 rows x a 2048-token chunk, k = 5120: 8 blocks -> 8 slices.
        assert_eq!(prefill_slices(64, 5120, 2048), 8);
        // 256 tokens: one block, still eight slices (8 blocks).
        assert_eq!(prefill_slices(64, 5120, 256), 8);
        // A 1024-token chunk is 4 blocks, eight slices still fit one wave.
        assert_eq!(prefill_slices(64, 5120, 1024), 8);
        // 128 rows x 2048 tokens is 8 blocks as well, and slicing still pays.
        assert_eq!(prefill_slices(128, 5120, 2048), 8);
        // k must divide into whole 64-value tiles; 5120 / 8 = 640 is one.
        for (m, k, tokens) in [(64, 5120, 2048), (64, 6144, 512), (256, 5120, 2048)] {
            let n = prefill_slices(m, k, tokens);
            assert!(n >= 1 && n <= 8);
            assert_eq!(k % n, 0, "k={k} n={n}");
            assert_eq!((k / n) % 64, 0, "k={k} n={n}");
        }
        // Enough blocks already (40 of a 64-block wave is not starved): no
        // split, so the ffn shapes keep their single-pass kernel.
        assert_eq!(prefill_slices(5120, 5120, 2048), 1);
    }
}
