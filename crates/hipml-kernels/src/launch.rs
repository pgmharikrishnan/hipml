//! Device-side dispatch: one match arm per kernel, argument lists written out.
//!
//! Shapes and scalars come from the parity case, so the launcher and the
//! reference are driven by exactly the same numbers. The argument arrays are
//! explicit because they are the kernel ABI: `hipModuleLaunchKernel` takes
//! pointers to the arguments, so every value needs a stable home in a local.

use hipml_core::parity::OpCase;
use hipml_core::tensor::Tensor;

#[cfg(feature = "hip")]
pub(crate) mod device {
    use super::*;
    use hipml_hip::{Buffer, ByteBuffer, Device, Event, LaunchConfig, Module, Stream};
    use std::ffi::c_void;

    #[path = "graph_launch.rs"]
    mod graph_launch;
    pub use graph_launch::{
        gemv_job_tiles, GemvJob, GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80,
    };
    #[path = "prefill_launch.rs"]
    mod prefill_launch;

    pub use prefill_launch::prefill_resident;
    #[path = "quant_launch.rs"]
    mod quant_launch;
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
            Ok(Launcher {
                module,
                stream,
                side,
                ev_in,
                ev_out,
                device_name: device.name().to_string(),
            })
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
                "qg_split_f32" | "head_rmsnorm_f32" | "mrope_f32" | "q_gate_f32"
                | "kv_append_f32" | "attn_decode_split_f32" => {
                    self.run_attn_case(case, inputs, reps)
                }
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

            let mut v_seq = seq as i32;

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
                &mut v_seq as *mut _ as *mut c_void,
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
            crate::launch_kv::run(&self.module, &self.stream, case, inputs, reps)
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
            for _ in 0..reps {
                self.kv_encode(dtype, &k, &v, &kc, &vc, n_kv, head_dim, seq, pos)?;
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
pub use device::prefill_resident;
#[cfg(feature = "hip")]
pub use device::{gemv_job_tiles, GemvJob, Launcher, ReadJob};
#[cfg(feature = "hip")]
pub use device::{GEMV_JOB_Q4K, GEMV_JOB_Q5K, GEMV_JOB_Q6K, GEMV_JOB_Q80};

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

    pub fn dequant_q4_k(
        &self,
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
