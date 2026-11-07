//! Device-side dispatch: one match arm per kernel, argument lists written out.
//!
//! Shapes and scalars come from the parity case, so the launcher and the
//! reference are driven by exactly the same numbers. The argument arrays are
//! explicit because they are the kernel ABI: `hipModuleLaunchKernel` takes
//! pointers to the arguments, so every value needs a stable home in a local.

use hipml_core::parity::OpCase;
use hipml_core::tensor::Tensor;

#[cfg(feature = "hip")]
mod device {
    use super::*;
    use hipml_hip::{Buffer, ByteBuffer, Device, LaunchConfig, Module, Stream};
    use std::ffi::c_void;

    pub struct Launcher {
        module: Module,
        stream: Stream,
        pub device_name: String,
    }

    impl Launcher {
        pub fn new() -> Result<Launcher, String> {
            let dir = crate::kernel_dir()
                .ok_or_else(|| "kernels were not compiled: set HIP_PATH and rebuild".to_string())?;
            let device = Device::first()?;
            let module = Module::load(&dir.join("hipml.co"))?;
            let stream = Stream::new()?;
            Ok(Launcher {
                module,
                stream,
                device_name: device.name().to_string(),
            })
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
                "attn_decode_f32" => self.attention(case, inputs, reps),
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

        /// Launch a quantized GEMV once. `w` is the raw block payload, row-major.
        pub fn run_quant(
            &self,
            name: &str,
            w: &[u8],
            x: &[f32],
            m: usize,
            k: usize,
        ) -> Result<Vec<f32>, String> {
            self.run_quant_reps(name, w, x, m, k, 1).map(|(y, _)| y)
        }

        /// Launch a quantized GEMV `reps` times over one upload.
        ///
        /// Returns the output and the wall time spent inside the launch loop
        /// (uploads excluded), so the caller can report kernel bandwidth rather
        /// than PCIe bandwidth.
        pub fn run_quant_reps(
            &self,
            name: &str,
            w: &[u8],
            x: &[f32],
            m: usize,
            k: usize,
            reps: usize,
        ) -> Result<(Vec<f32>, f64), String> {
            assert!(reps > 0, "reps must be positive");
            let spec = crate::spec(name).ok_or_else(|| format!("{name} is not in the manifest"))?;
            let wb = ByteBuffer::from_host(w)?;
            let xb = Buffer::from_host(x)?;
            let y = Buffer::alloc(m)?;
            let kernel = self.module.kernel(name)?;

            let mut p_w = wb.as_ptr();
            let mut p_x = xb.as_ptr();
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
            let cfg = match spec.kind {
                crate::KernelKind::RowTile { rows, block } => {
                    LaunchConfig::row_tile(m, rows, block)
                }
                crate::KernelKind::PerRow { block } => LaunchConfig::per_row(m, block),
                _ => return Err(format!("{name}: unsupported quantized grid")),
            };
            let start = std::time::Instant::now();
            for _ in 0..reps {
                kernel.launch(cfg, &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            let secs = start.elapsed().as_secs_f64();
            Ok((y.to_host()?, secs))
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

            let mut params: [*mut c_void; 9] = [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_out as *mut _ as *mut c_void,
                &mut v_heads as *mut _ as *mut c_void,
                &mut v_kv as *mut _ as *mut c_void,
                &mut v_dim as *mut _ as *mut c_void,
                &mut v_seq as *mut _ as *mut c_void,
                &mut v_scale as *mut _ as *mut c_void,
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
pub use device::Launcher;

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
}
