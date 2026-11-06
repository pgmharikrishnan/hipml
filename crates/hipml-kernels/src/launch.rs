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
    use hipml_hip::{Buffer, Device, LaunchConfig, Module, Stream};
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
                "rmsnorm_f32" => self.rmsnorm(case, inputs, reps),
                "rope_neox_f32" => self.rope(case, inputs, reps),
                "silu_mul_f32" => self.linear2(case, inputs, reps),
                "residual_add_f32" => self.linear2(case, inputs, reps),
                "softmax_f32" => self.softmax(case, inputs, reps),
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
}
