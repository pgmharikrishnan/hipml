//! Single-op parity launchers for the attention kernels.
//!
//! The layer itself lives in `attn_gpu.rs`; these entry points exist so every
//! kernel in the manifest has a case the parity harness can run standalone.

#[cfg(feature = "hip")]
mod imp {
    use crate::launch::device::Launcher;
    use hipml_core::parity::OpCase;
    use hipml_core::tensor::Tensor;
    use hipml_hip::{Buffer, LaunchConfig};
    use std::ffi::c_void;

    // ---- single-op parity cases -------------------------------------------

    impl Launcher {
        /// Wait for every launch queued on the launcher's stream.
        pub fn synchronize(&self) -> Result<(), String> {
            self.stream.synchronize()
        }

        pub fn run_attn_case(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            match case.name {
                "qg_split_f32" => self.case_qg_split(case, inputs, reps),
                "head_rmsnorm_f32" => self.case_head_norm(case, inputs, reps),
                "mrope_f32" => self.case_mrope(case, inputs, reps),
                "q_gate_f32" => self.case_q_gate(case, inputs, reps),
                "kv_append_f32" => self.case_kv_append(case, inputs, reps),
                "attn_decode_split_f32" => self.case_attn_split(case, inputs, reps),
                other => Err(format!("{other} has no attention launcher")),
            }
        }

        fn case_qg_split(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let nh = case.scalar("n_heads") as usize;
            let hd = case.scalar("head_dim") as usize;
            let src = Buffer::from_host(inputs[0].data())?;
            let out = Buffer::alloc(2 * nh * hd)?;
            let kernel = self.module.kernel("qg_split_f32")?;
            let mut p_src = src.as_ptr();
            let mut p_q = out.as_ptr();
            let mut p_g = unsafe { (out.as_ptr() as *mut f32).add(nh * hd) as *mut c_void };
            let mut v_nh = nh as i32;
            let mut v_hd = hd as i32;
            let mut params: [*mut c_void; 5] = [
                &mut p_src as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut v_hd as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (nh as u32, 1, 1),
                block: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            for _ in 0..reps {
                kernel.launch(cfg, &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            out.to_host()
        }

        fn case_head_norm(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let nh = case.scalar("n_heads") as usize;
            let hd = case.scalar("head_dim") as usize;
            let eps = case.scalar("eps");
            let x = Buffer::from_host(inputs[0].data())?;
            let w = Buffer::from_host(inputs[1].data())?;
            let y = Buffer::alloc(nh * hd)?;
            let kernel = self.module.kernel("head_rmsnorm_f32")?;
            let mut p_x = x.as_ptr();
            let mut p_w = w.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_nh = nh as i32;
            let mut v_hd = hd as i32;
            let mut v_eps = eps;
            let mut params: [*mut c_void; 6] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut v_hd as *mut _ as *mut c_void,
                &mut v_eps as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(LaunchConfig::per_row(nh, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn case_mrope(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let nh = case.scalar("n_heads") as usize;
            let hd = case.scalar("head_dim") as usize;
            let x = Buffer::from_host(inputs[0].data())?;
            let y = Buffer::alloc(nh * hd)?;
            let kernel = self.module.kernel("mrope_f32")?;
            let mut p_x = x.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_nh = nh as i32;
            let mut v_hd = hd as i32;
            let mut v_nrot = case.scalar("n_rot") as i32;
            let mut s0 = case.scalar("s0") as i32;
            let mut s1 = case.scalar("s1") as i32;
            let mut s2 = case.scalar("s2") as i32;
            let mut s3 = case.scalar("s3") as i32;
            let mut pos = case.scalar("pos") as i32;
            let mut base = case.scalar("theta_base");
            let mut params: [*mut c_void; 14] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut v_hd as *mut _ as *mut c_void,
                &mut v_nrot as *mut _ as *mut c_void,
                &mut s0 as *mut _ as *mut c_void,
                &mut s1 as *mut _ as *mut c_void,
                &mut s2 as *mut _ as *mut c_void,
                &mut s3 as *mut _ as *mut c_void,
                &mut pos as *mut _ as *mut c_void,
                &mut pos as *mut _ as *mut c_void,
                &mut pos as *mut _ as *mut c_void,
                &mut pos as *mut _ as *mut c_void,
                &mut base as *mut _ as *mut c_void,
            ];
            let cfg = LaunchConfig {
                grid: (nh as u32, 1, 1),
                block: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            for _ in 0..reps {
                kernel.launch(cfg, &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn case_q_gate(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n = case.scalar("n") as usize;
            let x = Buffer::from_host(inputs[0].data())?;
            let g = Buffer::from_host(inputs[1].data())?;
            let y = Buffer::alloc(n)?;
            let kernel = self.module.kernel("q_gate_f32")?;
            let mut p_x = x.as_ptr();
            let mut p_g = g.as_ptr();
            let mut p_y = y.as_ptr();
            let mut v_n = n as i32;
            let mut params: [*mut c_void; 4] = [
                &mut p_x as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_y as *mut _ as *mut c_void,
                &mut v_n as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(LaunchConfig::linear(n, 256), &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            y.to_host()
        }

        fn case_kv_append(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n_ctx = case.scalar("n_ctx") as usize;
            let width = case.scalar("width") as usize;
            let off = case.scalar("off") as usize;
            let mut host = inputs[0].data().to_vec();
            host.extend_from_slice(inputs[1].data());
            let cache = Buffer::from_host(&host)?;
            let k = Buffer::from_host(inputs[2].data())?;
            let v = Buffer::from_host(inputs[3].data())?;
            let kernel = self.module.kernel("kv_append_f32")?;
            let mut p_kc = cache.as_ptr();
            let mut p_vc =
                unsafe { (cache.as_ptr() as *mut f32).add(n_ctx * width) as *mut c_void };
            let mut p_k = k.as_ptr();
            let mut p_v = v.as_ptr();
            let mut v_off = off as i32;
            let mut v_w = width as i32;
            let mut params: [*mut c_void; 6] = [
                &mut p_kc as *mut _ as *mut c_void,
                &mut p_vc as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut v_off as *mut _ as *mut c_void,
                &mut v_w as *mut _ as *mut c_void,
            ];
            for _ in 0..reps {
                kernel.launch(
                    LaunchConfig::linear(2 * width, 256),
                    &self.stream,
                    &mut params,
                )?;
            }
            self.stream.synchronize()?;
            cache.to_host()
        }

        /// Split-K attention + merge, position-major cache.
        fn case_attn_split(
            &self,
            case: &OpCase,
            inputs: &[Tensor],
            reps: usize,
        ) -> Result<Vec<f32>, String> {
            let n_heads = case.scalar("n_heads") as usize;
            let n_kv = case.scalar("n_kv") as usize;
            let hd = case.scalar("head_dim") as usize;
            let seq = case.scalar("seq") as usize;
            let scale = case.scalar("scale");
            let splits = case.scalar("splits") as usize;
            let q = Buffer::from_host(inputs[0].data())?;
            let k = Buffer::from_host(inputs[1].data())?;
            let v = Buffer::from_host(inputs[2].data())?;
            let part_acc = Buffer::alloc(n_heads * splits * hd)?;
            let part_ml = Buffer::alloc(n_heads * splits * 2)?;
            let out = Buffer::alloc(n_heads * hd)?;

            let split = self.module.kernel("attn_decode_split_f32")?;
            let reduce = self.module.kernel("attn_reduce_f32")?;
            let mut p_q = q.as_ptr();
            let mut p_k = k.as_ptr();
            let mut p_v = v.as_ptr();
            let mut p_acc = part_acc.as_ptr();
            let mut p_ml = part_ml.as_ptr();
            let mut v_nh = n_heads as i32;
            let mut v_kv = n_kv as i32;
            let mut v_hd = hd as i32;
            let mut v_seq = seq as i32;
            let mut v_scale = scale;
            let mut v_splits = splits as i32;
            let mut width = (n_kv * hd) as i32;
            let mut params: [*mut c_void; 12] = [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_v as *mut _ as *mut c_void,
                &mut p_acc as *mut _ as *mut c_void,
                &mut p_ml as *mut _ as *mut c_void,
                &mut v_nh as *mut _ as *mut c_void,
                &mut v_kv as *mut _ as *mut c_void,
                &mut v_hd as *mut _ as *mut c_void,
                &mut v_seq as *mut _ as *mut c_void,
                &mut v_scale as *mut _ as *mut c_void,
                &mut v_splits as *mut _ as *mut c_void,
                &mut width as *mut _ as *mut c_void,
            ];
            let group = n_heads / n_kv;
            let cfg = LaunchConfig {
                grid: ((n_kv * splits) as u32, 1, 1),
                block: ((group * 32) as u32, 1, 1),
                shared_mem_bytes: (2 * 8 * hd * 4) as u32,
            };
            for _ in 0..reps {
                split.launch(cfg, &self.stream, &mut params)?;
                let mut p_acc = part_acc.as_ptr();
                let mut p_ml = part_ml.as_ptr();
                let mut p_out = out.as_ptr();
                let mut params: [*mut c_void; 5] = [
                    &mut p_acc as *mut _ as *mut c_void,
                    &mut p_ml as *mut _ as *mut c_void,
                    &mut p_out as *mut _ as *mut c_void,
                    &mut v_hd as *mut _ as *mut c_void,
                    &mut v_splits as *mut _ as *mut c_void,
                ];
                let cfg = LaunchConfig {
                    grid: (n_heads as u32, 1, 1),
                    block: (hd as u32, 1, 1),
                    shared_mem_bytes: 0,
                };
                reduce.launch(cfg, &self.stream, &mut params)?;
            }
            self.stream.synchronize()?;
            out.to_host()
        }
    }
}
