//! The elementwise launches the decode graph chains between its GEMVs.
//!
//! Every method here works on device buffers and does not synchronize; the
//! graph issues a whole token and syncs once. Argument arrays are written out
//! explicitly because they are the kernel ABI.

use super::*;

impl Launcher {
    /// `y = x / sqrt(mean(x^2) + eps) * w`, one block per row (row count 1 here).
    pub fn rmsnorm_buf(
        &self,
        x: &Buffer,
        w: &Buffer,
        y: &Buffer,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
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
        kernel.launch(LaunchConfig::per_row(1, 256), &self.stream, &mut params)
    }

    /// `y = a + b`, one thread per element.
    pub fn residual_buf(&self, a: &Buffer, b: &Buffer, y: &Buffer, n: usize) -> Result<(), String> {
        self.linear3("residual_add_f32", a, b, y, n)
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
}
