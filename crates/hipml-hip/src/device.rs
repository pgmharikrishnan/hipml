//! A thin, explicit device layer: memory, streams, modules, launches.
//!
//! There is no allocator abstraction here on purpose. The caching allocator and
//! the weight arena sit above this (see `docs/architecture.md`); this file only
//! knows how to allocate, copy and launch.

use std::marker::PhantomData;

#[cfg(feature = "hip")]
use crate::ffi;

/// One HIP device.
#[derive(Clone, Debug)]
pub struct Device {
    ordinal: i32,
    name: String,
    total_mem_bytes: usize,
    compute_capability: (i32, i32),
}

impl Device {
    /// Enumerate devices. Without the `hip` feature this is an error, not a
    /// silent zero, so callers can distinguish "no GPU" from "not built for one".
    #[cfg(feature = "hip")]
    pub fn probe() -> Result<Vec<Device>, String> {
        unsafe {
            check(ffi::hipInit(0), "hipInit")?;
            let mut count: i32 = 0;
            check(ffi::hipGetDeviceCount(&mut count), "hipGetDeviceCount")?;
            let mut devices = Vec::with_capacity(count.max(0) as usize);
            for ordinal in 0..count {
                let mut handle: ffi::HipDevice = 0;
                check(ffi::hipGetDevice(&mut handle, ordinal), "hipGetDevice")?;
                check(ffi::hipSetDevice(handle), "hipSetDevice")?;

                let mut buf = [0i8; 128];
                check(
                    ffi::hipDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, handle),
                    "hipDeviceGetName",
                )?;
                let name = std::ffi::CStr::from_ptr(buf.as_ptr())
                    .to_string_lossy()
                    .into_owned();

                let mut total: usize = 0;
                check(
                    ffi::hipDeviceTotalMem(&mut total, handle),
                    "hipDeviceTotalMem",
                )?;

                let (mut major, mut minor) = (0i32, 0i32);
                check(
                    ffi::hipDeviceComputeCapability(&mut major, &mut minor, handle),
                    "hipDeviceComputeCapability",
                )?;

                devices.push(Device {
                    ordinal,
                    name,
                    total_mem_bytes: total,
                    compute_capability: (major, minor),
                });
            }
            Ok(devices)
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn probe() -> Result<Vec<Device>, String> {
        Err("hipml-hip was built without the `hip` feature".to_string())
    }

    /// The first device, or a readable error.
    pub fn first() -> Result<Device, String> {
        Self::probe()?
            .into_iter()
            .next()
            .ok_or_else(|| "no HIP devices found".to_string())
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn total_mem_bytes(&self) -> usize {
        self.total_mem_bytes
    }

    /// Roughly `(12, 1)` on gfx1201.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// `(free, total)` in bytes right now.
    #[cfg(feature = "hip")]
    pub fn mem_info(&self) -> Result<(usize, usize), String> {
        unsafe {
            let (mut free, mut total) = (0usize, 0usize);
            check(ffi::hipMemGetInfo(&mut free, &mut total), "hipMemGetInfo")?;
            Ok((free, total))
        }
    }
}

#[cfg(feature = "hip")]
fn check(code: ffi::HipError, what: &str) -> Result<(), String> {
    if code == ffi::HIP_SUCCESS {
        return Ok(());
    }
    let text = unsafe {
        let p = ffi::hipGetErrorString(code);
        if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    Err(format!("{what} failed: {code} ({text})"))
}

/// A device allocation of f32s. Frees on drop.
pub struct Buffer {
    #[cfg(feature = "hip")]
    ptr: *mut std::ffi::c_void,
    len: usize,
}

// The handle is owned and only touched from the owning thread's stream today.
unsafe impl Send for Buffer {}

impl Buffer {
    #[cfg(feature = "hip")]
    pub fn alloc(len: usize) -> Result<Buffer, String> {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            check(ffi::hipMalloc(&mut ptr, len * 4), "hipMalloc")?;
            Ok(Buffer { ptr, len })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn alloc(_len: usize) -> Result<Buffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Upload host data.
    #[cfg(feature = "hip")]
    pub fn from_host(data: &[f32]) -> Result<Buffer, String> {
        let buf = Buffer::alloc(data.len())?;
        unsafe {
            check(
                ffi::hipMemcpy(
                    buf.ptr,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len() * 4,
                    ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "hipMemcpy host->device",
            )?;
        }
        Ok(buf)
    }

    #[cfg(not(feature = "hip"))]
    pub fn from_host(_data: &[f32]) -> Result<Buffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Download into host memory.
    #[cfg(feature = "hip")]
    pub fn to_host(&self) -> Result<Vec<f32>, String> {
        let mut out = vec![0.0f32; self.len];
        unsafe {
            check(
                ffi::hipMemcpy(
                    out.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr,
                    self.len * 4,
                    ffi::HIP_MEMCPY_DEVICE_TO_HOST,
                ),
                "hipMemcpy device->host",
            )?;
        }
        Ok(out)
    }

    #[cfg(not(feature = "hip"))]
    pub fn to_host(&self) -> Result<Vec<f32>, String> {
        Err("built without the `hip` feature".to_string())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Device pointer as a kernel argument.
    #[cfg(feature = "hip")]
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

#[cfg(feature = "hip")]
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipFree(self.ptr);
        }
    }
}

/// A device allocation of raw bytes, for quantized weight blocks.
///
/// Separate from `Buffer` because `Buffer` counts f32 elements; a Q4_K payload
/// is not a multiple of four bytes and must not be rounded.
pub struct ByteBuffer {
    #[cfg(feature = "hip")]
    ptr: *mut std::ffi::c_void,
    len: usize,
}

unsafe impl Send for ByteBuffer {}

impl ByteBuffer {
    #[cfg(feature = "hip")]
    pub fn alloc(len: usize) -> Result<ByteBuffer, String> {
        unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            check(ffi::hipMalloc(&mut ptr, len), "hipMalloc")?;
            Ok(ByteBuffer { ptr, len })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn alloc(_len: usize) -> Result<ByteBuffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(feature = "hip")]
    pub fn from_host(data: &[u8]) -> Result<ByteBuffer, String> {
        let buf = ByteBuffer::alloc(data.len())?;
        if !data.is_empty() {
            unsafe {
                check(
                    ffi::hipMemcpy(
                        buf.ptr,
                        data.as_ptr() as *const std::ffi::c_void,
                        data.len(),
                        ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                    ),
                    "hipMemcpy bytes host->device",
                )?;
            }
        }
        Ok(buf)
    }

    #[cfg(not(feature = "hip"))]
    pub fn from_host(_data: &[u8]) -> Result<ByteBuffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(feature = "hip")]
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }
}

#[cfg(feature = "hip")]
impl Drop for ByteBuffer {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipFree(self.ptr);
        }
    }
}

/// A stream. Kernels on one stream run in order.
pub struct Stream {
    #[cfg(feature = "hip")]
    handle: ffi::HipStream,
}

impl Stream {
    #[cfg(feature = "hip")]
    pub fn new() -> Result<Stream, String> {
        unsafe {
            let mut handle: ffi::HipStream = std::ptr::null_mut();
            check(ffi::hipStreamCreate(&mut handle), "hipStreamCreate")?;
            Ok(Stream { handle })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn new() -> Result<Stream, String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(feature = "hip")]
    pub fn synchronize(&self) -> Result<(), String> {
        unsafe {
            check(
                ffi::hipStreamSynchronize(self.handle),
                "hipStreamSynchronize",
            )
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn synchronize(&self) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }
}

#[cfg(feature = "hip")]
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipStreamDestroy(self.handle);
        }
    }
}

/// A loaded code object (`.hsaco` / `.co`).
pub struct Module {
    #[cfg(feature = "hip")]
    handle: ffi::HipModule,
}

impl Module {
    #[cfg(feature = "hip")]
    pub fn load(path: &std::path::Path) -> Result<Module, String> {
        let c_path = std::ffi::CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("path contains a nul byte: {e}"))?;
        unsafe {
            let mut handle: ffi::HipModule = std::ptr::null_mut();
            check(
                ffi::hipModuleLoad(&mut handle, c_path.as_ptr()),
                "hipModuleLoad",
            )?;
            Ok(Module { handle })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn load(_path: &std::path::Path) -> Result<Module, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Look up a kernel by name. The returned handle borrows the module.
    #[cfg(feature = "hip")]
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_>, String> {
        let c_name = std::ffi::CString::new(name).map_err(|e| format!("bad kernel name: {e}"))?;
        unsafe {
            let mut function: ffi::HipFunction = std::ptr::null_mut();
            check(
                ffi::hipModuleGetFunction(&mut function, self.handle, c_name.as_ptr()),
                "hipModuleGetFunction",
            )?;
            Ok(Kernel {
                function,
                _module: PhantomData,
            })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn kernel(&self, _name: &str) -> Result<Kernel<'_>, String> {
        Err("built without the `hip` feature".to_string())
    }
}

#[cfg(feature = "hip")]
impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipModuleUnload(self.handle);
        }
    }
}

/// A kernel handle. Launch parameters are passed as an array of pointers to the
/// arguments, matching `hipModuleLaunchKernel`.
pub struct Kernel<'a> {
    #[cfg(feature = "hip")]
    function: ffi::HipFunction,
    _module: PhantomData<&'a Module>,
}

/// Grid and block geometry.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    /// One block per output row, `block` threads in x.
    pub fn per_row(rows: usize, block: u32) -> Self {
        Self {
            grid: (rows.max(1) as u32, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// 1-D grid covering `n` elements.
    pub fn linear(n: usize, block: u32) -> Self {
        let blocks = ((n + block as usize - 1) / block as usize).max(1);
        Self {
            grid: (blocks as u32, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl<'a> Kernel<'a> {
    #[cfg(feature = "hip")]
    pub fn launch(
        &self,
        cfg: LaunchConfig,
        stream: &Stream,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        unsafe {
            check(
                ffi::hipModuleLaunchKernel(
                    self.function,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_mem_bytes,
                    stream.handle,
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "hipModuleLaunchKernel",
            )
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn launch(
        &self,
        _cfg: LaunchConfig,
        _stream: &Stream,
        _params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }
}
