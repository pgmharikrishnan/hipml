// SPDX-License-Identifier: Apache-2.0
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
pub(crate) fn check(code: ffi::HipError, what: &str) -> Result<(), String> {
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

    /// Raw handle, for the async copy in `memory.rs`.
    #[cfg(feature = "hip")]
    pub(crate) fn handle(&self) -> ffi::HipStream {
        self.handle
    }

    /// Block this stream until `event` has completed. Ordering only: the work
    /// is not copied or flushed.
    #[cfg(feature = "hip")]
    pub fn wait(&self, event: &Event) -> Result<(), String> {
        unsafe {
            check(
                ffi::hipStreamWaitEvent(self.handle, event.handle, 0),
                "hipStreamWaitEvent",
            )
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn wait(&self, _event: &Event) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(not(feature = "hip"))]
    pub fn synchronize(&self) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Start recording every launch on this stream into a graph instead of
    /// executing it. Nothing runs until the graph is launched.
    #[cfg(feature = "hip")]
    pub fn begin_capture(&self) -> Result<(), String> {
        unsafe {
            check(
                ffi::hipStreamBeginCapture(self.handle, HIP_CAPTURE_MODE_GLOBAL),
                "hipStreamBeginCapture",
            )
        }
    }

    /// Stop recording and hand back the executable graph.
    #[cfg(feature = "hip")]
    pub fn end_capture(&self) -> Result<Graph, String> {
        unsafe {
            let mut graph: ffi::HipGraph = std::ptr::null_mut();
            check(
                ffi::hipStreamEndCapture(self.handle, &mut graph),
                "hipStreamEndCapture",
            )?;
            let mut exec: ffi::HipGraphExec = std::ptr::null_mut();
            check(
                ffi::hipGraphInstantiate(&mut exec, graph, std::ptr::null_mut(), std::ptr::null_mut(), 0),
                "hipGraphInstantiate",
            )?;
            Ok(Graph {
                handle: graph,
                exec,
            })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn begin_capture(&self) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(not(feature = "hip"))]
    pub fn end_capture(&self) -> Result<Graph, String> {
        Err("built without the `hip` feature".to_string())
    }
}

/// `hipStreamCaptureModeGlobal`: no other thread may break the capture.
#[cfg(feature = "hip")]
const HIP_CAPTURE_MODE_GLOBAL: i32 = 0;

/// An executable graph: what a captured stream of launches becomes, so one
/// `hipGraphLaunch` submits the whole token instead of one call per kernel.
pub struct Graph {
    #[cfg(feature = "hip")]
    handle: ffi::HipGraph,
    #[cfg(feature = "hip")]
    exec: ffi::HipGraphExec,
}

impl Graph {
    /// Submit the whole graph. The nodes keep the arguments they were captured
    /// with, so anything that changes per step has to live in device memory.
    #[cfg(feature = "hip")]
    pub fn launch(&self, stream: &Stream) -> Result<(), String> {
        unsafe { check(ffi::hipGraphLaunch(self.exec, stream.handle), "hipGraphLaunch") }
    }

    /// Build the device-side launch state ahead of time; the first `launch`
    /// then does not stall on upload.
    #[cfg(feature = "hip")]
    pub fn upload(&self, stream: &Stream) -> Result<(), String> {
        unsafe { check(ffi::hipGraphUpload(self.exec, stream.handle), "hipGraphUpload") }
    }

    #[cfg(not(feature = "hip"))]
    pub fn launch(&self, _stream: &Stream) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(not(feature = "hip"))]
    pub fn upload(&self, _stream: &Stream) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }
}

#[cfg(feature = "hip")]
impl Drop for Graph {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipGraphExecDestroy(self.exec);
            let _ = ffi::hipGraphDestroy(self.handle);
        }
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

/// A timing event. Recorded on a stream and read back after a synchronize, so
/// a caller can say whether time went inside a kernel or between launches.
pub struct Event {
    #[cfg(feature = "hip")]
    handle: ffi::HipEvent,
}

impl Event {
    #[cfg(feature = "hip")]
    pub fn new() -> Result<Event, String> {
        unsafe {
            let mut handle: ffi::HipEvent = std::ptr::null_mut();
            check(ffi::hipEventCreate(&mut handle), "hipEventCreate")?;
            Ok(Event { handle })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn new() -> Result<Event, String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(feature = "hip")]
    pub fn record(&self, stream: &Stream) -> Result<(), String> {
        unsafe { check(ffi::hipEventRecord(self.handle, stream.handle), "hipEventRecord") }
    }

    #[cfg(not(feature = "hip"))]
    pub fn record(&self, _stream: &Stream) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Milliseconds from `self` (recorded earlier) to `stop`, in ms.
    #[cfg(feature = "hip")]
    pub fn since(&self, stop: &Event) -> Result<f32, String> {
        let mut ms = 0.0f32;
        unsafe {
            check(
                ffi::hipEventElapsedTime(&mut ms, self.handle, stop.handle),
                "hipEventElapsedTime",
            )?;
        }
        Ok(ms)
    }

    #[cfg(not(feature = "hip"))]
    pub fn since(&self, _stop: &Event) -> Result<f32, String> {
        Err("built without the `hip` feature".to_string())
    }
}

#[cfg(feature = "hip")]
impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::hipEventDestroy(self.handle);
        }
    }
}

/// A loaded code object (`.hsaco` / `.co`).
pub struct Module {
    #[cfg(feature = "hip")]
    handle: ffi::HipModule,
    /// Entry points already looked up. `hipModuleGetFunction` measured 2.3 us
    /// per call on this driver, and the decode graph asks for the same names
    /// 1299 times a token, so the lookup is cached here rather than at each
    /// call site.
    #[cfg(feature = "hip")]
    functions: std::cell::RefCell<std::collections::HashMap<String, ffi::HipFunction>>,
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
            Ok(Module {
                handle,
                functions: std::cell::RefCell::new(std::collections::HashMap::new()),
            })
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn load(_path: &std::path::Path) -> Result<Module, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Look up a kernel by name. The returned handle borrows the module.
    #[cfg(feature = "hip")]
    pub fn kernel(&self, name: &str) -> Result<Kernel<'_>, String> {
        if let Some(&function) = self.functions.borrow().get(name) {
            return Ok(Kernel::from_function(function));
        }
        let c_name = std::ffi::CString::new(name).map_err(|e| format!("bad kernel name: {e}"))?;
        unsafe {
            let mut function: ffi::HipFunction = std::ptr::null_mut();
            check(
                ffi::hipModuleGetFunction(&mut function, self.handle, c_name.as_ptr()),
                "hipModuleGetFunction",
            )?;
            self.functions
                .borrow_mut()
                .insert(name.to_string(), function);
            Ok(Kernel::from_function(function))
        }
    }

    /// How many entry points have been resolved (the cache's size).
    #[cfg(feature = "hip")]
    pub fn resolved(&self) -> usize {
        self.functions.borrow().len()
    }

    #[cfg(not(feature = "hip"))]
    pub fn kernel(&self, _name: &str) -> Result<Kernel<'_>, String> {
        Err("built without the `hip` feature".to_string())
    }

    #[cfg(not(feature = "hip"))]
    pub fn resolved(&self) -> usize {
        0
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

impl Kernel<'_> {
    /// Wrap an already-resolved entry point (see `Module::kernel`'s cache).
    #[cfg(feature = "hip")]
    fn from_function(function: ffi::HipFunction) -> Kernel<'static> {
        Kernel {
            function,
            _module: PhantomData,
        }
    }
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

    /// One block per `rows_per_block` output rows; the kernel stages its x tile
    /// in statically sized shared memory, so `shared_mem_bytes` stays zero.
    pub fn row_tile(rows: usize, rows_per_block: u32, block: u32) -> Self {
        let rpb = rows_per_block.max(1) as usize;
        let blocks = ((rows + rpb - 1) / rpb).max(1);
        Self {
            grid: (blocks as u32, 1, 1),
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

    /// One block per `rows_per_block` x `tokens_per_block` output tile.
    pub fn prefill(
        rows: usize,
        tokens: usize,
        rows_per_block: u32,
        tokens_per_block: u32,
        block: u32,
    ) -> Self {
        let rpb = rows_per_block.max(1) as usize;
        let tpb = tokens_per_block.max(1) as usize;
        Self {
            grid: (
                ((rows + rpb - 1) / rpb).max(1) as u32,
                ((tokens + tpb - 1) / tpb).max(1) as u32,
                1,
            ),
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
