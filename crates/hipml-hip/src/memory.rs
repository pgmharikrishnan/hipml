//! Device memory: f32 buffers for activations, byte buffers for weights.
//!
//! No allocator lives here. The weight loader allocates one `ByteBuffer` per
//! GGUF tensor (see `hipml-model`) and activations are allocated per graph.

#[cfg(feature = "hip")]
use crate::device::check;
#[cfg(feature = "hip")]
use crate::ffi;

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
            // Make the upload complete before a caller starts timing a kernel
            // on a different stream; otherwise the copy competes for bandwidth.
            check(ffi::hipDeviceSynchronize(), "hipDeviceSynchronize")?;
        }
        Ok(buf)
    }

    #[cfg(not(feature = "hip"))]
    pub fn from_host(_data: &[f32]) -> Result<Buffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Copy host data into an existing device buffer.
    #[cfg(feature = "hip")]
    pub fn write(&self, data: &[f32]) -> Result<(), String> {
        assert_eq!(data.len(), self.len, "write length must match the buffer");
        unsafe {
            check(
                ffi::hipMemcpy(
                    self.ptr,
                    data.as_ptr() as *const std::ffi::c_void,
                    data.len() * 4,
                    ffi::HIP_MEMCPY_HOST_TO_DEVICE,
                ),
                "hipMemcpy host->device",
            )
        }
    }

    #[cfg(not(feature = "hip"))]
    pub fn write(&self, _data: &[f32]) -> Result<(), String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Zero every element.
    #[cfg(feature = "hip")]
    pub fn zero(&self) -> Result<(), String> {
        unsafe { check(ffi::hipMemset(self.ptr, 0, self.len * 4), "hipMemset") }
    }

    #[cfg(not(feature = "hip"))]
    pub fn zero(&self) -> Result<(), String> {
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

    /// Device address as a plain integer, for reports.
    pub fn device_ptr(&self) -> usize {
        #[cfg(feature = "hip")]
        {
            self.ptr as usize
        }
        #[cfg(not(feature = "hip"))]
        {
            0
        }
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
                check(ffi::hipDeviceSynchronize(), "hipDeviceSynchronize")?;
            }
        }
        Ok(buf)
    }

    #[cfg(not(feature = "hip"))]
    pub fn from_host(_data: &[u8]) -> Result<ByteBuffer, String> {
        Err("built without the `hip` feature".to_string())
    }

    /// Copy the whole payload back to host memory. This is the load verifier.
    #[cfg(feature = "hip")]
    pub fn to_host(&self) -> Result<Vec<u8>, String> {
        let mut out = vec![0u8; self.len];
        if self.len != 0 {
            unsafe {
                check(
                    ffi::hipMemcpy(
                        out.as_mut_ptr() as *mut std::ffi::c_void,
                        self.ptr,
                        self.len,
                        ffi::HIP_MEMCPY_DEVICE_TO_HOST,
                    ),
                    "hipMemcpy bytes device->host",
                )?;
                check(ffi::hipDeviceSynchronize(), "hipDeviceSynchronize")?;
            }
        }
        Ok(out)
    }

    #[cfg(not(feature = "hip"))]
    pub fn to_host(&self) -> Result<Vec<u8>, String> {
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

    /// Device address as a plain integer, for reports.
    pub fn device_ptr(&self) -> usize {
        #[cfg(feature = "hip")]
        {
            self.ptr as usize
        }
        #[cfg(not(feature = "hip"))]
        {
            0
        }
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
