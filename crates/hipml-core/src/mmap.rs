//! A read-only memory map of the GGUF file.
//!
//! The weights are 17 GB. Holding them in an anonymous `Vec<u8>` makes them
//! swap candidates and doubles the residency (a copy plus the page cache), which
//! on a shared machine turns every token into a page-fault storm. Mapping the
//! file instead keeps pages file-backed: the kernel can evict them for free and
//! re-read from the SSD, and the same pages serve as the page cache.
//!
//! Unix only. Other platforms fall back to reading each tensor into memory
//! (`Mat::load`), which is correct but needs the model to fit.
//!
//! Nothing is mutated through the map; `Map` is `Send + Sync` because the
//! mapping is read-only for its whole lifetime.

#[cfg(unix)]
use std::fs::File;
use std::io;
use std::path::Path;

/// A read-only mapping of a whole file.
pub struct Map {
    #[cfg(unix)]
    ptr: *mut u8,
    #[cfg(unix)]
    len: usize,
    /// Fallback storage where mmap is unavailable.
    #[cfg(not(unix))]
    bytes: Vec<u8>,
}

#[cfg(unix)]
mod sys {
    extern "C" {
        pub fn mmap(
            addr: *mut u8,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut u8;
        pub fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
    pub const MAP_FAILED: *mut u8 = usize::MAX as *mut u8;
}

impl Map {
    pub fn open(path: impl AsRef<Path>) -> io::Result<Map> {
        let path = path.as_ref();
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let f = File::open(path)?;
            let len = f.metadata()?.len() as usize;
            let ptr = unsafe {
                sys::mmap(
                    std::ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    f.as_raw_fd(),
                    0,
                )
            };
            if ptr == sys::MAP_FAILED || ptr.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Map { ptr, len })
        }
        #[cfg(not(unix))]
        {
            Ok(Map {
                bytes: std::fs::read(path)?,
            })
        }
    }

    pub fn len(&self) -> usize {
        #[cfg(unix)]
        {
            self.len
        }
        #[cfg(not(unix))]
        {
            self.bytes.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole file. Panics if `range` is out of bounds.
    pub fn slice(&self, start: usize, len: usize) -> &[u8] {
        assert!(
            start.checked_add(len).map(|e| e <= self.len()).unwrap_or(false),
            "map slice {start}+{len} outside {} bytes",
            self.len()
        );
        #[cfg(unix)]
        unsafe {
            std::slice::from_raw_parts(self.ptr.add(start), len)
        }
        #[cfg(not(unix))]
        {
            &self.bytes[start..start + len]
        }
    }
}

#[cfg(unix)]
impl Drop for Map {
    fn drop(&mut self) {
        unsafe {
            sys::munmap(self.ptr, self.len);
        }
    }
}

// The mapping is read-only and shared; slices borrow it for their lifetime.
#[cfg(unix)]
unsafe impl Send for Map {}
#[cfg(unix)]
unsafe impl Sync for Map {}
