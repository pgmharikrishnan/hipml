//! Raw `amdhip64` declarations — the only place hipml touches the vendor API.
//!
//! Deliberately narrow: no opaque property structs (their layout is a version
//! trap), no helper headers, no rocBLAS yet. Everything here is a flat function
//! with C types, so the surface stays auditable.

#![cfg(feature = "hip")]

use std::ffi::{c_char, c_int, c_void};

pub type HipError = c_int;
pub type HipDevice = c_int;
pub type HipStream = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;

pub const HIP_SUCCESS: HipError = 0;

/// `hipMemcpyKind`
pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const HIP_MEMCPY_DEVICE_TO_DEVICE: c_int = 3;

#[link(name = "amdhip64")]
extern "C" {
    pub fn hipInit(flags: u32) -> HipError;
    pub fn hipGetDeviceCount(count: *mut c_int) -> HipError;
    pub fn hipGetDevice(device: *mut HipDevice, ordinal: c_int) -> HipError;
    pub fn hipSetDevice(device: HipDevice) -> HipError;
    pub fn hipDeviceGetName(name: *mut c_char, len: c_int, device: HipDevice) -> HipError;
    pub fn hipDeviceTotalMem(bytes: *mut usize, device: HipDevice) -> HipError;
    pub fn hipDeviceComputeCapability(
        major: *mut c_int,
        minor: *mut c_int,
        device: HipDevice,
    ) -> HipError;
    pub fn hipGetErrorString(code: HipError) -> *const c_char;

    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> HipError;
    pub fn hipFree(ptr: *mut c_void) -> HipError;
    pub fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> HipError;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> HipError;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> HipError;
    pub fn hipDeviceSynchronize() -> HipError;

    pub fn hipStreamCreate(stream: *mut HipStream) -> HipError;
    pub fn hipStreamDestroy(stream: HipStream) -> HipError;
    pub fn hipStreamSynchronize(stream: HipStream) -> HipError;

    pub fn hipModuleLoad(module: *mut HipModule, file: *const c_char) -> HipError;
    pub fn hipModuleUnload(module: HipModule) -> HipError;
    pub fn hipModuleGetFunction(
        function: *mut HipFunction,
        module: HipModule,
        name: *const c_char,
    ) -> HipError;
    pub fn hipModuleLaunchKernel(
        function: HipFunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_mem_bytes: u32,
        stream: HipStream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
}
