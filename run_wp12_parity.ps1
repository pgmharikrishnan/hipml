$ErrorActionPreference = "Continue"
& C:\models\gpu-lock.ps1 acquire wp12
try {
  Set-Location C:\hipml-fullattn
  $env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
  $env:HIPML_ARCH = "gfx1201"
  $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;" + $env:PATH
  $exe = "target\release\hipml.exe"
  & $exe parity mrope_f32
  & $exe parity head_rmsnorm_f32
  & $exe parity qg_split_f32
  & $exe parity q_gate_f32
  & $exe parity kv_append_f32
  & $exe parity attn_decode_pm_f32
  & $exe attn-parity --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf
} finally {
  & C:\models\gpu-lock.ps1 release wp12
}
