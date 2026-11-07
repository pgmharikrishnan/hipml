$ErrorActionPreference = "Continue"
& C:\models\gpu-lock.ps1 acquire wp12
try {
  Set-Location C:\hipml-fullattn
  $env:HIP_PATH = "C:\rocm-sdk\_rocm_sdk_core"
  $env:HIPML_ARCH = "gfx1201"
  $env:PATH = "C:\rocm-sdk\_rocm_sdk_core\bin;" + $env:PATH
  & target\release\hipml.exe attn-bench --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf --depths 0,8192,100000 --reps 7
} finally {
  & C:\models\gpu-lock.ps1 release wp12
}
