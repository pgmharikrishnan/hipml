@echo off
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set MODEL=C:\models\Qwen3.8-27B-Q4_K_M.gguf
set HIPML=C:\Users\Hari\hipml-skinny\target\release\hipml.exe

powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp33
if errorlevel 1 exit /b 2

echo ===== PARITY =====
%HIPML% parity skinny
echo ===== SKINNY COST CURVE =====
%HIPML% skinny-bench --reps 20
echo ===== BATCH _b kernels =====
for %%B in (2 4 8) do %HIPML% batch-bench --gguf %MODEL% --tokens 16 --slots %%B
echo ===== BATCH skinny wmma =====
set HIPML_BATCH_SKINNY=1
for %%B in (2 4 8) do %HIPML% batch-bench --gguf %MODEL% --tokens 16 --slots %%B
echo ===== BATCH skinny check =====
%HIPML% batch-bench --gguf %MODEL% --tokens 16 --slots 4 --check
set HIPML_BATCH_SKINNY=

powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp33
endlocal
