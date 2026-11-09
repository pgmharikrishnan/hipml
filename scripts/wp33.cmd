@echo off
rem WP33 skinny-GEMM measurement, one GPU-lock hold. Run from the tree root.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp33
if errorlevel 1 exit /b 1
echo ===== PARITY =====
.\target\release\hipml.exe parity skinny
echo ===== SKINNY BENCH =====
.\target\release\hipml.exe skinny-bench --reps 10 --geometry
set RC=%ERRORLEVEL%
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp33
exit /b %RC%
