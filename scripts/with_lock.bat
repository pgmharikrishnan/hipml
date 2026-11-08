@echo off
rem GPU-locked wrapper for WP16 benchmarks. Usage: with_lock.bat <command...>
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp16
if errorlevel 1 exit /b 1
call %*
set RC=%ERRORLEVEL%
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp16
exit /b %RC%
