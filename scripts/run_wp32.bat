@echo off
rem GPU-locked wrapper (WP32). Usage: run_wp32.bat <script-or-command...>
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp32
if errorlevel 1 exit /b 1
echo ===ACQUIRED===
call %*
set RC=%ERRORLEVEL%
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp32
echo ===RELEASED===
endlocal
exit /b %RC%
