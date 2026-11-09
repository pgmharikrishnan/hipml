@echo off
rem WP32: acquire the GPU lock, run the shape sweep, always release.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 status
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp32
if errorlevel 1 exit /b 1
echo ===ACQUIRED===
call C:\hipml-gemm18\scripts\sweep_wp32.bat
echo ===SWEEP_DONE===
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp32
endlocal
