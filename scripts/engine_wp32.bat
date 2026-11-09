@echo off
rem WP32 engine measurement: v1 vs v2 prefill at 512 and 8192 tokens.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set EXE=target\release\hipml.exe
set MDL=C:\models\Qwen3.8-27B-Q4_K_M.gguf
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 acquire wp32
if errorlevel 1 exit /b 1
echo ===ACQUIRED===
echo --- prefill-bench v1
%EXE% prefill-bench --reps 5
echo --- prefill-bench v2
set HIPML_PREFILL_V2=1
%EXE% prefill-bench --reps 5
echo --- gpu-bench prefill 512 v2
%EXE% gpu-bench --gguf %MDL% --prefill 512
echo --- gpu-bench prefill 512 v1
set HIPML_PREFILL_V2=
%EXE% gpu-bench --gguf %MDL% --prefill 512
echo --- gpu-bench prefill 8192 v1
%EXE% gpu-bench --gguf %MDL% --prefill 8192
echo --- gpu-bench prefill 8192 v2
set HIPML_PREFILL_V2=1
%EXE% gpu-bench --gguf %MDL% --prefill 8192
echo --- gpu-bench prefill 8192 v2 split
%EXE% gpu-bench --gguf %MDL% --prefill 8192 --prefill-split
echo ===DONE===
powershell -NoProfile -ExecutionPolicy Bypass -File C:\models\gpu-lock.ps1 release wp32
endlocal
