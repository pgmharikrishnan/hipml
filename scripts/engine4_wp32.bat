@echo off
rem WP32 engine: 8K and split ledgers, plus the single-tensor bench.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set EXE=target\release\hipml.exe
set MDL=C:\models\Qwen3.8-27B-Q4_K_M.gguf
set HIPML_PREFILL_V2=1
echo --- prefill-bench v2 xhalf
%EXE% prefill-bench --reps 5
echo --- gpu-bench prefill 8192 v2
%EXE% gpu-bench --gguf %MDL% --prefill 8192
echo --- gpu-bench prefill 8192 v2 split
%EXE% gpu-bench --gguf %MDL% --prefill 8192 --prefill-split
echo --- gpu-bench prefill 512 v2 split
%EXE% gpu-bench --gguf %MDL% --prefill 512 --prefill-split
echo ===DONE===
endlocal
