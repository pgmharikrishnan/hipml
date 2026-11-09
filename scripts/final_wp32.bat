@echo off
rem WP32 acceptance run: v2 is now the default; no HIPML_PREFILL_V2 needed.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set EXE=target\release\hipml.exe
set MDL=C:\models\Qwen3.8-27B-Q4_K_M.gguf
echo --- prefill-bench default
%EXE% prefill-bench --reps 5
echo --- gpu-bench prefill 512
%EXE% gpu-bench --gguf %MDL% --prefill 512
echo --- gpu-bench prefill 8192
%EXE% gpu-bench --gguf %MDL% --prefill 8192
echo --- gpu-bench prefill 8192 split
%EXE% gpu-bench --gguf %MDL% --prefill 8192 --prefill-split
echo --- gpu-bench prefill 32768
%EXE% gpu-bench --gguf %MDL% --prefill 32768
echo --- gpu-bench prefill 102400
%EXE% gpu-bench --gguf %MDL% --prefill 102400
echo --- parity prefill deep
%EXE% parity prefill --deep
echo --- generate token check
%EXE% generate --gguf %MDL% --ids 760,6511,314,9338,369 --tokens 20
echo ===DONE===
endlocal
