@echo off
rem WP32: single-tensor bench after the fp16-activation mapping fix.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set HIPML_PREFILL_V2=1
target\release\hipml.exe prefill-bench --reps 5
endlocal
