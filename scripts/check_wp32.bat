@echo off
rem WP32 check: is the k=17408 q6k drop at 8K tokens real or noise?
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
p2_E0lb.exe q6kx 5120 17408 8192 3
p2_E0lb.exe q6kx 5120 17408 8192 3
p2_E0lb.exe q4kx 5120 17408 8192 3
p2_E0lb.exe q6kx 5120 17408 4096 3
p2_E0lb.exe q6kx 5120 17408 2048 3
p2_E0lb.exe q6kx 17408 5120 8192 3
