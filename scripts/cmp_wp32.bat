@echo off
rem WP32: bit-identity of the shipped v1 kernel and the v2 shape, fp16 x.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
probe_cmp.exe q4k 5120 5120 200
probe_cmp.exe q6k 5120 5120 200
probe_cmp.exe q8 5120 5120 200
probe_cmp.exe half 5120 5120 200
probe_cmp.exe q6k 5120 5120 333
probe_cmp.exe q4k 17408 5120 256
