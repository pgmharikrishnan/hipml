@echo off
rem WP32: KW=64 (fewer barriers, bigger staged tile) against the KW=32 shape.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
echo --- KW=64 up
p2_E0k64.exe q6kx 17408 5120 512 6
p2_E0k64.exe q6kx 17408 5120 2048 6
p2_E0k64.exe q6kx 17408 5120 8192 6
p2_E0k64.exe q4kx 17408 5120 512 6
p2_E0k64.exe q4kx 17408 5120 2048 6
echo --- KW=64 gate
p2_E0k64.exe q6kx 5120 17408 512 6
p2_E0k64.exe q6kx 5120 17408 2048 6
p2_E0k64.exe q4kx 5120 17408 512 6
p2_E0k64.exe q4kx 5120 17408 2048 6
