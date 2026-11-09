@echo off
rem WP32 sweep 4: launch-bound E0 against the x-only pipeline, both ffn shapes.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
echo --- E0 launch_bounds up
p2_E0lb.exe q6kx 17408 5120 512 6
p2_E0lb.exe q6kx 17408 5120 2048 6
p2_E0lb.exe q6kx 17408 5120 8192 6
p2_E0lb.exe q4kx 17408 5120 512 6
p2_E0lb.exe q4kx 17408 5120 2048 6
p2_E0lb.exe q4kx 17408 5120 8192 6
echo --- E0 launch_bounds gate
p2_E0lb.exe q6kx 5120 17408 512 6
p2_E0lb.exe q6kx 5120 17408 2048 6
p2_E0lb.exe q6kx 5120 17408 8192 6
p2_E0lb.exe q4kx 5120 17408 512 6
p2_E0lb.exe q4kx 5120 17408 2048 6
p2_E0lb.exe q4kx 5120 17408 8192 6
echo --- E0 PF1 up
p2_E0pf1.exe q6kx 17408 5120 2048 6
p2_E0pf1.exe q4kx 17408 5120 2048 6
echo --- E0 PF3 up
p2_E0pf3.exe q6kx 17408 5120 2048 6
p2_E0pf3.exe q4kx 17408 5120 2048 6
p2_E0pf3.exe q6kx 17408 5120 512 6
p2_E0pf3.exe q4kx 17408 5120 512 6
p2_E0pf3.exe q6kx 5120 17408 2048 6
p2_E0pf3.exe q4kx 5120 17408 2048 6
