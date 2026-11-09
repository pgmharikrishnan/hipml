@echo off
rem WP32 sweep 5: a half-height block (4 warps, 128 threads, two blocks per CU)
rem aimed at the short-prompt grid: 512-token ffn shapes are 2.5-8.5 waves.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set INC=-I crates\hipml-kernels\kernels
set FLAGS=-O3 --offload-arch=gfx1201
set G=-DPWW2_MIN_BLOCKS=1 -DPWW2_RT=64 -DPWW2_WR=64 -DPWW2_THREADS=128
hipcc %FLAGS% %INC% %G% scripts\probe_wmma2.cpp -o p2_F0.exe || exit /b 1
echo --- F0 up
p2_F0.exe q6kx 17408 5120 512 6
p2_F0.exe q4kx 17408 5120 512 6
p2_F0.exe q6kx 17408 5120 2048 6
p2_F0.exe q4kx 17408 5120 2048 6
echo --- F0 gate
p2_F0.exe q6kx 5120 17408 512 6
p2_F0.exe q4kx 5120 17408 512 6
p2_F0.exe q6kx 5120 17408 2048 6
p2_F0.exe q4kx 5120 17408 2048 6
echo --- E0 reference at 512 tokens
p2_E0lb.exe q6kx 17408 5120 512 6
p2_E0lb.exe q4kx 17408 5120 512 6
p2_E0lb.exe q6kx 5120 17408 512 6
p2_E0lb.exe q4kx 5120 17408 512 6
endlocal
