@echo off
rem WP32 sweep 3: the 256-thread / 16-accumulator shape across both ffn shapes.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
echo --- E0 up
p2_E0.exe q6kx 17408 5120 512 6
p2_E0.exe q6kx 17408 5120 2048 6
p2_E0.exe q6kx 17408 5120 8192 6
p2_E0.exe q4kx 17408 5120 512 6
p2_E0.exe q4kx 17408 5120 2048 6
p2_E0.exe q4kx 17408 5120 8192 6
echo --- E0 gate
p2_E0.exe q6kx 5120 17408 512 6
p2_E0.exe q6kx 5120 17408 2048 6
p2_E0.exe q6kx 5120 17408 8192 6
p2_E0.exe q4kx 5120 17408 512 6
p2_E0.exe q4kx 5120 17408 2048 6
p2_E0.exe q4kx 5120 17408 8192 6
echo --- E0 KW=64 up
p2_E0k64.exe q6kx 17408 5120 512 6
p2_E0k64.exe q6kx 17408 5120 2048 6
p2_E0k64.exe q6kx 17408 5120 8192 6
p2_E0k64.exe q4kx 17408 5120 512 6
p2_E0k64.exe q4kx 17408 5120 2048 6
p2_E0k64.exe q4kx 17408 5120 8192 6
echo --- E0 KW=64 gate
p2_E0k64.exe q6kx 5120 17408 512 6
p2_E0k64.exe q6kx 5120 17408 2048 6
p2_E0k64.exe q6kx 5120 17408 8192 6
p2_E0k64.exe q4kx 5120 17408 512 6
p2_E0k64.exe q4kx 5120 17408 2048 6
p2_E0k64.exe q4kx 5120 17408 8192 6
echo --- E1 double buffer only
p2_E1.exe q6kx 17408 5120 2048 6
p2_E1.exe q4kx 17408 5120 2048 6
echo --- E0 split
p2_E0nst.exe q6kx 17408 5120 2048 6
p2_E0nst.exe q4kx 17408 5120 2048 6
p2_E0nfr.exe q6kx 17408 5120 2048 6
p2_E0nfr.exe q4kx 17408 5120 2048 6
echo --- v1 vs v2 bit identity
probe_cmp.exe q4k 5120 5120 200
probe_cmp.exe q6k 5120 5120 200
probe_cmp.exe q8 5120 5120 200
probe_cmp.exe half 5120 5120 200
probe_cmp.exe q6k 5120 5120 333
probe_cmp.exe q4k 17408 5120 256
