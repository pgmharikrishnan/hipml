@echo off
rem WP32 sweep 2: staging/multiply split at the 256x128 shape, plus candidates.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
echo --- base C0
p2_C0.exe q6kx 17408 5120 2048 6
p2_C0.exe q4kx 17408 5120 2048 6
echo --- no stage
p2_C0nst.exe q6kx 17408 5120 2048 6
p2_C0nst.exe q4kx 17408 5120 2048 6
echo --- no frag reads
p2_C0nfr.exe q6kx 17408 5120 2048 6
p2_C0nfr.exe q4kx 17408 5120 2048 6
echo --- no compute
p2_C0ncm.exe q6kx 17408 5120 2048 6
p2_C0ncm.exe q4kx 17408 5120 2048 6
echo --- E0 256thr 256x128 16acc
p2_E0.exe q6kx 17408 5120 512 6
p2_E0.exe q6kx 17408 5120 2048 6
p2_E0.exe q4kx 17408 5120 512 6
p2_E0.exe q4kx 17408 5120 2048 6
echo --- E2 same with register pipeline
p2_E2.exe q6kx 17408 5120 512 6
p2_E2.exe q6kx 17408 5120 2048 6
p2_E2.exe q4kx 17408 5120 512 6
p2_E2.exe q4kx 17408 5120 2048 6
echo --- C0 k-tile 64
p2_C0k64.exe q6kx 17408 5120 2048 6
p2_C0k64.exe q4kx 17408 5120 2048 6
echo --- v1 vs v2 bit identity
probe_cmp.exe q4k 5120 5120 200
probe_cmp.exe q6k 5120 5120 200
probe_cmp.exe q8 5120 5120 200
probe_cmp.exe half 5120 5120 200
probe_cmp.exe q6k 5120 5120 333
