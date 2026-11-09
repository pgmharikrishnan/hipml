@echo off
rem WP32 probe builds: geometry E0 at the shipped defaults, plus the pipelines.
setlocal
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
set INC=-I crates\hipml-kernels\kernels
set FLAGS=-O3 --offload-arch=gfx1201
hipcc %FLAGS% %INC% -DPWW2_MIN_BLOCKS=1 scripts\probe_wmma2.cpp -o p2_E0lb.exe || exit /b 1
hipcc %FLAGS% %INC% -DPWW2_MIN_BLOCKS=1 -DPWW2_PF=1 scripts\probe_wmma2.cpp -o p2_E0pf1.exe || exit /b 1
hipcc %FLAGS% %INC% -DPWW2_MIN_BLOCKS=1 -DPWW2_PF=3 scripts\probe_wmma2.cpp -o p2_E0pf3.exe || exit /b 1
echo --- built
hipcc %FLAGS% %INC% -DPWW2_MIN_BLOCKS=1 -DPWW2_PF=3 -Rpass-analysis=kernel-resource-usage scripts\probe_wmma2.cpp -o nul 2>&1 | findstr /i "prefill_q6_k_w2h"
endlocal
