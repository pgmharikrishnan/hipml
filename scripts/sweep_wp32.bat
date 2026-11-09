@echo off
rem Shape/pipeline sweep for WP32. Run under the GPU lock.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cd /d C:\hipml-gemm18
for %%E in (probe_gemm p2_A0 p2_A1 p2_A2 p2_A2T p2_B0 p2_B2 p2_C0) do (
  for %%K in (q6kx q4kx) do (
    for %%M in (512 2048 8192) do (
      echo --- %%E %%K M=%%M up
      %%E.exe %%K 17408 5120 %%M 6
      echo --- %%E %%K M=%%M gate
      %%E.exe %%K 5120 17408 %%M 6
    )
  )
)
