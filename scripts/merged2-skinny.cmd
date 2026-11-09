@echo off
rem Opt into the non-bit-identical WMMA skinny GEMM (WP33/WP37).
setlocal
set HIPML_SKINNY=1
call C:\hipml-merged2\scripts\merged2-env.cmd %*
