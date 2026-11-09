@echo off
rem Pin the staged quantized decode-attention kernel (A/B).
setlocal
set HIPML_DECODE_KERNEL=staged
call C:\hipml-merged2\scripts\merged2-env.cmd %*
