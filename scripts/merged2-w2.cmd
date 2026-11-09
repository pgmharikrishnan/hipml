@echo off
rem Pin the whole-group warp decode-attention kernel (A/B).
setlocal
set HIPML_DECODE_KERNEL=w2
call C:\hipml-merged2\scripts\merged2-env.cmd %*
