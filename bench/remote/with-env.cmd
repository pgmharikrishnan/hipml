@echo off
rem Environment for every hipml invocation on the bench host.
rem
rem Nothing here is machine-specific: set HIP_PATH and HIPML_BENCH_REMOTE_DIR in
rem the shell that starts the bench, and this file fails loudly if either is
rem missing rather than falling back to somebody's drive layout.
rem
rem   set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core        (example; yours may differ)
rem   set HIPML_BENCH_REMOTE_DIR=C:\src\hipml
rem   bench\remote\with-env.cmd cargo build --release -p hipml-cli --features hip

if "%HIP_PATH%"=="" (
  echo with-env.cmd: set HIP_PATH to your ROCm SDK root ^(the directory that holds bin\^)>&2
  exit /b 1
)
if "%HIPML_BENCH_REMOTE_DIR%"=="" (
  echo with-env.cmd: set HIPML_BENCH_REMOTE_DIR to the checkout on this host>&2
  exit /b 1
)

if "%HIPML_ARCH%"=="" set HIPML_ARCH=gfx1201
set PATH=%HIP_PATH%\bin;%PATH%
cd /d "%HIPML_BENCH_REMOTE_DIR%" || exit /b 1
%*
