@echo off
rem One-shot WP31 measurement batch, run under bench.bat (wp31 GPU lock).
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set B=C:\hipml-decattn\target\release\hipml.exe
set NF=C:\hipml-decattn-nf\target\release\hipml.exe
set M=C:\models\Qwen3.8-27B-Q4_K_M.gguf

echo === parity kv (shipped s4) ===
%B% parity attn_decode_kv
echo === parity deep kv ===
%B% parity attn_decode_kv --deep
echo === bench-kv attribution ===
%B% bench-kv --depths 100000 --reps 3 --kernels 8,9,10,11,12 --splits 64
echo === bench-kv s4 at engine splits ===
%B% bench-kv --depths 32768,100000 --reps 3 --kernels 13
echo === decode fused ===
%B% gpu-bench --gguf %M% --depth 0 --kv q8 --tokens 20
%B% gpu-bench --gguf %M% --depth 32768 --kv q8 --tokens 20
%B% gpu-bench --gguf %M% --depth 100000 --kv q8 --tokens 20
echo === decode unfused ===
%NF% gpu-bench --gguf %M% --depth 0 --kv q8 --tokens 20
%NF% gpu-bench --gguf %M% --depth 100000 --kv q8 --tokens 20
echo === graph probe (fused, launches) ===
%B% gpu-bench --gguf %M% --depth 0 --kv q8 --graph-probe
echo === generate tokens ===
%B% generate --device --gguf %M% --ids 760,6511,314,9338,369 --tokens 20
echo === done ===
