@echo off
rem WP31 batch 4: kernel choice at the engine split count, fusion A/B with
rem 50 tokens, and the token check through the shipped q8 decode path.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set B=C:\hipml-decattn\target\release\hipml.exe
set NF=C:\hipml-decattn-nf\target\release\hipml.exe
set M=C:\models\Qwen3.8-27B-Q4_K_M.gguf

echo === bench-kv variants at 256 splits (grid fixed) ===
%B% bench-kv --depths 100000 --reps 3 --kernels 4,7,13,14 --splits 256
echo === decode fused 50 tokens ===
%B% gpu-bench --gguf %M% --depth 0 --kv q8 --tokens 50
%B% gpu-bench --gguf %M% --depth 32768 --kv q8 --tokens 50
%B% gpu-bench --gguf %M% --depth 100000 --kv q8 --tokens 50
echo === decode unfused 50 tokens ===
%NF% gpu-bench --gguf %M% --depth 0 --kv q8 --tokens 50
%NF% gpu-bench --gguf %M% --depth 100000 --kv q8 --tokens 50
echo === tokens through the q8 decode path ===
%B% generate --device --kv q8 --gguf %M% --ids 760,6511,314,9338,369 --tokens 20
echo === done ===
