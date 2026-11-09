@echo off
rem WP31 batch 3: attn-parity with the step fix, corrected attn-bench numbers,
rem and the engine-config attention comparison.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set B=C:\hipml-decattn\target\release\hipml.exe
set M=C:\models\Qwen3.8-27B-Q4_K_M.gguf

echo === attn-parity (step fix) ===
%B% attn-parity --gguf %M%
echo === attn-bench (step fix) ===
%B% attn-bench --gguf %M% --depths 0,8192,32768,100000 --reps 3
echo === bench-kv engine config: s4 vs staged at 256 splits ===
%B% bench-kv --depths 100000 --reps 3 --kernels 13,6 --splits 256
echo === bench-kv s4 splits 384 ===
%B% bench-kv --depths 100000 --reps 3 --kernels 13 --splits 384
echo === done ===
