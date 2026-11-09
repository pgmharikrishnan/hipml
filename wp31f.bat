@echo off
rem WP31 final acceptance batch: shipped kernel w2, fusion reverted.
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set B=C:\hipml-decattn\target\release\hipml.exe
set M=C:\models\Qwen3.8-27B-Q4_K_M.gguf

echo === parity kv (shipped w2) ===
%B% parity attn_decode_kv
echo === parity deep kv ===
%B% parity attn_decode_kv --deep
echo === decode 50 tokens ===
%B% gpu-bench --gguf %M% --depth 0 --kv q8 --tokens 50
%B% gpu-bench --gguf %M% --depth 32768 --kv q8 --tokens 50
%B% gpu-bench --gguf %M% --depth 100000 --kv q8 --tokens 50
echo === tokens through the q8 decode path ===
%B% generate --device --kv q8 --gguf %M% --ids 760,6511,314,9338,369 --tokens 20
echo === attn-parity ===
%B% attn-parity --gguf %M%
echo === done ===
