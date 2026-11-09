@echo off
rem WP31 correctness batch: KV parity after the grid fix, and attn-parity on
rem three trees (current, base e9705d2, pre-fusion 2d3a824).
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
set B=C:\hipml-decattn\target\release\hipml.exe
set BS=C:\hipml-base\target\release\hipml.exe
set BP=C:\hipml-prefuse\target\release\hipml.exe
set M=C:\models\Qwen3.8-27B-Q4_K_M.gguf

echo === parity kv (grid fix) ===
%B% parity attn_decode_kv
echo === parity deep kv (grid fix) ===
%B% parity attn_decode_kv --deep
echo === attn-parity current ===
%B% attn-parity --gguf %M%
echo === attn-parity base e9705d2 ===
%BS% attn-parity --gguf %M%
echo === attn-parity pre-fusion 2d3a824 ===
%BP% attn-parity --gguf %M%
echo === done ===
