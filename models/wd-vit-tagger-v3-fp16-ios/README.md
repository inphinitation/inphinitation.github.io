# wd-vit-tagger-v3, fp16, batch of 1 (iOS)

A copy of [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (revision `7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f`) prepared for hexagon's image check on iOS, where it runs through ONNX Runtime's Core ML execution provider.

**Changes**, in order, to the original `model.onnx`:

1. The batch dimension was fixed to 1 (`python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch_size --dim_value 1`).
2. ONNX Runtime 1.29's basic graph optimizations were applied and saved (`GraphOptimizationLevel.ORT_ENABLE_BASIC`, standard ONNX operators only), with `MatMulAddFusion` disabled. With the shapes fixed, this folds away the attention blocks' shape arithmetic (`Shape`/`Gather`), which Core ML can't run: the original is cut into 13 Core ML partitions with CPU work between them, this one runs as one (641 of 641 nodes, per `onnxruntime.tools.check_onnx_model_mobile_usability`).
3. The classifier head's `Gemm` was rewritten as the equivalent `MatMul` and `Add` (weights transposed to match).
4. The weights were converted to fp16 (`onnxruntime.transformers.float16.convert_float_to_float16`, `keep_io_types=True`, so the input and output stay fp32).

Nothing else was changed. The tag list is the original `selected_tags.csv`, downloaded from the model's repository.

**Checked** against the original on live thumbnails (anime and furry art, young and adult characters): after steps 1–3 (fp32) every output matched to four decimals on 320 thumbnails; after step 4, the highest confidence among hexagon's built-in blacklist tags differed by at most 0.001 on 160 thumbnails, and no flag changed at 10%, 20% or 35%.

**Parts:** GitHub refuses files over 100 MB, so the model is split in two. Joined in order, they make `model.ios.onnx`:

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `model.ios.onnx.1` | 95,000,000 | `7a520e6e5ae5a1fd8d7ed14cebd5b73c2bf9856773ec7538101510e4263c7739` |
| `model.ios.onnx.2` | 94,334,916 | `2b6b5369fa31f60c2c0a349d5ad86428daa36e21d4d57c8653f198220c1acf3c` |
| `model.ios.onnx` (joined) | 189,334,916 | `815766f8ac0879f919dfa338a845436391689a9f66ccc1c5431eddfdcf2a6046` |

**License:** Apache License 2.0, as the original model. Copyright SmilingWolf; converted for hexagon.
