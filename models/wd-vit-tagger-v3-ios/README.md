# wd-vit-tagger-v3, batch of 1 (iOS)

A copy of [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (revision `7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f`) prepared for hexagon's image check on iOS, where it runs through ONNX Runtime's Core ML execution provider, on the GPU (or the CPU if Core ML can't take it). The weights stay fp32.

**Changes**, in order, to the original `model.onnx`:

1. The batch dimension was fixed to 1 (`python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch_size --dim_value 1`).
2. ONNX Runtime 1.29's basic graph optimizations were applied and saved (`GraphOptimizationLevel.ORT_ENABLE_BASIC`, standard ONNX operators only), with `MatMulAddFusion` disabled. With the shapes fixed, this folds away the attention blocks' shape arithmetic (`Shape`/`Gather`), which Core ML can't run: the original is cut into 13 Core ML partitions with CPU work between them, this one runs as one (479 of 479 nodes, per `onnxruntime.tools.check_onnx_model_mobile_usability`).
3. The classifier head's `Gemm` was rewritten as the equivalent `MatMul` and `Add` (weights transposed to match).
4. In each block's GELU (tanh approximation), the input to the `x + 0.044715·x³` branch is clipped to ±10. That branch only feeds `tanh`, which is saturated well before |x| = 10, so the result is unchanged, but x³ no longer reaches about 2·10⁷, which overflows if anything computes it in fp16.

Nothing else was changed. The tag list is the original `selected_tags.csv`, downloaded from the model's repository.

**Checked** against the original on live thumbnails: after steps 1–3 every output matched to four decimals on 320 thumbnails (anime and furry art, young and adult characters); step 4 changed no output by more than 0.000002.

**Parts:** GitHub refuses files over 100 MB, so the model is split in four. Joined in order, they make `model.ios32.onnx`:

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `model.ios32.onnx.1` | 95,000,000 | `ba0850f4271ac67cb3173959d623d67227a0c25d868d37dc86885d23aa7d3d1e` |
| `model.ios32.onnx.2` | 95,000,000 | `5dd2cf1854e14f43ee4f01334ae243f8790af35aea0e0c7b1d168f2f5137e90c` |
| `model.ios32.onnx.3` | 95,000,000 | `f937adb7b3eeb067b46cba9b74378afa22ea9bd8b67259313cbde08b121ecbb0` |
| `model.ios32.onnx.4` | 93,502,831 | `44a6a15be3eefb417f50727d2b4e148665eb3e9540a6b13e7cb02378f4b4bb01` |
| `model.ios32.onnx` (joined) | 378,502,831 | `3e704ba20b39e91a91b784ea431d728a9caf98f26e521fbdef0c0d6866e2ca63` |

**License:** Apache License 2.0, as the original model. Copyright SmilingWolf; converted for hexagon.
