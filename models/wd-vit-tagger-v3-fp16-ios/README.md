# wd-vit-tagger-v3, fp16, batch of 1 (iOS)

A copy of [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (revision `7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f`) prepared for hexagon's image check on iOS, where it runs through ONNX Runtime's Core ML execution provider, on the Neural Engine in fp16.

**Changes**, in order, to the original `model.onnx`:

1. The batch dimension was fixed to 1 (`python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch_size --dim_value 1`).
2. ONNX Runtime 1.29's basic graph optimizations were applied and saved (`GraphOptimizationLevel.ORT_ENABLE_BASIC`, standard ONNX operators only), with `MatMulAddFusion` disabled. With the shapes fixed, this folds away the attention blocks' shape arithmetic (`Shape`/`Gather`), which Core ML can't run: the original is cut into 13 Core ML partitions with CPU work between them, this one runs as one.
3. The classifier head's `Gemm` was rewritten as the equivalent `MatMul` and `Add` (weights transposed to match).
4. In each block's GELU (tanh approximation), the input to the `x + 0.044715·x³` branch is clipped to ±10. That branch only feeds `tanh`, which is saturated well before |x| = 10, so the result is unchanged (largest output difference 0.000002), but x³ no longer reaches about 2·10⁷, which overflows fp16 (largest value 65,504) and turned into infinities and NaNs.
5. The weights were converted to fp16 (`onnxruntime.transformers.float16.convert_float_to_float16`, `keep_io_types=True`, so the input and output stay fp32), keeping the 25 `LayerNormalization` nodes in fp32, since squaring the residual stream's values (up to about 2,900) would overflow fp16 too.

Nothing else was changed. The tag list is the original `selected_tags.csv`, downloaded from the model's repository.

**Checked** against the original on live thumbnails (anime and furry art, young and adult characters): no fp16 tensor overflows (the copy without steps 4–5 had 17 that did), the largest fp16 value is about 2,900, the highest confidence among hexagon's built-in blacklist tags differed by at most 0.0005 on 160 thumbnails, and no flag changed at 10%, 20% or 35%. Core ML takes all 663 nodes as one partition (`onnxruntime.tools.check_onnx_model_mobile_usability`).

**Parts:** GitHub refuses files over 100 MB, so the model is split in two. Joined in order, they make `model.ios2.onnx`:

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `model.ios2.onnx.1` | 95,000,000 | `65aa864c90dfeaf71a4938a9cb579d804867fd846ac01c193bae08e814e68d17` |
| `model.ios2.onnx.2` | 94,430,159 | `563ef194ee55a9e64e5a716483afd4e6e5ea0568c1c7dd55e39b3c7b935aa46f` |
| `model.ios2.onnx` (joined) | 189,430,159 | `735a43f328cae60131be4577ad4c426c95b0f025e7da1f678e161ea5375852e3` |

**License:** Apache License 2.0, as the original model. Copyright SmilingWolf; converted for hexagon.
