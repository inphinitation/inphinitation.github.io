# wd-vit-tagger-v3, fp16, batch of 1 (iOS)

A copy of [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (revision `7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f`) prepared for hexagon's image check on iOS, where it runs through ONNX Runtime's Core ML execution provider.

**Changes**, in order, to the original `model.onnx`:

1. The batch dimension was fixed to 1 (`python -m onnxruntime.tools.make_dynamic_shape_fixed --dim_param batch_size --dim_value 1`).
2. ONNX Runtime 1.29's basic graph optimizations were applied and saved (`GraphOptimizationLevel.ORT_ENABLE_BASIC`, standard ONNX operators only). With the shapes fixed, this folds away the attention blocks' shape arithmetic (`Shape`/`Gather`), which Core ML can't run: the original is cut into 13 Core ML partitions with CPU work between them, this one runs as a single partition (641 of 641 nodes, per `onnxruntime.tools.check_onnx_model_mobile_usability`).
3. The weights were converted to fp16 (`onnxruntime.transformers.float16.convert_float_to_float16`, `keep_io_types=True`, so the input and output stay fp32).

Nothing else was changed. The tag list is the original `selected_tags.csv`, downloaded from the model's repository.

**Parts:** GitHub refuses files over 100 MB, so the model is split in two. Joined in order, they make `model.fp16.b1.onnx`:

| File | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `model.fp16.b1.onnx.1` | 95,000,000 | `73568b6a1af6ec5dd532ea0874538bd65b27a60c58b893810517bb17842c879a` |
| `model.fp16.b1.onnx.2` | 94,348,744 | `d2622b5367a8f3256c397935085c8eea19cf6561211282478a339c2e54991cb2` |
| `model.fp16.b1.onnx` (joined) | 189,348,744 | `452b9590c6777297f20c79fdec4dd43f9ed6a8891faa280aca1c74fb2cd2f3e9` |

**License:** Apache License 2.0, as the original model. Copyright SmilingWolf; converted for hexagon.
