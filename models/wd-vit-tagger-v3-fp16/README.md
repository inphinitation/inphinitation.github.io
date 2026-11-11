# wd-vit-tagger-v3, fp16

A half-precision copy of [SmilingWolf/wd-vit-tagger-v3](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3) (revision `7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f`), used by hexagon's image check on iOS.

- **Changes:** the weights of `model.onnx` were converted from fp32 to fp16 with ONNX Runtime's `onnxruntime.transformers.float16.convert_float_to_float16` (`keep_io_types=True`, so the input and output stay fp32). Nothing else was changed. The tag list is the original `selected_tags.csv`, downloaded from the model's repository.
- **Parts:** GitHub refuses files over 100 MB, so the model is split into `model.fp16.onnx.1` and `model.fp16.onnx.2`. Joined in order, they make `model.fp16.onnx`:

  | File | Size (bytes) | SHA-256 |
  | --- | --- | --- |
  | `model.fp16.onnx.1` | 95,000,000 | `11ccc30fee4ae7bf3457c1c6e9e51d969296a53baa9f8169e347f8301ea75bae` |
  | `model.fp16.onnx.2` | 94,390,871 | `aa7684f40cb1f7aa50e316e9edb27b3544fa681f4a31ba5177cffbbd2f70167f` |
  | `model.fp16.onnx` (joined) | 189,390,871 | `8711d89a9b55e08a7f3d8ef34294b54bf692003dfde2d6009a89d98a7c3ca914` |

- **License:** Apache License 2.0, as the original model. Copyright SmilingWolf; converted for hexagon.
