#%%
"""
Bertのエクスポートを前提に作成しました。
他のモデルは適切は最適化方法があると思います。
"""
#%%
from typing import Optional

import torch
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    BertJapaneseTokenizer,
)
from wtfml.engine.image.classification.model import SimpleAttentionEfficientNetwork
from wtfml.engine.nlp.model import BERTBaseClassifier
from wtfml.engine.pl_engine.ImageMultiTaskClassification import (
    ImageMTClassificationPlEngineForServering,
)

pl_model_path = "/Users/yongtae/Documents/JX_press/data/fa_model/combine_model_epoch=1-val_loss=0.0037.ckpt"
onnx_save_path = ""
onnx_save_path = onnx_save_path or ".".join(pl_model_path.split(".")[:-1]) + ".onnx"
is_optimeze: bool = False
is_quantize: bool = True
tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

bert_model = BERTBaseClassifier(num_classes=8 + 2, output_attentions=True)
image_model = SimpleAttentionEfficientNetwork(model_name_num=4)
image_model.base_model.set_swish(memory_efficient=False)

pl_engine = ImageMTClassificationPlEngineForServering.load_from_checkpoint(
    pl_model_path, bert_model=bert_model, image_embedding_model=image_model
)
#%%


symbolic_names = {0: "batch_size", 1: "max_seq_len"}
inputs = tokenizer.encode_plus(
    "ONNXのエクスポートのためのSample Inputの作成をここでしています",
    None,
    add_special_tokens=True,
    truncation=True,
)

ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)
token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0)
dummy_data = torch.ones((1, 3, 224, 224))
torch.onnx.export(
    pl_engine,
    (ids, mask, token_type_ids, dummy_data),
    onnx_save_path,
    input_names=["ids", "mask", "token_type_ids", "image"],
    export_params=True,
    output_names=["prediction", "text_attention_raw", "image_attention"],
    opset_version=12,
    dynamic_axes={
        "ids": symbolic_names,  # variable length axes
        "mask": symbolic_names,
        "token_type_ids": symbolic_names,
        "image": {0: "batch_size"},
        "prediction": {0: "batch_size"},
        "text_attention_raw": {0: "batch_size", 2: "max_seq_len", 3: "max_seq_len"},
        "image_attention": {0: "batch_size"},
    },
)

#%%
if is_optimeze:
    from onnxruntime.transformers import optimizer

    optimized_model_path = (
        ".".join(onnx_save_path.split(".")[:-1])
        + "_opt."
        + onnx_save_path.split(".")[-1]
    )
    optimized_model = optimizer.optimize_model(
        onnx_save_path, model_type="bert", num_heads=12, hidden_size=768
    )
    optimized_model.save_model_to_file(optimized_model_path)
    print(f"model is saved at {optimized_model_path}")
#%%
if is_quantize:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    model_fp32_path = optimized_model_path if is_optimeze else onnx_save_path
    model_quant_path = (
        ".".join(model_fp32_path.split(".")[:-1])
        + "_quant."
        + model_fp32_path.split(".")[-1]
    )
    quantize_dynamic(model_fp32_path, model_quant_path)
    print(f"model is saved at {model_quant_path}")

# %%
