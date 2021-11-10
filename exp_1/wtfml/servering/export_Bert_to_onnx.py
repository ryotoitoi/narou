#%%
"""
Bertのエクスポートを前提に作成しました。
他のモデルは適切は最適化方法があると思います。
"""
from typing import Optional

import torch
from transformers.models.bert_japanese.tokenization_bert_japanese import (
    BertJapaneseTokenizer,
)
from wtfml.engine.nlp.model import BERTBaseClassifier
from wtfml.engine.pl_engine.BERT_classification import BERTClassificationPlEngine
from wtfml.engine.pl_engine.BertMultiTaskClassification import (
    BertMTClassificationPlEngine,
)


#%%
def bert_onnx_export(
    pl_model_path: str,
    pl_engine_base: BERTClassificationPlEngine,
    tokenizer: BertJapaneseTokenizer,
    onnx_save_path: Optional[str] = None,
    is_optimeze: bool = True,
    is_quantize: bool = True,
) -> None:
    classification_model = pl_engine_base
    pl_engine = BertMTClassificationPlEngine(
        model=classification_model, output_attentions=True
    )
    pl_engine.load_from_checkpoint(
        pl_model_path, model=classification_model, output_attentions=True
    )
    onnx_save_path = onnx_save_path or ".".join(pl_model_path.split(".")[:-1]) + ".onnx"

    symbolic_names = {0: "batch_size", 1: "max_seq_len"}
    inputs = tokenizer.encode_plus(
        "ONNXのエクスポートのためのSample Inputの作成をここでしています",
        None,
        add_special_tokens=True,
        truncation=True,
    )

    ids = torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(
        0
    )
    torch.onnx.export(
        pl_engine,
        (ids, mask, token_type_ids),
        onnx_save_path,
        input_names=["ids", "mask", "token_type_ids"],
        export_params=True,
        output_names=["prediction", "attention_raw"],
        opset_version=12,
        dynamic_axes={
            "ids": symbolic_names,  # variable length axes
            "mask": symbolic_names,
            "token_type_ids": symbolic_names,
            "prediction": {0: "batch_size"},
            "attention_raw": {0: "batch_size", 2: "max_seq_len_2", 3: "max_seq_len_3"},
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
        quantize_dynamic(model_fp32_path, model_quant_path, weight_type=QuantType.QInt8)
        print(f"model is saved at {model_quant_path}")
    # %%


if __name__ == "__main__":
    pl_model_path = "/Users/yongtae/Documents/JX_press/data/fa_model/text_model_epoch=1-valid_loss=0.5102.ckpt"
    onnx_save_path = ""

    onnx_save_path = onnx_save_path or ".".join(pl_model_path.split(".")[:-1]) + ".onnx"
    is_optimeze: bool = False
    is_quantize: bool = True
    classification_model = BERTBaseClassifier(num_classes=10, output_attentions=True)
    tokenizer = BertJapaneseTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    bert_onnx_export(
        pl_model_path=pl_model_path,
        pl_engine_base=classification_model,
        tokenizer=tokenizer,
        onnx_save_path=onnx_save_path,
        is_optimeze=is_optimeze,
        is_quantize=is_quantize,
    )

# %%
