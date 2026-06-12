"""
model_utils.py
──────────────
Load model VLM + LoRA menggunakan Unsloth FastVisionModel.
Fix: auto-set chat template setelah model load.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

_TEMPLATE_CANDIDATES = [
    Path(__file__).parent.parent / "chat_template.jinja",
    Path("chat_template.jinja"),
    Path("/content/drive/MyDrive/dfk_sft_pipeline/chat_template.jinja"),
]


def _set_chat_template(processor: Any) -> None:
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "chat_template", None) is not None:
        logger.info("Chat template sudah ada ✓")
        return
    for path in _TEMPLATE_CANDIDATES:
        if path.exists():
            tok.chat_template = path.read_text(encoding="utf-8")
            logger.info("Chat template di-set dari: %s ✓", path)
            return
    logger.warning("chat_template.jinja tidak ditemukan — akan pakai format manual")


def load_model_and_processor(model_cfg: Dict, lora_cfg: Dict) -> Tuple[Any, Any]:
    model_id                   = model_cfg["model_id"]
    load_in_4bit               = model_cfg.get("load_in_4bit", True)
    max_seq_length             = model_cfg.get("max_seq_length", 2048)
    use_gradient_checkpointing = model_cfg.get("use_gradient_checkpointing", "unsloth")

    try:
        model, processor = _load_with_unsloth(
            model_id, load_in_4bit, max_seq_length,
            use_gradient_checkpointing, lora_cfg
        )
    except Exception as exc:
        logger.warning("Unsloth gagal (%s) — fallback ke transformers+PEFT", exc)
        model, processor = _load_with_transformers(model_id, load_in_4bit, lora_cfg)

    _set_chat_template(processor)
    return model, processor


def _load_with_unsloth(
    model_id, load_in_4bit, max_seq_length, use_gradient_checkpointing, lora_cfg
):
    from unsloth import FastVisionModel  # type: ignore

    logger.info("Loading model dengan Unsloth: %s", model_id)
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_id,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing=use_gradient_checkpointing,
        max_seq_length=max_seq_length,
    )

    logger.info("Menerapkan LoRA ...")
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = lora_cfg.get("finetune_vision_layers", False),
        finetune_language_layers   = lora_cfg.get("finetune_language_layers", True),
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r            = lora_cfg.get("r", 16),
        lora_alpha   = lora_cfg.get("lora_alpha", 32),
        lora_dropout = lora_cfg.get("lora_dropout", 0.05),
        bias         = lora_cfg.get("bias", "none"),
        use_rslora   = lora_cfg.get("use_rslora", False),
        random_state = lora_cfg.get("random_state", 42),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100*trainable/total)
    return model, processor


def _load_with_transformers(model_id, load_in_4bit, lora_cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    logger.info("Loading model dengan transformers: %s", model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if load_in_4bit else None

    model     = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16), lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules",
            ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"), task_type=TaskType.CAUSAL_LM,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor
