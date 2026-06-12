"""
model_utils.py
"""
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

_TEMPLATE_CANDIDATES = [
    Path(__file__).parent.parent / "chat_template.jinja",
    Path("chat_template.jinja"),
]


def _set_chat_template(tokenizer: Any) -> None:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if getattr(tok, "chat_template", None) is not None:
        logger.info("Chat template sudah ada ✓")
        return
    for path in _TEMPLATE_CANDIDATES:
        if path.exists():
            tok.chat_template = path.read_text(encoding="utf-8")
            logger.info("Chat template di-set dari: %s ✓", path)
            return
    logger.warning("chat_template.jinja tidak ditemukan")


def load_model_and_processor(model_cfg: Dict, lora_cfg: Dict) -> Tuple[Any, Any]:
    model_id       = model_cfg["model_id"]
    load_in_4bit   = model_cfg.get("load_in_4bit", True)
    max_seq_length = model_cfg.get("max_seq_length", 2048)

    try:
        model, tokenizer = _load_with_unsloth(model_id, load_in_4bit, max_seq_length, lora_cfg)
    except Exception as exc:
        logger.warning("Unsloth gagal (%s) — fallback ke transformers+PEFT", exc)
        model, tokenizer = _load_with_transformers(model_id, load_in_4bit, lora_cfg)

    _set_chat_template(tokenizer)
    return model, tokenizer


def _load_with_unsloth(model_id, load_in_4bit, max_seq_length, lora_cfg):
    from unsloth import FastModel  # type: ignore

    logger.info("Loading model dengan Unsloth FastModel: %s", model_id)

    model, tokenizer = FastModel.from_pretrained(
        model_name     = model_id,
        max_seq_length = max_seq_length,
        load_in_4bit   = load_in_4bit,
        dtype          = None,
    )

    logger.info("Menerapkan LoRA ...")
    model = FastModel.get_peft_model(
        model,
        r              = lora_cfg.get("r", 16),
        lora_alpha     = lora_cfg.get("lora_alpha", 32),
        target_modules = lora_cfg.get("target_modules",
            ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout   = lora_cfg.get("lora_dropout", 0.05),
        bias           = lora_cfg.get("bias", "none"),
        use_gradient_checkpointing = "unsloth",
        random_state   = lora_cfg.get("random_state", 42),
        use_rslora     = lora_cfg.get("use_rslora", False),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable: %s / %s (%.2f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    return model, tokenizer


def _load_with_transformers(model_id, load_in_4bit, lora_cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    logger.info("Loading model dengan transformers: %s", model_id)
    bnb = BitsAndBytesConfig(
        load_in_4bit             = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type      = "nf4",
        bnb_4bit_compute_dtype   = torch.bfloat16,
    ) if load_in_4bit else None

    model     = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    tokenizer = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    lora_config = LoraConfig(
        r              = lora_cfg.get("r", 16),
        lora_alpha     = lora_cfg.get("lora_alpha", 32),
        target_modules = lora_cfg.get("target_modules",
            ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),
        lora_dropout   = lora_cfg.get("lora_dropout", 0.05),
        bias           = lora_cfg.get("bias", "none"),
        task_type      = TaskType.CAUSAL_LM,
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer
