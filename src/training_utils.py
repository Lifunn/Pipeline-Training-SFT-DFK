"""
training_utils.py
─────────────────
Switch ke SFTTrainer + SFTConfig dari TRL.
Fix: dynamo suppress + handle pixel_values Pixtral.
"""

import logging
import os
from typing import Any, Optional

import torch
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


class DFKSFTTrainer(SFTTrainer):
    """
    Custom SFTTrainer untuk Pixtral/Mistral3 VLM.
    Override _prepare_inputs untuk handle pixel_values tensor/list.
    """

    def _prepare_inputs(self, inputs: dict) -> dict:
        pv  = inputs.pop("pixel_values", None)
        isz = inputs.pop("image_sizes", None)

        # Move semua tensor lain ke device
        inputs = super()._prepare_inputs(inputs)

        # Handle pixel_values
        if pv is not None:
            if isinstance(pv, (list, tuple)):
                tensors = [p for p in pv if isinstance(p, torch.Tensor)]
                if tensors:
                    pv = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
                else:
                    pv = None
            if isinstance(pv, torch.Tensor):
                # Pastikan 4D [batch/tiles, C, H, W]
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)
                inputs["pixel_values"] = pv.to(self.args.device)

        # Handle image_sizes
        if isz is not None:
            if isinstance(isz, (list, tuple)):
                tensors = [s for s in isz if isinstance(s, torch.Tensor)]
                if tensors:
                    isz = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
                else:
                    isz = None
            if isinstance(isz, torch.Tensor):
                inputs["image_sizes"] = isz.to(self.args.device)

        return inputs


def build_training_args(
    train_cfg: dict,
    output_dir: str,
    is_smoke_test: bool = False,
) -> SFTConfig:
    os.makedirs(output_dir, exist_ok=True)

    return SFTConfig(
        output_dir=output_dir,

        # Steps / epochs
        num_train_epochs = train_cfg.get("num_train_epochs", 3),
        max_steps        = train_cfg.get("max_steps", -1),

        # Batch
        per_device_train_batch_size = train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size  = train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 4),

        # Optimizer
        learning_rate     = float(train_cfg.get("learning_rate", 2e-5)),
        warmup_ratio      = train_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type = train_cfg.get("lr_scheduler_type", "cosine"),
        optim             = train_cfg.get("optim", "adamw_8bit"),
        weight_decay      = train_cfg.get("weight_decay", 0.01),
        max_grad_norm     = train_cfg.get("max_grad_norm", 1.0),

        # Precision
        bf16 = train_cfg.get("bf16", True),
        fp16 = train_cfg.get("fp16", False),

        # NEFTune
        neftune_noise_alpha = train_cfg.get("neftune_noise_alpha", 5),

        # Logging
        logging_steps    = train_cfg.get("logging_steps", 10),
        logging_strategy = "steps",

        # Save
        save_strategy    = train_cfg.get("save_strategy", "steps"),
        save_steps       = train_cfg.get("save_steps", 200),
        save_total_limit = train_cfg.get("save_total_limit", 3),

        # Eval
        eval_strategy          = train_cfg.get("eval_strategy", "no"),
        eval_steps             = train_cfg.get("eval_steps", 200),
        load_best_model_at_end = train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model  = train_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better      = train_cfg.get("greater_is_better", False),

        # DataLoader
        dataloader_num_workers = 0 if is_smoke_test else train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory  = not is_smoke_test,

        # Misc
        report_to                     = train_cfg.get("report_to", "none"),
        remove_unused_columns         = False,
        label_names                   = ["labels"],
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        seed                          = 42,

        # SFT specific — skip default text formatting karena data sudah pre-tokenized
        dataset_kwargs = {"skip_prepare_dataset": True},
    )


def build_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    collator: Any,
    training_args: SFTConfig,
    eval_dataset: Optional[Any] = None,
) -> DFKSFTTrainer:
    logger.info("Membangun SFTTrainer ...")
    logger.info("  Train : %d samples", len(train_dataset))
    if eval_dataset:
        logger.info("  Eval  : %d samples", len(eval_dataset))
    logger.info("  Effective BS : %d",
                training_args.per_device_train_batch_size *
                training_args.gradient_accumulation_steps)

    return DFKSFTTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        data_collator = collator,
    )


def save_model(model: Any, processor: Any, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Menyimpan model ke: %s", output_dir)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "training_info.txt"), "w") as f:
        f.write("Base  : aitf-komdigi/KomdigiITS-8B-DFK-CPT\n")
        f.write("Type  : LoRA adapter (PEFT)\n")
        f.write("Task  : SFT DFK multimodal\n")
    logger.info("Model tersimpan ✓")
