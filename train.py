"""
train.py — SFT Pipeline DFK Multimodal

Usage:
    python train.py --smoke_test
    python train.py
    python train.py --hf_token hf_xxx
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_utils import DFKDataCollator, DFKDataset
from src.model_utils import load_model_and_processor
from src.training_utils import build_trainer, build_training_args, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default=str(ROOT / "config.yaml"))
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--hf_token",  type=str, default=os.environ.get("HF_TOKEN", ""))
    return p.parse_args()


def resolve(path: str) -> str:
    return str(ROOT / path) if not Path(path).is_absolute() else path


def preflight(cfg: dict, hf_token: str) -> None:
    import torch
    logger.info("─── Pre-flight ──────────────────────────────────────────────")
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("  GPU    : %s (%.1f GB)", gpu, vram)
    else:
        logger.warning("  GPU    : TIDAK ADA")

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("  HF     : Token OK ✓")
        except Exception as e:
            logger.warning("  HF     : %s", e)

    data = cfg["data"]
    for key, label in [("train_file","Train"), ("eval_file","Eval")]:
        p = data.get(key)
        if not p:
            continue
        full = resolve(p)
        exists = Path(full).exists()
        logger.info("  %-6s : %s %s", label, full, "✓" if exists else "✗")
        if not exists and key == "train_file":
            sys.exit(1)

    img = resolve(data.get("image_dir", "./dataset_images"))
    if Path(img).exists():
        n = sum(1 for _ in Path(img).rglob("*.jpg"))
        logger.info("  Images : %s (%d files) ✓", img, n)
    else:
        logger.warning("  Images : tidak ditemukan — %s", img)
    logger.info("─────────────────────────────────────────────────────────────")


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    preflight(cfg, args.hf_token)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = dict(cfg["training"])
    smoke_cfg = cfg.get("smoke_test", {})

    jsonl_path  = resolve(data_cfg.get("train_file", "train_final_fixed.jsonl"))
    eval_path   = data_cfg.get("eval_file")
    if eval_path:
        eval_path = resolve(eval_path)
    image_dir   = resolve(data_cfg.get("image_dir", "./dataset_images"))
    max_seq_len = model_cfg.get("max_seq_length", 2048)

    is_smoke = args.smoke_test
    if is_smoke:
        logger.info("=" * 60)
        logger.info("MODE: SMOKE TEST")
        logger.info("=" * 60)
        max_samples      = smoke_cfg.get("num_samples", 80)
        max_eval_samples = smoke_cfg.get("num_eval_samples", 20)
        output_dir       = smoke_cfg.get("output_dir", "./output/smoke_test")
        max_steps = smoke_cfg.get("max_steps", 5)

        # Fix: save_steps dan eval_steps harus sama, load_best_model_at_end=False
        train_cfg["max_steps"]              = max_steps
        train_cfg["num_train_epochs"]       = 1
        train_cfg["save_steps"]             = max_steps
        train_cfg["eval_steps"]             = max_steps
        train_cfg["logging_steps"]          = 1
        train_cfg["load_best_model_at_end"] = False
    else:
        logger.info("=" * 60)
        logger.info("MODE: FULL TRAINING")
        logger.info("=" * 60)
        max_samples      = None
        max_eval_samples = None
        output_dir       = train_cfg.get("output_dir", "./output/dfk_sft")

    # 1. Load model
    logger.info("[1/5] Loading model ...")
    model, processor = load_model_and_processor(model_cfg, cfg["lora"])

    # 2. Train dataset
    logger.info("[2/5] Train dataset ...")
    train_dataset = DFKDataset(
        jsonl_path=jsonl_path,
        image_dir=image_dir,
        processor=processor,
        max_seq_length=max_seq_len,
        max_samples=max_samples,
    )

    # 3. Eval dataset
    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        logger.info("[3/5] Eval dataset ...")
        eval_dataset = DFKDataset(
            jsonl_path=eval_path,
            image_dir=image_dir,
            processor=processor,
            max_seq_length=max_seq_len,
            max_samples=max_eval_samples,
        )
    else:
        logger.info("[3/5] Eval dataset — dilewati")

    # 4. Trainer
    logger.info("[4/5] Setup trainer ...")
    tok    = getattr(processor, "tokenizer", processor)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    collator = DFKDataCollator(pad_token_id=pad_id)

    training_args = build_training_args(train_cfg, output_dir, is_smoke)
    trainer = build_trainer(model, processor, train_dataset, collator, training_args, eval_dataset)

    # 5. Train
    logger.info("[5/5] Training ...")
    result = trainer.train()

    if not is_smoke:
        save_model(model, processor, output_dir)
        logger.info("=" * 60)
        logger.info("TRAINING SELESAI — output: %s", output_dir)
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("SMOKE TEST BERHASIL!")
        logger.info("Jalankan full training: python train.py")
        logger.info("=" * 60)

    logger.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
