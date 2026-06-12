"""
evaluate.py — Evaluasi model setelah training

Usage:
    python evaluate.py --model_dir ./output/dfk_sft --hf_token hf_xxx
    python evaluate.py --model_dir ./output/dfk_sft --max_samples 300 --hf_token hf_xxx
    python evaluate.py --model_dir ./output/dfk_sft --bertscore --hf_token hf_xxx
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

LABEL_MAP = {
    "disinformasi": "Disinformasi", "hoax": "Disinformasi",
    "fitnah": "Fitnah", "defamasi": "Fitnah",
    "ujaran kebencian": "Ujaran Kebencian", "kebencian": "Ujaran Kebencian",
    "hate speech": "Ujaran Kebencian",
    "fakta": "Fakta",
    "non-dfk": "Non-DFK", "non dfk": "Non-DFK", "bukan dfk": "Non-DFK",
}


def extract_label(text: str) -> str:
    if not text:
        return "Unknown"
    m = re.search(r"label\s*:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).strip().rstrip(".,;").lower()
        for key, val in LABEL_MAP.items():
            if key in raw:
                return val
        return m.group(1).strip().title()
    return "Unknown"


def load_eval_data(jsonl_path, image_dir, max_samples=None):
    from PIL import Image
    samples = []
    img_dir = Path(image_dir)
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line.strip())
            messages = item.get("messages", [])
            system_text = user_text = gt_label = ""
            img_path = image = None
            for msg in messages:
                role, content = msg.get("role"), msg.get("content", "")
                if role == "system" and isinstance(content, str):
                    system_text = content
                elif role == "user":
                    if isinstance(content, list):
                        for b in content:
                            if b.get("type") == "text" and b.get("text"):
                                user_text = b["text"]
                            elif b.get("type") == "image" and b.get("image"):
                                img_path = b["image"]
                    else:
                        user_text = content
                elif role == "assistant":
                    txt = content if isinstance(content, str) else \
                        " ".join(b.get("text","") for b in content
                                 if isinstance(b,dict) and b.get("type")=="text")
                    gt_label = extract_label(txt)
            if img_path:
                fp = img_dir / img_path
                if fp.exists():
                    try:
                        image = Image.open(fp).convert("RGB")
                    except Exception:
                        pass
            samples.append({"system_text":system_text,"user_text":user_text,
                             "image":image,"gt_label":gt_label})
    logger.info("Loaded %d eval samples", len(samples))
    return samples


def load_model(model_dir, no_lora=False):
    try:
        from unsloth import FastVisionModel
        model, processor = FastVisionModel.from_pretrained(
            model_name=model_dir, load_in_4bit=False, dtype=torch.bfloat16)
        FastVisionModel.for_inference(model)
        logger.info("Model loaded via Unsloth ✓")
    except Exception as e:
        logger.warning("Unsloth gagal (%s) — fallback transformers", e)
        from transformers import AutoModelForCausalLM, AutoProcessor
        from peft import PeftModel
        base = "aitf-komdigi/KomdigiITS-8B-DFK-CPT"
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
        if not no_lora:
            model = PeftModel.from_pretrained(model, model_dir)
        processor = AutoProcessor.from_pretrained(model_dir if not no_lora else base)
        model.eval()
    return model, processor


def generate_one(sample, model, processor, device, max_new_tokens=256):
    tok = getattr(processor, "tokenizer", processor)
    bos = getattr(tok, "bos_token", "<s>") or "<s>"
    user = f"{sample['system_text']}\n\n{sample['user_text']}".strip()
    prompt = f"{bos}[INST] {user} [/INST] "
    try:
        if sample["image"]:
            inputs = processor(text=prompt, images=[sample["image"]], return_tensors="pt").to(device)
        else:
            inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    except Exception as e:
        logger.warning("Generate gagal: %s", e)
        return ""


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
    valid = [(t,p) for t,p in zip(y_true,y_pred) if p != "Unknown"]
    if not valid:
        return {"error": "Semua prediksi Unknown"}
    yt, yp = [v[0] for v in valid], [v[1] for v in valid]
    labels = sorted(set(yt)|set(yp))
    return {
        "total": len(y_true), "valid": len(valid),
        "unknown": len(y_true)-len(valid),
        "accuracy":         accuracy_score(yt, yp),
        "precision_macro":  precision_score(yt, yp, average="macro",    zero_division=0, labels=labels),
        "recall_macro":     recall_score(yt, yp,    average="macro",    zero_division=0, labels=labels),
        "f1_macro":         f1_score(yt, yp,        average="macro",    zero_division=0, labels=labels),
        "f1_weighted":      f1_score(yt, yp,        average="weighted", zero_division=0, labels=labels),
        "report":           classification_report(yt, yp, labels=labels, zero_division=0),
    }


def print_results(m, bs=None):
    sep = "═"*60
    print(f"\n{sep}\n  HASIL EVALUASI\n{sep}")
    if "error" in m:
        print(f"  Error: {m['error']}"); return
    print(f"  Total       : {m['total']:,}")
    print(f"  Valid pred  : {m['valid']:,}")
    print(f"  Unknown     : {m['unknown']:,}")
    print(f"\n{'─'*60}")
    print(f"  Accuracy    : {m['accuracy']:.4f} ({m['accuracy']*100:.2f}%)")
    print(f"  Precision   : {m['precision_macro']:.4f}")
    print(f"  Recall      : {m['recall_macro']:.4f}")
    print(f"  F1 macro    : {m['f1_macro']:.4f}")
    print(f"  F1 weighted : {m['f1_weighted']:.4f}")
    print(f"\n{'─'*60}\n  PER-CLASS REPORT\n{'─'*60}")
    print(m["report"])
    if bs:
        print(f"{'─'*60}\n  BERTSCORE\n{'─'*60}")
        if "error" in bs:
            print(f"  {bs['error']}")
        else:
            print(f"  Precision : {bs['bertscore_precision']:.4f}")
            print(f"  Recall    : {bs['bertscore_recall']:.4f}")
            print(f"  F1        : {bs['bertscore_f1']:.4f}")
    print(sep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",      type=str, default="./output/dfk_sft")
    p.add_argument("--config",         type=str, default=str(ROOT/"config.yaml"))
    p.add_argument("--eval_file",      type=str, default=None)
    p.add_argument("--image_dir",      type=str, default=None)
    p.add_argument("--max_samples",    type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--hf_token",       type=str, default=os.environ.get("HF_TOKEN",""))
    p.add_argument("--bertscore",      action="store_true")
    p.add_argument("--no_lora",        action="store_true")
    p.add_argument("--output_json",    type=str, default=None)
    args = p.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg.get("data", {})
    eval_file = args.eval_file or data_cfg.get("eval_file", "valid_final_fixed.jsonl")
    image_dir = args.image_dir or data_cfg.get("image_dir", "./dataset_images")
    if not Path(eval_file).is_absolute():
        eval_file = str(ROOT / eval_file)
    if not Path(image_dir).is_absolute():
        image_dir = str(ROOT / image_dir)

    samples = load_eval_data(eval_file, image_dir, args.max_samples)
    model, processor = load_model(args.model_dir, args.no_lora)
    device = str(next(model.parameters()).device)

    y_true, y_pred, preds = [], [], []
    for i, s in enumerate(samples):
        if i % 50 == 0:
            logger.info("Progress: %d/%d", i, len(samples))
        gen = generate_one(s, model, processor, device, args.max_new_tokens)
        y_true.append(s["gt_label"])
        y_pred.append(extract_label(gen))
        preds.append(gen)

    metrics = compute_metrics(y_true, y_pred)

    bs = None
    if args.bertscore:
        try:
            from bert_score import score as bscore
            P, R, F = bscore(preds, [""] * len(preds), lang="id", verbose=False)
            bs = {"bertscore_precision": P.mean().item(),
                  "bertscore_recall": R.mean().item(),
                  "bertscore_f1": F.mean().item()}
        except ImportError:
            bs = {"error": "pip install bert-score"}

    print_results(metrics, bs)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"metrics": {k:v for k,v in metrics.items() if k!="report"},
                       "bertscore": bs}, f, indent=2)
        logger.info("Saved: %s", args.output_json)


if __name__ == "__main__":
    main()
