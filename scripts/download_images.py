"""
download_images.py — Download images dari HF dataset

Usage:
    python scripts/download_images.py --hf_token hf_xxx
    python scripts/download_images.py --hf_token hf_xxx --output_dir /path/to/dir
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "aitf-its-tim3-dfk/aitf-dfk3-vlm-dataset-jsonl"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_token",   type=str, default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--dataset_id", type=str, default=DATASET_ID)
    p.add_argument("--output_dir", type=str, default="./dataset_images")
    p.add_argument("--verify",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.hf_token:
        logger.error("HF_TOKEN tidak ada. Gunakan --hf_token atau export HF_TOKEN=...")
        sys.exit(1)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("pip install huggingface-hub")
        sys.exit(1)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset   : %s", args.dataset_id)
    logger.info("Output    : %s", output.resolve())

    try:
        local = snapshot_download(
            repo_id         = args.dataset_id,
            repo_type       = "dataset",
            token           = args.hf_token,
            local_dir       = str(output),
            allow_patterns  = ["images/**", "framevideo/**"],
            ignore_patterns = ["*.jsonl", "README*", ".gitattributes"],
        )
        logger.info("Download selesai: %s", local)
    except Exception as e:
        logger.error("Download gagal: %s", e)
        sys.exit(1)

    if args.verify:
        total = sum(1 for f in output.rglob("*")
                    if f.suffix.lower() in {".jpg",".jpeg",".png",".webp"})
        logger.info("Total images: %d", total)

    logger.info("Done ✓ — update image_dir di config.yaml ke: %s", output.resolve())


if __name__ == "__main__":
    main()
