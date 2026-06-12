"""
data_utils.py
─────────────
Fix:
  - Skip sample dengan valid tokens < 10 (mencegah loss NaN)
  - pixel_values list dari PixtralProcessor di-normalize ke tensor 4D
  - __getitem__ iteratif (tidak rekursif)
  - Fallback manual Mistral [INST]...[/INST]
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Minimum token assistant yang harus ada agar sample dipakai training
MIN_VALID_TOKENS = 10


class DFKDataset(Dataset):

    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        processor: Any,
        max_seq_length: int = 2048,
        max_samples: Optional[int] = None,
    ) -> None:
        self.image_dir      = Path(image_dir)
        self.processor      = processor
        self.max_seq_length = max_seq_length
        self.samples: List[Dict] = []
        self._tok = getattr(processor, "tokenizer", processor)
        self._load_jsonl(jsonl_path, max_samples)
        logger.info(
            "DFKDataset: %d samples loaded%s",
            len(self.samples),
            f" (smoke test: {max_samples})" if max_samples else "",
        )

    def _load_jsonl(self, path: str, max_samples: Optional[int]) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _collect_images(self, messages: List[Dict]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image":
                    continue
                rel = block.get("image", "")
                if not rel:
                    continue
                full = self.image_dir / rel
                if full.exists():
                    try:
                        images.append(Image.open(full).convert("RGB"))
                    except Exception as e:
                        logger.debug("Cannot open image %s: %s", full, e)
        return images

    # ── Text formatting ───────────────────────────────────────────────────────

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b["text"] for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            )
        return ""

    @staticmethod
    def _to_processor_messages(messages: List[Dict]) -> List[Dict]:
        out = []
        for msg in messages:
            role    = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                out.append({"role": role, "content": content})
                continue
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and b.get("text"):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image" and b.get("image"):
                    parts.append({"type": "image"})
            if parts:
                out.append({"role": role, "content": parts})
        return out

    @staticmethod
    def _split_prompt(messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "assistant":
                return messages[:i], messages[i:]
        return messages, []

    def _build_texts(self, messages: List[Dict]) -> Tuple[str, str]:
        formatted      = self._to_processor_messages(messages)
        prompt_msgs, _ = self._split_prompt(formatted)
        try:
            full_text   = self._tok.apply_chat_template(
                formatted, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self._tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            return full_text, prompt_text
        except Exception:
            pass
        return self._manual_mistral_format(messages)

    def _manual_mistral_format(self, messages: List[Dict]) -> Tuple[str, str]:
        system_text = user_text = asst_text = ""
        for msg in messages:
            role = msg.get("role", "")
            text = self._content_to_text(msg.get("content", ""))
            if role == "system":
                system_text = text
            elif role == "user":
                user_text = text
            elif role == "assistant":
                asst_text = text

        user_part = f"{system_text}\n\n{user_text}".strip() if system_text else user_text
        bos = getattr(self._tok, "bos_token", "<s>") or "<s>"
        eos = getattr(self._tok, "eos_token", "</s>") or "</s>"
        full_text   = f"{bos}[INST] {user_part} [/INST] {asst_text}{eos}"
        prompt_text = f"{bos}[INST] {user_part} [/INST] "
        return full_text, prompt_text

    # ── Pixel values helpers ──────────────────────────────────────────────────

    @staticmethod
    def _normalize_pixel_values(pixel_values: Any) -> Optional[torch.Tensor]:
        """
        Normalize pixel_values ke 4D tensor [tiles/batch, C, H, W].
        PixtralProcessor return list of tensors untuk dynamic resolution.
        PENTING: Tidak pernah squeeze ke 3D.
        """
        if pixel_values is None:
            return None

        if isinstance(pixel_values, (list, tuple)):
            if not pixel_values:
                return None
            if len(pixel_values) == 1:
                pixel_values = pixel_values[0]
            else:
                try:
                    tensors = []
                    for p in pixel_values:
                        if isinstance(p, torch.Tensor):
                            if p.dim() == 3:
                                p = p.unsqueeze(0)
                            tensors.append(p)
                    pixel_values = torch.cat(tensors, dim=0) if tensors else None
                except Exception:
                    pixel_values = pixel_values[0]

        if not isinstance(pixel_values, torch.Tensor):
            return None

        # Pastikan 4D — JANGAN squeeze ke 3D
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)   # [C,H,W] → [1,C,H,W]
        elif pixel_values.dim() == 5:
            pixel_values = pixel_values.squeeze(0)     # [1,t,C,H,W] → [t,C,H,W]

        return pixel_values

    @staticmethod
    def _normalize_image_sizes(image_sizes: Any) -> Optional[torch.Tensor]:
        """Normalize image_sizes ke 2D tensor [num_images, 2]."""
        if image_sizes is None:
            return None

        if isinstance(image_sizes, (list, tuple)):
            if not image_sizes:
                return None
            tensors = []
            for s in image_sizes:
                if isinstance(s, torch.Tensor):
                    if s.dim() == 0:
                        s = s.unsqueeze(0)
                    if s.dim() == 1:
                        s = s.unsqueeze(0)
                    tensors.append(s)
            if not tensors:
                return None
            try:
                image_sizes = torch.cat(tensors, dim=0)
            except Exception:
                image_sizes = tensors[0]

        if not isinstance(image_sizes, torch.Tensor):
            return None

        if image_sizes.dim() == 0:
            image_sizes = image_sizes.unsqueeze(0).unsqueeze(0)
        elif image_sizes.dim() == 1:
            image_sizes = image_sizes.unsqueeze(0)

        return image_sizes

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode(
        self, messages: List[Dict], images: List[Image.Image]
    ) -> Optional[Dict[str, torch.Tensor]]:
        try:
            full_text, prompt_text = self._build_texts(messages)
        except Exception as e:
            logger.debug("_build_texts failed: %s", e)
            return None

        has_images  = len(images) > 0
        proc_kwargs = dict(
            return_tensors = "pt",
            truncation     = True,
            max_length     = self.max_seq_length,
        )

        try:
            if has_images:
                full_enc   = self.processor(
                    text=full_text, images=images, **proc_kwargs)
                prompt_enc = self.processor(
                    text=prompt_text, images=images, return_tensors="pt")
            else:
                full_enc   = self._tok(full_text,   **proc_kwargs)
                prompt_enc = self._tok(prompt_text, return_tensors="pt")
        except Exception as e:
            logger.debug("Encoding failed: %s", e)
            return None

        try:
            input_ids      = full_enc["input_ids"][0]
            attention_mask = full_enc["attention_mask"][0]
            labels         = input_ids.clone()
            prompt_len     = prompt_enc["input_ids"].shape[1]
            labels[:min(prompt_len, len(labels))] = -100
        except Exception as e:
            logger.debug("Label masking failed: %s", e)
            return None

        # ── FIX UTAMA: Skip sample dengan valid tokens terlalu sedikit ────────
        # Terjadi ketika teks terlalu panjang sehingga assistant response
        # terpotong oleh max_seq_length — semua labels = -100, loss = NaN
        valid_tokens = (labels != -100).sum().item()
        if valid_tokens < MIN_VALID_TOKENS:
            logger.debug(
                "Skip sample: valid tokens=%d < %d (teks terlalu panjang/terpotong)",
                valid_tokens, MIN_VALID_TOKENS
            )
            return None

        result: Dict[str, torch.Tensor] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

        pv = self._normalize_pixel_values(full_enc.get("pixel_values"))
        if pv is not None:
            result["pixel_values"] = pv

        isz = self._normalize_image_sizes(full_enc.get("image_sizes"))
        if isz is not None:
            result["image_sizes"] = isz

        return result

    @staticmethod
    def _dummy_sample() -> Dict[str, torch.Tensor]:
        seq = 16
        return {
            "input_ids":      torch.zeros(seq, dtype=torch.long),
            "attention_mask": torch.ones(seq,  dtype=torch.long),
            "labels":         torch.full((seq,), -100, dtype=torch.long),
        }

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        total = len(self.samples)
        for offset in range(total):
            try_idx  = (idx + offset) % total
            sample   = self.samples[try_idx]
            messages = sample.get("messages", [])
            images   = self._collect_images(messages)
            result   = self._encode(messages, images)
            if result is not None:
                return result

        logger.error("Semua sample gagal — return dummy")
        return self._dummy_sample()


class DFKDataCollator:

    def __init__(self, pad_token_id: int, label_pad_id: int = -100) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
        max_len = max(f["input_ids"].shape[0] for f in features)
        ids_list, attn_list, lbl_list = [], [], []

        for f in features:
            pad = max_len - f["input_ids"].shape[0]
            ids_list.append(torch.cat([
                torch.full((pad,), self.pad_token_id, dtype=torch.long),
                f["input_ids"]
            ]))
            attn_list.append(torch.cat([
                torch.zeros(pad, dtype=torch.long),
                f["attention_mask"]
            ]))
            lbl_list.append(torch.cat([
                torch.full((pad,), self.label_pad_id, dtype=torch.long),
                f["labels"]
            ]))

        batch: Dict[str, Any] = {
            "input_ids":      torch.stack(ids_list),
            "attention_mask": torch.stack(attn_list),
            "labels":         torch.stack(lbl_list),
        }

        # pixel_values: cat along dim=0 (tiles/batch dim)
        pv = [f["pixel_values"] for f in features if "pixel_values" in f]
        if pv:
            if len(pv) == 1:
                batch["pixel_values"] = pv[0]
            else:
                try:
                    batch["pixel_values"] = torch.cat(pv, dim=0)
                except RuntimeError:
                    batch["pixel_values"] = pv[0]

        # image_sizes: cat along dim=0
        isz = [f["image_sizes"] for f in features if "image_sizes" in f]
        if isz:
            if len(isz) == 1:
                batch["image_sizes"] = isz[0]
            else:
                try:
                    batch["image_sizes"] = torch.cat(isz, dim=0)
                except RuntimeError:
                    batch["image_sizes"] = isz[0]

        return batch
