#!/usr/bin/env python3
"""Train the frozen Owner-SHORT W10 classifier on the local Apple GPU.

This is the local execution form of the frozen RTX 3060 recipe in
``fable-trading/scripts/train_fixed_w10_cls.py``.  Dataset, pretrained model,
white-letterbox preprocessing, optimizer, schedule, seed, and disabled
augmentations are identical; only the compute device differs from CUDA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


class WhiteLetterbox:
    """Pad with renderer white, keeping the complete chart frame."""

    def __init__(self, size: int, fill: int = 255):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, round(width * scale))
        new_height = max(1, round(height * scale))
        image = image.resize((new_width, new_height), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        if image.mode != "RGB":
            image = image.convert("RGB")
        canvas.paste(
            image,
            ((self.size - new_width) // 2, (self.size - new_height) // 2),
        )
        return canvas


def full_frame_transforms(size: int):
    """Match last night's white-letterbox plus ToTensor preprocessing."""
    import torchvision.transforms as transforms

    return transforms.Compose([WhiteLetterbox(size), transforms.ToTensor()])


def trainer_class():
    """Install the frozen full-frame transform for both train and validation."""
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    class FullFrameClassificationTrainer(ClassificationTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            dataset.torch_transforms = full_frame_transforms(int(self.args.imgsz))
            return dataset

    return FullFrameClassificationTrainer


def count_split(root: Path, split: str) -> dict[str, int]:
    """Count the two frozen classification folders."""
    return {
        label: len(list((root / split / label).glob("*.png")))
        for label in ("SIGNAL", "NO_SIGNAL")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--name", default="fixed_w10_gold500_short_ownerhn_v1_cls_local_mps"
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()

    if not args.data.is_dir() or not args.model.is_file():
        raise SystemExit("missing frozen dataset or pretrained model")
    if (args.project / args.name).exists():
        raise SystemExit(f"refusing to overwrite run: {args.project / args.name}")
    train_counts = count_split(args.data, "train")
    val_counts = count_split(args.data, "val")
    if min(*train_counts.values(), *val_counts.values()) < 1:
        raise SystemExit(f"empty training class: train={train_counts} val={val_counts}")

    import torch
    from ultralytics import YOLO

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS is not available")
    print(
        json.dumps(
            {
                "run": args.name,
                "device": args.device,
                "train": train_counts,
                "val": val_counts,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "epochs": args.epochs,
                "patience": args.patience,
                "seed": args.seed,
                "augs": "all_off",
                "transform": "white_letterbox_full_frame",
                "holdout_read": False,
                "auto_promote": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        cache=False,
        project=str(args.project),
        name=args.name,
        exist_ok=False,
        plots=True,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        warmup_epochs=0.5,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        scale=0.0,
        translate=0.0,
        erasing=0.0,
        auto_augment=None,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        bgr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        amp=True,
        val=True,
    )
    model = YOLO(str(args.model), task="classify")
    model.train(trainer=trainer_class(), **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
