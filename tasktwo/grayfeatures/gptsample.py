import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import f1_score, roc_auc_score, classification_report
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False
    raise ImportError("This script requires 'timm' to be installed.")

import torchvision.transforms as T
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster


def find_classes_in_folder(root: Path) -> Tuple[List[str], Dict[str, int]]:
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def build_dataframe_from_imagefolder(root: Path) -> pd.DataFrame:
    classes, class_to_idx = find_classes_in_folder(root)
    rows = []
    for cls in classes:
        for img_path in (root / cls).rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                rows.append({"filepath": str(img_path), "label": class_to_idx[cls]})
    df = pd.DataFrame(rows)
    return df

class TracksDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            img_size: int = 224,
            is_train: bool = True,
            grayscale_p: float = 0.5,
            use_alb: bool = True,
            mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
            std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
            clahe_p: float = 0.4,
            brightness_contrast_p: float = 0.7,
            blur_p: float = 0.2,
            erasing_p: float = 0.25,
            rotate_limit: int = 15,
            center_crop_val: float = 1.0,
    ):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.is_train = is_train
        self.use_alb = use_alb and _HAS_ALB

        if self.use_alb:
            train_aug = [
                A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=rotate_limit, border_mode=0, p=0.3),
                A.RandomBrightnessContrast(p=brightness_contrast_p),
                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=clahe_p),
                A.GaussianBlur(blur_limit=(3, 7), p=blur_p),
                A.OneOf([A.Cutout(num_holes=8, max_h_size=img_size//12, max_w_size=img_size//12, p=1.0),
                         A.CoarseDropout(max_holes=12, max_height=img_size//10, max_width=img_size//10, p=1.0)], p=0.2),
                A.ToGray(p=grayscale_p),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
            val_aug = [
                A.LongestMaxSize(max_size=int(img_size * center_crop_val)),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
                A.CenterCrop(img_size, img_size, p=1.0),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
            self.alb_transform = A.Compose(train_aug if is_train else val_aug)
        else:
            if is_train:
                self.tv_transform = T.Compose([
                    T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(p=0.2),
                    T.RandomRotation(rotate_limit),
                    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                    T.RandomGrayscale(p=grayscale_p),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])
            else:
                self.tv_transform = T.Compose([
                    T.Resize(int(img_size * center_crop_val)),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std),
                ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.use_alb:
            img = self.alb_transform(image=img)["image"]
        else:
            img = self.tv_transform(Image.fromarray(img))

        return img, label

class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def build_model(model_name: str, num_classes: int, drop_rate: float = 0.2, drop_path_rate: float = 0.1):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate
    )
    return model

def compute_metrics(y_true: List[int], y_pred: List[int], y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    metrics["accuracy"] = (y_true == y_pred).mean().item()
    if _HAS_SK:
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        if y_proba is not None:
            try:
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    proba = y_proba if y_proba.ndim == 1 else y_proba[:, 0]
                    metrics["roc_auc"] = roc_auc_score(y_true, proba)
                else:
                    metrics["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
            except Exception:
                pass
    return metrics

def get_sampler(df: pd.DataFrame) -> Optional[WeightedRandomSampler]:
    class_counts = df["label"].value_counts().sort_index().values.astype(float)
    weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = df["label"].map(lambda x: weights[int(x)]).values
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)
    return sampler


def get_transforms_flags():
    return {"albumentations": _HAS_ALB}


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, grad_accum_steps=1, max_norm=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast():
            logits = model(images)
            loss = criterion(logits, targets) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            if max_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * grad_accum_steps
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)

    train_acc = correct / max(total, 1)
    return running_loss / max(len(loader), 1), train_acc


@torch.no_grad()
def tta_predict(model, images, tta: int = 4):
    logits_sum = model(images)
    if tta >= 2:
        logits_sum += model(torch.flip(images, dims=[3]))
    if tta >= 3:
        logits_sum += model(torch.flip(images, dims=[2]))
    if tta >= 4:
        logits_sum += model(torch.flip(torch.flip(images, dims=[2]), dims=[3]))
    return logits_sum / float(min(tta, 4))


@torch.no_grad()
def evaluate(model, loader, criterion, device, tta: int = 4):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []
    all_proba = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = tta_predict(model, images, tta=tta)
        loss = criterion(logits, targets)

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        losses.append(loss.item())
        all_targets.append(targets.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_proba.append(probs.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_proba = np.concatenate(all_proba)

    metrics = compute_metrics(all_targets, all_preds, all_proba)
    metrics["loss"] = float(np.mean(losses))
    return metrics, all_targets, all_preds, all_proba

def main():
    parser = argparse.ArgumentParser(description="Animal Tracks Classifier")
    parser.add_argument("--data_csv", type=str, default="", help="CSV with columns: filepath,label")
    parser.add_argument("--train_dir", type=str, default="", help="ImageFolder-like directory: class_subdir/images")
    parser.add_argument("--val_csv", type=str, default="", help="Optional CSV for validation")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--center_crop_val", type=float, default=1.0, help="Resize factor before center crop for val")
    parser.add_argument("--val_size", type=float, default=0.15, help="If no val_csv: split portion for val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grayscale_p", type=float, default=0.5)
    parser.add_argument("--use_alb", action="store_true", help="Use Albumentations if available")
    parser.add_argument("--model", type=str, default="swin_tiny") # Модель
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--drop_path", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--tta", type=int, default=4, help="1-4")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_name", type=str, default="best_model.pth")
    parser.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler")
    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.data_csv:
        df = pd.read_csv(args.data_csv)
        assert {"filepath", "label"}.issubset(df.columns), "CSV must have columns: filepath,label"
    elif args.train_dir:
        df = build_dataframe_from_imagefolder(Path(args.train_dir))
    else:
        raise ValueError("Provide either --data_csv or --train_dir.")

    if args.val_csv:
        df_val = pd.read_csv(args.val_csv)
        assert {"filepath", "label"}.issubset(df_val.columns), "val CSV must have columns: filepath,label"
        df_train = df
    else:
        if not _HAS_SK:
            msk = np.random.rand(len(df)) > args.val_size
            df_train = df[msk].reset_index(drop=True)
            df_val = df[~msk].reset_index(drop=True)
        else:
            df_train, df_val = train_test_split(
                df, test_size=args.val_size, random_state=args.seed, stratify=df["label"]
            )
            df_train = df_train.reset_index(drop=True)
            df_val = df_val.reset_index(drop=True)

    num_classes = int(df["label"].nunique())
    ds_train = TracksDataset(
        df_train,
        img_size=args.img_size,
        is_train=True,
        grayscale_p=args.grayscale_p,
        use_alb=args.use_alb,
        center_crop_val=args.center_crop_val,
    )
    ds_val = TracksDataset(
        df_val,
        img_size=args.img_size,
        is_train=False,
        grayscale_p=0.0,
        use_alb=args.use_alb,
        center_crop_val=args.center_crop_val,
    )

    if args.balanced:
        sampler = get_sampler(df_train)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model(args.model, num_classes=num_classes, drop_rate=args.drop_rate, drop_path_rate=args.drop_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    alpha = None
    if args.balanced:
        class_counts = df_train["label"].value_counts().sort_index().values.astype(float)
        inv_freq = 1.0 / np.maximum(class_counts, 1.0)
        inv_freq = inv_freq / inv_freq.sum() * len(inv_freq)  # normalize
        alpha = torch.tensor(inv_freq, dtype=torch.float32, device=device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=alpha, gamma=2.0, reduction="mean")
    else:
        criterion = nn.CrossEntropyLoss(weight=alpha)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return float(epoch + 1) / float(max(1, args.warmup_epochs))
        progress = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    scaler = GradScaler()

    best_acc = -1.0
    best_f1 = -1.0
    best_path = os.path.join(args.out_dir, args.save_name)

    history = []
    for epoch in range(args.epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            grad_accum_steps=args.accum, max_norm=args.max_norm
        )
        scheduler.step()
        metrics, y_true, y_pred, y_proba = evaluate(model, val_loader, criterion, device, tta=args.tta)
        elapsed = time.time() - start

        log = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_acc": train_acc,
            **metrics,
            "time_sec": elapsed,
        }
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

        primary = metrics.get("f1_macro", metrics.get("accuracy", 0.0))
        if primary >= best_f1:
            best_f1 = primary
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "mean": (0.485, 0.456, 0.406),
                    "std": (0.229, 0.224, 0.225),
                },
                best_path
            )
            print(f"Saved best checkpoint to: {best_path} (f1_macro={best_f1:.4f})")

    hist_path = os.path.join(args.out_dir, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"History saved to: {hist_path}")


if __name__ == "__main__":
    main()
