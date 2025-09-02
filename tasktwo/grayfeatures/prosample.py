import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import timm
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    train_dir = "train_folder/train"
    test_dir = "test_folder/test"
    label_map_path = "train_folder/train/_classes.csv"
    sample_submission_path = "animals_sample.csv"
    output_dir = "submissions"

    backbone = "convnext_tiny"
    num_classes = 6
    img_size = 224
    quant_levels = 50

    n_splits = 5
    epochs = 35
    batch_size = 32
    num_workers = 4
    lr = 1e-3
    weight_decay = 1e-4
    seed = 42
    use_amp = True
    tta_n = 8
    class_names = ["Bear", "Bird", "Cat", "Dog", "Leopard", "Otter"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def quantize_gray(img_gray_uint8: np.ndarray, levels: int) -> np.ndarray:
    arr = img_gray_uint8.astype(np.float32) / 255.0
    arr_q = np.floor(arr * (levels - 1)) / (levels - 1)
    out = (arr_q * 255.0).astype(np.uint8)
    return out


def get_train_transforms(cfg: Config):
    s = cfg.img_size
    return A.Compose([
        A.ToGray(p=1.0),
        A.RandomResizedCrop(height=s, width=s, scale=(0.7, 1.0), ratio=(0.8, 1.2), interpolation=cv2.INTER_CUBIC),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=25, interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0), p=0.4),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])


def get_valid_transforms(cfg: Config):
    s = cfg.img_size
    return A.Compose([
        A.ToGray(p=1.0),
        A.Resize(height=s, width=s, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])


def get_tta_transforms(cfg: Config) -> List[A.Compose]:
    s = cfg.img_size
    base = [
        A.ToGray(p=1.0),
        A.Resize(height=s, width=s, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ]
    tta_list = []
    tta_list.append(A.Compose(base))
    tta_list.append(A.Compose([A.HorizontalFlip(p=1.0)] + base))
    tta_list.append(A.Compose([A.VerticalFlip(p=1.0)] + base))
    tta_list.append(A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base))
    for deg in [-20, -10, 10, 20]:
        tta_list.append(A.Compose([A.Rotate(limit=(deg, deg), interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_REFLECT_101, p=1.0)] + base))
    return tta_list


class TracksDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, cfg: Config, mode: str = "train"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.cfg = cfg
        self.mode = mode
        self.train_tf = get_train_transforms(cfg)
        self.val_tf = get_valid_transforms(cfg)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = row["filename"]
        label = row.get("label", None)
        path = os.path.join(self.img_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = quantize_gray(img, self.cfg.quant_levels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.mode == "train":
            aug = self.train_tf(image=img)
        else:
            aug = self.val_tf(image=img)
        tensor = aug["image"]

        if label is None or self.mode == "test":
            return tensor, fname
        return tensor, int(label)


class TestImages(Dataset):
    def __init__(self, img_paths: List[str], cfg: Config, tta_tf: Optional[A.Compose] = None):
        self.img_paths = img_paths
        self.cfg = cfg
        self.tf = tta_tf if tta_tf is not None else get_valid_transforms(cfg)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        img = quantize_gray(img, self.cfg.quant_levels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        aug = self.tf(image=img)
        return aug["image"], os.path.basename(path)


class Classifier(nn.Module):
    def __init__(self, backbone: str, num_classes: int, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        if backbone == "convnext_tiny":
            self.model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        elif backbone == "swin_tiny":
            self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(cfg.output_dir, exist_ok=True)

    def _make_model(self):
        model = Classifier(self.cfg.backbone, self.cfg.num_classes, in_chans=1, pretrained=True)
        return model.to(self.device)

    def _make_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        ds_train = TracksDataset(train_df, self.cfg.train_dir, self.cfg, mode="train")
        ds_val = TracksDataset(val_df, self.cfg.train_dir, self.cfg, mode="val")

        counts = train_df["label"].value_counts().to_dict()
        weights = [1.0 / counts[int(l)] for l in train_df["label"].tolist()]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

        dl_train = DataLoader(ds_train, batch_size=self.cfg.batch_size, sampler=sampler,
                              num_workers=self.cfg.num_workers, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=self.cfg.num_workers, pin_memory=True)
        return dl_train, dl_val

    def _build_optim(self, model: nn.Module):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return optimizer, scheduler

    def _validate(self, model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
        model.eval()
        losses = []
        gts, prs = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                with autocast(enabled=self.cfg.use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                prs.extend(preds.tolist())
                gts.extend(labels.detach().cpu().numpy().tolist())
        f1 = f1_score(gts, prs, average="macro")
        return f1, float(np.mean(losses))

    def train_fold(self, fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[float, str]:
        model = self._make_model()
        optimizer, scheduler = self._build_optim(model)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.cfg.use_amp)

        dl_train, dl_val = self._make_loaders(train_df, val_df)

        best_f1 = 0.0
        best_path = os.path.join(self.cfg.output_dir, f"{self.cfg.backbone}_fold{fold}_best.pth")

        for epoch in range(self.cfg.epochs):
            model.train()
            epoch_losses = []
            gts, prs = [], []

            for imgs, labels in dl_train:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.cfg.use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                prs.extend(preds.tolist())
                gts.extend(labels.detach().cpu().numpy().tolist())

            scheduler.step()
            train_f1 = f1_score(gts, prs, average="macro")
            val_f1, val_loss = self._validate(model, dl_val, criterion)

            print(f"Fold {fold} | Epoch {epoch+1}/{self.cfg.epochs} | train_loss={np.mean(epoch_losses):.4f} train_f1={train_f1:.4f} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), best_path)

        print(f"Fold {fold} best val F1: {best_f1:.4f}")
        return best_f1, best_path

    def predict_test_ensemble(self, checkpoints: List[str]) -> List[int]:
        test_files = sorted([f for f in os.listdir(self.cfg.test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        test_paths = [os.path.join(self.cfg.test_dir, f) for f in test_files]

        tta_list = get_tta_transforms(self.cfg)
        probs_sum = None

        for ckpt in checkpoints:
            model = self._make_model()
            state = torch.load(ckpt, map_location=self.device)
            model.load_state_dict(state, strict=True)
            model.eval()

            fold_probs = None
            with torch.no_grad():
                for tta_tf in tta_list[: self.cfg.tta_n]:
                    ds = TestImages(test_paths, self.cfg, tta_tf)
                    dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=False,
                                    num_workers=self.cfg.num_workers, pin_memory=True)

                    tta_probs_batches = []
                    for imgs, _ in dl:
                        imgs = imgs.to(self.device)
                        with autocast(enabled=self.cfg.use_amp):
                            logits = model(imgs)
                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                        tta_probs_batches.append(probs)
                    tta_probs = np.vstack(tta_probs_batches)

                    if fold_probs is None:
                        fold_probs = tta_probs
                    else:
                        fold_probs += tta_probs

            fold_probs /= min(len(tta_list), self.cfg.tta_n)

            if probs_sum is None:
                probs_sum = fold_probs
            else:
                probs_sum += fold_probs

        probs_avg = probs_sum / len(checkpoints)
        final_preds = np.argmax(probs_avg, axis=1).tolist()
        return final_preds


def load_label_map(label_map_path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(label_map_path)
    cols = set(df.columns)
    df.columns = [c.strip() for c in df.columns]
    one_hot = set(cfg.class_names)

    if one_hot.issubset(df.columns):
        df["label_name"] = df[list(one_hot)].idxmax(axis=1)
        df["label"] = df["label_name"].map(cfg.class_to_idx).astype(int)
    elif "label" in df.columns:
        if df["label"].dtype == object:
            df["label"] = df["label"].map(lambda x: cfg.class_to_idx[str(x)])
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise ValueError(f"label_map csv has unexpected columns: {df.columns.tolist()}")

    if "filename" not in df.columns:
        raise ValueError("label_map must contain 'filename' column")

    df["exists"] = df["filename"].apply(lambda f: os.path.exists(os.path.join(cfg.train_dir, f)))
    missing = df.loc[~df["exists"]]
    if len(missing) > 0:
        print(f"Warning: {len(missing)} files from label_map not found in train_dir. They will be dropped.")
        df = df.loc[df["exists"]].copy()
    df = df.drop(columns=["exists"])

    return df[["filename", "label"]].reset_index(drop=True)


def run_training_kfold(cfg: Config) -> Tuple[List[str], List[float]]:
    seed_everything(cfg.seed)

    df = load_label_map(cfg.label_map_path, cfg)

    X = df["filename"].values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    trainer = Trainer(cfg)
    fold_paths = []
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Starting fold {fold}/{cfg.n_splits}")
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        va_df = df.iloc[val_idx].reset_index(drop=True)

        best_f1, best_path = trainer.train_fold(fold, tr_df, va_df)
        fold_paths.append(best_path)
        fold_scores.append(best_f1)

    print("All folds finished. F1 per fold:", fold_scores)
    print("Mean F1:", float(np.mean(fold_scores)))

    final_preds = trainer.predict_test_ensemble(fold_paths)
    os.makedirs(cfg.output_dir, exist_ok=True)
    sub_df = pd.DataFrame({
        "idx": range(1, len(final_preds) + 1),
        "label": final_preds,
    })
    sub_path = os.path.join(cfg.output_dir, f"{cfg.backbone}_submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")

    return fold_paths, fold_scores


if __name__ == "__main__":
    cfg = Config()
    run_training_kfold(cfg)