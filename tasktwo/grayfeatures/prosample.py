import os
import random
import math
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TracksDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, mode: str = 'train',
                 transform=None, quantize_levels: int = 50):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform
        self.quantize_levels = quantize_levels

    def __len__(self):
        return len(self.df)

    def quantize(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        arr_q = np.floor(arr * (self.quantize_levels - 1)) / (self.quantize_levels - 1)
        arr_q = (arr_q * 255.0).astype(np.uint8)
        return Image.fromarray(arr_q, mode='L')

    def __getitem__(self, idx: int):
        row = self.df.loc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img = Image.open(img_path).convert('L')
        img = self.quantize(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        if self.mode == 'test':
            return img, row['id'] if 'id' in row else row['filename']
        return img, int(row['label'])


def get_train_transforms(size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30, resample=Image.BICUBIC),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0)
    ])


def get_val_transforms(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.25])
    ])


class ConvNextTinyWrapper(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True, in_chans: int = 1):
        super().__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

    def forward(self, x):
        return self.model(x)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        log_prob = torch.nn.functional.log_softmax(preds, dim=-1)
        nll = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 output_dir: str = 'outputs',
                 model_name: str = 'convnext_tiny'):
        self.model = model.to(device)
        self.device = device
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(output_dir, exist_ok=True)

    def train_fold(self, train_df: pd.DataFrame, val_df: pd.DataFrame, img_dir: str,
                   epochs: int = 20, batch_size: int = 32, lr: float = 1e-3,
                   weight_decay: float = 1e-4, num_workers: int = 4):

        train_ds = TracksDataset(train_df, img_dir, mode='train', transform=get_train_transforms())
        val_ds = TracksDataset(val_df, img_dir, mode='val', transform=get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                        steps_per_epoch=len(train_loader), epochs=epochs)

        best_f1 = 0.0
        best_path = None

        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            all_preds = []
            all_targets = []

            for imgs, targets in train_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(imgs)
                loss = criterion(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.detach().cpu().numpy().tolist())

            train_f1 = f1_score(all_targets, all_preds, average='macro')

            val_f1, val_loss = self.validate(val_loader, criterion)

            print(f"Epoch {epoch+1}/{epochs} — train_loss={np.mean(train_losses):.4f}, train_f1={train_f1:.4f} | val_loss={val_loss:.4f}, val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_path = os.path.join(self.output_dir, f'best_model_f1_{best_f1:.4f}.pth')
                torch.save(self.model.state_dict(), best_path)

        return best_f1, best_path

    def validate(self, val_loader: DataLoader, criterion):
        self.model.eval()
        losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(imgs)
                loss = criterion(logits, targets)
                losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.detach().cpu().numpy().tolist())
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        return val_f1, np.mean(losses)

    def predict_tta(self, img_paths: List[str], tta_transforms: List[transforms.Compose], batch_size: int = 32):
        self.model.eval()
        probs_sum = None
        for t in tta_transforms:
            ds = TestDataset(img_paths, transform=t)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
            preds_list = []
            with torch.no_grad():
                for imgs in dl:
                    imgs = imgs.to(self.device)
                    logits = self.model(imgs)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds_list.append(probs)
            probs_full = np.vstack(preds_list)
            if probs_sum is None:
                probs_sum = probs_full
            else:
                probs_sum += probs_full
        probs_avg = probs_sum / len(tta_transforms)
        return probs_avg


class TestDataset(Dataset):
    def __init__(self, img_paths: List[str], transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert('L')
        arr = np.array(img).astype(np.float32) / 255.0
        arr_q = np.floor(arr * 49) / 49.0
        img_q = Image.fromarray((arr_q * 255).astype(np.uint8), mode='L')
        if self.transform:
            img = self.transform(img_q)
        else:
            img = TF.to_tensor(img_q)
        return img


def run_training_kfold(train_dir: str,
                       label_map_path: str,
                       out_dir: str = 'submissions',
                       n_splits: int = 5,
                       epochs: int = 20,
                       batch_size: int = 32,
                       lr: float = 1e-3,
                       seed: int = 42):

    seed_everything(seed)
    df_map = pd.read_csv(label_map_path)
    class_cols = ['Bear', 'Bird', 'Cat', 'Wolf', 'Leopard', 'Otter']
    if set(class_cols).issubset(df_map.columns):
        df_map['label'] = df_map[class_cols].idxmax(axis=1).map({c: i for i, c in enumerate(class_cols)})
    elif 'label' in df_map.columns:
        pass
    else:
        raise ValueError('label_map csv has unexpected columns')

    if 'filename' not in df_map.columns:
        raise ValueError('label_map must contain filename column')

    if 'id' not in df_map.columns:
        df_map['id'] = df_map['filename'].apply(lambda x: Path(x).stem)

    X = df_map['filename'].values
    y = df_map['label'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_results = []
    model_paths = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Starting fold {fold+1}/{n_splits}")
        train_df = df_map.iloc[tr_idx].reset_index(drop=True)
        val_df = df_map.iloc[val_idx].reset_index(drop=True)

        model = ConvNextTinyWrapper(num_classes=6, pretrained=True, in_chans=1)
        trainer = Trainer(model, device, output_dir=out_dir, model_name='convnext_tiny')

        best_f1, best_path = trainer.train_fold(train_df, val_df, img_dir=train_dir,
                                                epochs=epochs, batch_size=batch_size, lr=lr)
        print(f"Fold {fold+1} best val F1: {best_f1:.4f}")
        fold_results.append(best_f1)
        model_paths.append(best_path)

    print("All folds finished. F1 per fold:", fold_results)
    print("Mean F1:", np.mean(fold_results))
    return model_paths, fold_results


def save_submission(final_preds: List[int], output_dir: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)
    sub_df = pd.DataFrame({
        "idx": range(1, len(final_preds)+1),
        "label": final_preds
    })
    sub_path = os.path.join(output_dir, f"{model_name}_submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == '__main__':
    train_dir = "train_folder/train"
    label_map_path = "train_folder/train/_classes.csv"
    out_dir = "submissions"

    model_paths, fold_results = run_training_kfold(train_dir, label_map_path,
                                                   out_dir, n_splits=5, epochs=20, batch_size=32, lr=1e-3)

    print('Done')
