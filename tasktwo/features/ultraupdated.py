import os
import random
import math
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import RandAugment
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

try:
    from timm.utils import ModelEma
    _HAS_MODEL_EMA = True
except Exception:
    _HAS_MODEL_EMA = False

from torch.optim.swa_utils import AveragedModel, update_bn, SWALR

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, default="train_folder/train")
parser.add_argument("--test_dir", type=str, default="test_folder/test")
parser.add_argument("--label_map_path", type=str, default="train_folder/train/_classes.csv")
parser.add_argument("--sample_submission_path", type=str, default="animals_sample.csv")
parser.add_argument("--output_dir", type=str, default="submissions")
parser.add_argument("--model_name", type=str, default="convnext_small")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=40)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args([])  


os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device)
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

seed_everything(args.seed)
label_map_df = pd.read_csv(args.label_map_path)
label_map_df.columns = label_map_df.columns.str.strip()
if 'label' in label_map_df.columns and 'filename' in label_map_df.columns:
    df_all = label_map_df.copy()
else:
    CLASS_NAMES = ['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']
    filename_col = None
    for c in label_map_df.columns:
        if 'file' in c.lower() or 'img' in c.lower() or 'name' in c.lower():
            filename_col = c
            break
    if filename_col is None:
        filename_col = label_map_df.columns[0]
    class_cols = [c for c in label_map_df.columns if c in CLASS_NAMES]
    if len(class_cols) == 0:
        raise RuntimeError("Couldn't find class columns in label_map CSV. Expect columns like Bear,Bird,Cat,...")
    label_map_df['label'] = label_map_df[class_cols].idxmax(axis=1)
    label_to_int = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    label_map_df['label'] = label_map_df['label'].map(label_to_int)
    df_all = label_map_df.rename(columns={filename_col: 'filename'})[['filename', 'label']]

df_all['filename'] = df_all['filename'].astype(str).str.strip()
unique_labels = sorted(df_all['label'].unique())
inv_label_map = {v: k for k, v in {name: idx for idx, name in enumerate(['Bear','Bird','Cat','Dog','Leopard','Otter'])}.items()}

class AnimalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['filename'])
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_name
        label = int(row['label'])
        return image, label

img_size = args.img_size
train_aug = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
    RandAugment(num_ops=2, magnitude=9),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

val_aug = transforms.Compose([
    transforms.Resize(int(img_size * 1.15)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

train_df, val_df = train_test_split(
    df_all,
    test_size=0.2,
    stratify=df_all['label'],
    random_state=args.seed
)

class_counts = train_df['label'].value_counts().sort_index()
weights = 1.0 / class_counts
samples_weight = train_df['label'].map(weights).values
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
train_dataset = AnimalDataset(train_df, args.train_dir, transform=train_aug)
val_dataset   = AnimalDataset(val_df,   args.train_dir, transform=val_aug)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        n = preds.size(-1)
        log_preds = self.logsoftmax(preds)
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_preds.mean(dim=-1)
        return ((1.0 - self.eps) * nll + self.eps * smooth_loss).mean()

def validate(model, loader, device, use_ema=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, p = torch.max(outputs, 1)
            preds.extend(p.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return f1_score(targets, preds, average='macro'), preds, targets

num_classes = len(unique_labels)
model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
steps_per_epoch = math.ceil(len(train_loader))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)

criterion = LabelSmoothingCrossEntropy(eps=0.1).to(device)
ema = None
if _HAS_MODEL_EMA:
    try:
        ema = ModelEma(model, decay=0.9997, device=device)
        print("ModelEma enabled.")
    except Exception as e:
        print("ModelEma not available:", e)
        ema = None
else:
    print("timm.ModelEma not detected; continuing without EMA.")

swa_model = AveragedModel(model)
swa_start = int(0.8 * args.epochs)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

best_f1 = 0.0
best_path = os.path.join(args.output_dir, f"{args.model_name}_best.pth")
ema_best_path = os.path.join(args.output_dir, f"{args.model_name}_ema_best.pth")
swa_path = os.path.join(args.output_dir, f"{args.model_name}_swa.pth")

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("Starting training on device:", device)
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=120)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        use_mixup = False
        if use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch < swa_start:
            scheduler.step()

        if ema is not None:
            try:
                ema.update(model)
            except Exception:
                pass

        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    if ema is not None:
        val_model = getattr(ema, "module", ema.ema)
    else:
        val_model = model


    val_f1, _, _ = validate(val_model, val_loader, device)
    print(f"Epoch {epoch+1} validation F1 (macro): {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(val_model.state_dict(), best_path)
        print("Saved best model:", best_path)
        if ema is not None:
            torch.save(getattr(ema, "module", ema.ema).state_dict(), ema_best_path)
            print("Saved EMA best model:", ema_best_path)

    if epoch >= swa_start:
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print(f"SWA update at epoch {epoch+1}")

try:
    update_bn(train_loader, swa_model)
    torch.save(swa_model.state_dict(), swa_path)
    print("Saved SWA model:", swa_path)
except Exception as e:
    print("SWA finalization failed:", e)

print("Training finished. Best F1:", best_f1)
predict_model = timm.create_model(args.model_name, pretrained=False, num_classes=num_classes).to(device)
loaded = False
if os.path.exists(ema_best_path):
    try:
        predict_model.load_state_dict(torch.load(ema_best_path, map_location=device))
        print("Loaded EMA best for prediction.")
        loaded = True
    except Exception:
        pass
if (not loaded) and os.path.exists(swa_path):
    try:
        predict_model.load_state_dict(torch.load(swa_path, map_location=device))
        print("Loaded SWA model for prediction.")
        loaded = True
    except Exception:
        pass
if (not loaded) and os.path.exists(best_path):
    predict_model.load_state_dict(torch.load(best_path, map_location=device))
    print("Loaded best model for prediction.")

predict_model.eval()

tta_transforms = [
    val_aug,
    transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize(int(img_size * 1.3)),
        transforms.FiveCrop(img_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(c) for c in crops])),
        transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(t) for t in tensors])),
    ])
]

test_df = pd.read_csv(args.sample_submission_path)
if 'filename' in test_df.columns:
    test_filenames = test_df['filename'].astype(str).str.strip().tolist()
else:
    if 'idx' in test_df.columns:
        test_filenames = test_df['idx'].astype(str).str.strip().tolist()
    else:
        test_filenames = test_df.iloc[:,0].astype(str).str.strip().tolist()

test_dataset = AnimalDataset(pd.DataFrame({'filename': test_filenames}), args.test_dir, transform=val_aug, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
all_probs = None

with torch.no_grad():
    for t_idx, tform in enumerate(tta_transforms):
        print("Running TTA transform", t_idx+1)
        if any(isinstance(x, transforms.FiveCrop) for x in [getattr(tform, 'transforms', None)]):
            test_dataset.transform = val_aug
        else:
            test_dataset.transform = tform

        probs_list = []
        for images, fnames in tqdm(test_loader, desc=f"TTA {t_idx+1}", ncols=120):
            images = images.to(device)
            outputs = predict_model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            probs_list.append(probs)
        probs_stack = np.vstack(probs_list)
        if all_probs is None:
            all_probs = probs_stack
        else:
            all_probs += probs_stack

final_probs = all_probs / len(tta_transforms)
final_preds = np.argmax(final_probs, axis=1)

out_df = pd.DataFrame({
    "idx": range(1, len(final_preds) + 1),
    "label": final_preds
})
out_path = os.path.join(args.output_dir, f"{args.model_name}_submission.csv")
out_df.to_csv(out_path, index=False)
print("Saved submission to", out_path)