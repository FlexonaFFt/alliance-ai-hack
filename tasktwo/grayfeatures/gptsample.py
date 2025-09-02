import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_dir = "train_folder/train"
test_dir = "test_folder/test"
label_map_path = "train_folder/train/_classes.csv"
sample_submission_path = "animals_sample.csv"
output_dir = "submits"
os.makedirs(output_dir, exist_ok=True)
label_map = {'Bear': 0, 'Bird': 1, 'Cat': 2, 'Dog': 3, 'Leopard': 4, 'Otter': 5}
label_map_df = pd.read_csv(label_map_path)
label_map_df.columns = label_map_df.columns.str.strip()
label_map_df['label'] = label_map_df[['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']].idxmax(axis=1)
label_map_df['label'] = label_map_df['label'].map(label_map)
seed = 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(seed)

train_transforms = A.Compose([
    A.RandomResizedCrop(224,224,scale=(0.6,1.0),ratio=(0.9,1.1)),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.06,scale_limit=0.15,rotate_limit=20,p=0.4),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0,50.0)),
        A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5))
    ],p=0.3),
    A.ColorJitter(brightness=0.25,contrast=0.25,saturation=0.25,hue=0.05,p=0.5),
    A.CLAHE(p=0.2),
    A.CoarseDropout(max_holes=1,max_height=48,max_width=48,min_holes=1,fill_value=0,p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
    ToTensorV2()
])
val_transforms = A.Compose([
    A.Resize(256,256),
    A.CenterCrop(224,224),
    A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
    ToTensorV2()
])

class AnimalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False, use_gray=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.use_gray = use_gray
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        if self.use_gray:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.zeros((256,256),dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                image = np.zeros((256,256,3),dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.is_test:
            return image, img_name
        label = int(self.df.iloc[idx]['label'])
        return image, label

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * smooth_loss).mean()

class ModelEMA:
    def __init__(self, model, decay=0.9997):
        self.decay = decay
        self.shadow = {}
        self.collected_params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().cpu().clone()
                self.collected_params.append(name)
    def update(self, model):
        for name, p in model.named_parameters():
            if name in self.shadow:
                new = p.detach().cpu()
                self.shadow[name] = (1.0 - self.decay) * new + self.decay * self.shadow[name]
    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().cpu().clone()
                p.data.copy_(self.shadow[name].to(p.device))
    def restore(self, model):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_one_fold(fold, train_idx, val_idx, df, img_dir, use_gray, model_name, epochs=12, batch_size=32, lr=1e-3):
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    class_counts = train_df['label'].value_counts().sort_index()
    weights = 1.0 / class_counts
    samples_weight = train_df['label'].map(weights).values
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
    train_ds = AnimalDataset(train_df, img_dir, transform=train_transforms, use_gray=use_gray)
    val_ds = AnimalDataset(val_df, img_dir, transform=val_transforms, use_gray=use_gray)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = timm.create_model(model_name, pretrained=True, num_classes=6)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1).to(device)
    scaler = torch.cuda.amp.GradScaler()
    ema = ModelEMA(model, decay=0.9997)
    best_f1 = 0.0
    best_path = os.path.join(output_dir, f"fold{fold}_{'gray' if use_gray else 'rgb'}_best.pth")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Fold{fold} {'GRAY' if use_gray else 'RGB'} Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            running_loss += loss.item() * images.size(0)
        model.eval()
        preds = []
        targets = []
        ema.apply_shadow(model)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    outputs = model(images)
                _, p = torch.max(outputs, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(labels.cpu().numpy())
        ema.restore(model)
        val_f1 = f1_score(targets, preds, average="macro")
        avg_loss = running_loss / len(train_loader.dataset)
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
        if epoch==epochs-1:
            torch.save(model.state_dict(), os.path.join(output_dir, f"fold{fold}_{'gray' if use_gray else 'rgb'}_last.pth"))
    return best_path

def tta_predict(model, loader, device, tta_transforms):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in loader:
            bs = images.size(0)
            images = images.to(device)
            out = torch.zeros((bs,6), device=device)
            for tf in tta_transforms:
                if tf == "none":
                    inp = images
                elif tf == "hflip":
                    inp = torch.flip(images, dims=[3])
                elif tf == "vflip":
                    inp = torch.flip(images, dims=[2])
                else:
                    inp = images
                with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                    out += model(inp)
            out = out / len(tta_transforms)
            preds.append(out.softmax(1).cpu().numpy())
    return np.vstack(preds)

df = label_map_df.copy()
df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
model_name = "convnext_base.fb_in22k_ft_in1k"
fold_paths_rgb = []
fold_paths_gray = []
fold = 0
for train_idx, val_idx in skf.split(df['filename'], df['label']):
    p_rgb = train_one_fold(fold, train_idx, val_idx, df, train_dir, use_gray=False, model_name=model_name, epochs=12, batch_size=24, lr=1e-3)
    p_gray = train_one_fold(fold, train_idx, val_idx, df, train_dir, use_gray=True, model_name=model_name, epochs=12, batch_size=24, lr=1e-3)
    fold_paths_rgb.append(p_rgb)
    fold_paths_gray.append(p_gray)
    fold += 1

test_df = pd.read_csv(sample_submission_path)
test_dataset_rgb = AnimalDataset(test_df, test_dir, transform=val_transforms, is_test=True, use_gray=False)
test_dataset_gray = AnimalDataset(test_df, test_dir, transform=val_transforms, is_test=True, use_gray=True)
test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader_gray = DataLoader(test_dataset_gray, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
tta_transforms = ["none","hflip"]
all_preds = []
for fold_idx in range(len(fold_paths_rgb)):
    model_rgb = timm.create_model(model_name, pretrained=False, num_classes=6)
    model_rgb.load_state_dict(torch.load(fold_paths_rgb[fold_idx], map_location="cpu"))
    model_rgb.to(device)
    model_rgb.eval()
    model_gray = timm.create_model(model_name, pretrained=False, num_classes=6)
    model_gray.load_state_dict(torch.load(fold_paths_gray[fold_idx], map_location="cpu"))
    model_gray.to(device)
    model_gray.eval()
    preds_rgb = tta_predict(model_rgb, test_loader_rgb, device, tta_transforms)
    preds_gray = tta_predict(model_gray, test_loader_gray, device, tta_transforms)
    preds = (preds_rgb + preds_gray) / 2.0
    all_preds.append(preds)
final_preds = np.mean(all_preds, axis=0)
final_labels = np.argmax(final_preds, axis=1)
sub_df = pd.DataFrame({
    "idx": range(1, len(final_labels)+1),
    "label": final_labels
})
sub_path = os.path.join(output_dir, "convnext_ensemble_folds_rgb_gray_tta.csv")
sub_df.to_csv(sub_path, index=False)
print("Saved", sub_path)
