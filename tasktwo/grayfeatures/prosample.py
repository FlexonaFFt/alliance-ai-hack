import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

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
            image = cv2.merge([image, image, image])
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.is_test:
            return image, img_name
        label = self.df.iloc[idx]['label']
        return image, label

train_aug = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.3),
    A.CoarseDropout(max_holes=8, hole_height=32, hole_width=32, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

train_df, val_df = train_test_split(
    label_map_df,
    test_size=0.2,
    stratify=label_map_df['label'],
    random_state=42
)

class_counts = train_df['label'].value_counts().sort_index()
weights = 1. / class_counts
samples_weight = train_df['label'].map(weights).values
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = SoftTargetCrossEntropy().to(device)
    mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5, num_classes=6)
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            apply_mix = np.random.rand() < 0.5
            if apply_mix:
                images, labels = mixup_fn(images, labels)
            optimizer.zero_grad()
            outputs = model(images)
            if labels.dtype == torch.long:
                targets = nn.functional.one_hot(labels, num_classes=6).float().to(device)
            else:
                targets = labels
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        scheduler.step()
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"{model_name} Epoch {epoch+1}: Val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print("Saved best model")
    return model

train_dataset_rgb = AnimalDataset(train_df, train_dir, transform=train_aug, use_gray=False)
val_dataset_rgb = AnimalDataset(val_df, train_dir, transform=val_aug, use_gray=False)
train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=32, sampler=sampler)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=32, shuffle=False)
model_rgb = timm.create_model("convnext_base", pretrained=True, num_classes=6).to(device)
model_rgb = train_model(model_rgb, train_loader_rgb, val_loader_rgb, epochs=10, lr=1e-4, model_name="base_rgb")

train_dataset_gray = AnimalDataset(train_df, train_dir, transform=train_aug, use_gray=True)
val_dataset_gray = AnimalDataset(val_df, train_dir, transform=val_aug, use_gray=True)
train_loader_gray = DataLoader(train_dataset_gray, batch_size=32, sampler=sampler)
val_loader_gray = DataLoader(val_dataset_gray, batch_size=32, shuffle=False)
model_gray = timm.create_model("convnext_base", pretrained=True, num_classes=6).to(device)
model_gray = train_model(model_gray, train_loader_gray, val_loader_gray, epochs=10, lr=1e-4, model_name="base_gray")

test_df = pd.read_csv(sample_submission_path)
test_dataset_rgb = AnimalDataset(test_df, test_dir, transform=val_aug, is_test=True, use_gray=False)
test_dataset_gray = AnimalDataset(test_df, test_dir, transform=val_aug, is_test=True, use_gray=True)
test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=32, shuffle=False)
test_loader_gray = DataLoader(test_dataset_gray, batch_size=32, shuffle=False)

model_rgb.load_state_dict(torch.load("base_rgb_best.pth"))
model_gray.load_state_dict(torch.load("base_gray_best.pth"))
model_rgb.eval()
model_gray.eval()

def predict_and_save(model, loader, name):
    preds = []
    with torch.no_grad():
        for images, names in loader:
            images = images.to(device)
            outputs = model(images)
            preds.append(outputs.softmax(1).cpu().numpy())
    preds = np.vstack(preds)
    final_preds = np.argmax(preds, axis=1)
    sub_df = pd.DataFrame({"idx": range(1, len(final_preds)+1), "label": final_preds})
    sub_path = os.path.join(output_dir, f"{name}.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Saved submission: {sub_path}")
    return preds

preds_rgb = predict_and_save(model_rgb, test_loader_rgb, "base_rgb")
preds_gray = predict_and_save(model_gray, test_loader_gray, "base_gray")
final_preds = np.argmax((preds_rgb + preds_gray) / 2, axis=1)
sub_df = pd.DataFrame({"idx": range(1, len(final_preds)+1), "label": final_preds})
sub_path = os.path.join(output_dir, "base_rgb_gray_ensemble.csv")
sub_df.to_csv(sub_path, index=False)
print(f"Saved submission: {sub_path}")