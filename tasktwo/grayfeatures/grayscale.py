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
    A.Resize(height=256, width=256),
    A.RandomCrop(height=224, width=224),  # Замена RandomResizedCrop
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(input, target)
        pt = torch.exp(-ce_loss)
        return ((1-pt)**self.gamma * ce_loss).mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    criterion = FocalLoss().to(device)

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

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
        print(f"{model_name} Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print("✅ Saved best model!")

    return model

# RGB модель
train_dataset_rgb = AnimalDataset(train_df, train_dir, transform=train_aug, use_gray=False)
val_dataset_rgb = AnimalDataset(val_df, train_dir, transform=val_aug, use_gray=False)

train_loader_rgb = DataLoader(train_dataset_rgb, batch_size=32, sampler=sampler)
val_loader_rgb = DataLoader(val_dataset_rgb, batch_size=32, shuffle=False)

model_rgb = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=6).to(device)
model_rgb = train_model(model_rgb, train_loader_rgb, val_loader_rgb, epochs=20, lr=1e-4, model_name="swin_rgb")

# GRAY модель
train_dataset_gray = AnimalDataset(train_df, train_dir, transform=train_aug, use_gray=True)
val_dataset_gray = AnimalDataset(val_df, train_dir, transform=val_aug, use_gray=True)

train_loader_gray = DataLoader(train_dataset_gray, batch_size=32, sampler=sampler)
val_loader_gray = DataLoader(val_dataset_gray, batch_size=32, shuffle=False)

model_gray = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=6).to(device)
model_gray = train_model(model_gray, train_loader_gray, val_loader_gray, epochs=20, lr=1e-4, model_name="swin_gray")

test_df = pd.read_csv(sample_submission_path)
test_dataset_rgb = AnimalDataset(test_df, test_dir, transform=val_aug, is_test=True, use_gray=False)
test_dataset_gray = AnimalDataset(test_df, test_dir, transform=val_aug, is_test=True, use_gray=True)

test_loader_rgb = DataLoader(test_dataset_rgb, batch_size=32, shuffle=False)
test_loader_gray = DataLoader(test_dataset_gray, batch_size=32, shuffle=False)

model_rgb.load_state_dict(torch.load("swin_rgb_best.pth"))
model_gray.load_state_dict(torch.load("swin_gray_best.pth"))

model_rgb.eval()
model_gray.eval()

tta_preds = []
with torch.no_grad():
    preds_rgb, preds_gray = [], []

    for (images_rgb, _), (images_gray, _) in zip(test_loader_rgb, test_loader_gray):
        images_rgb = images_rgb.to(device)
        images_gray = images_gray.to(device)

        outputs_rgb = model_rgb(images_rgb)
        outputs_gray = model_gray(images_gray)

        preds_rgb.append(outputs_rgb.softmax(1).cpu().numpy())
        preds_gray.append(outputs_gray.softmax(1).cpu().numpy())

    preds_rgb = np.vstack(preds_rgb)
    preds_gray = np.vstack(preds_gray)
    final_preds = np.argmax((preds_rgb + preds_gray) / 2, axis=1)

sub_df = pd.DataFrame({
    "idx": range(1, len(final_preds)+1),
    "label": final_preds
})

sub_path = os.path.join(output_dir, "swin_rgb_gray_ensemble.csv")
sub_df.to_csv(sub_path, index=False)
print(f"✅ Saved submission: {sub_path}")