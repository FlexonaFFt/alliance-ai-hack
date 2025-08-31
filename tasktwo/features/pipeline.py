import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

from torchvision import transforms

# -------------------- Dataset --------------------
class AnimalDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_name
        label = self.df.iloc[idx]['label']
        return image, label

# -------------------- Augmentations --------------------
strong_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.2)   
])

medium_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.1)   
])

light_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -------------------- Training Function --------------------
def train_one_stage(model, train_loader, val_loader, epochs, lr, device, stage_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
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
        print(f"{stage_name} Epoch {epoch+1}/{epochs}: Loss={running_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{stage_name}_best.pth")
        scheduler.step()

    return model

# -------------------- KFold Training --------------------
def train_kfold(df, img_dir, n_splits=5, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"===== Fold {fold+1}/{n_splits} =====")
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        train_dataset = AnimalDataset(train_df, img_dir, transform=strong_aug)
        val_dataset = AnimalDataset(val_df, img_dir, transform=light_aug)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=6).to(device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True
        model = train_one_stage(model, train_loader, val_loader, epochs=5, lr=1e-4, device=device, stage_name=f"fold{fold}_stage1")


        for param in model.parameters():
            param.requires_grad = True
        train_dataset.transform = medium_aug
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = train_one_stage(model, train_loader, val_loader, epochs=15, lr=5e-5, device=device, stage_name=f"fold{fold}_stage2")


        train_dataset.transform = light_aug
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = train_one_stage(model, train_loader, val_loader, epochs=5, lr=1e-5, device=device, stage_name=f"fold{fold}_stage3")

        fold_models.append(model)

    return fold_models

# -------------------- Inference with Ensemble --------------------
def predict_ensemble(models, test_df, img_dir, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = AnimalDataset(test_df, img_dir, transform=light_aug, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for model in models:
            model.eval()
            preds = []
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds.append(outputs.softmax(1).cpu().numpy())
            preds = np.vstack(preds)
            all_preds.append(preds)

    final_preds = np.argmax(np.mean(all_preds, axis=0), axis=1)
    return final_preds


if __name__ == "__main__":
    train_dir = "train_folder/train"
    test_dir = "test_folder/test"
    label_map_path = "train_folder/train/_classes.csv"
    sample_submission_path = "animals_sample.csv"
    output_dir = "submissions_updated"
    os.makedirs(output_dir, exist_ok=True)

    label_map = {'Bear': 0, 'Bird': 1, 'Cat': 2, 'Dog': 3, 'Leopard': 4, 'Otter': 5}
    label_map_df = pd.read_csv(label_map_path)
    label_map_df.columns = label_map_df.columns.str.strip()
    label_map_df['label'] = label_map_df[['Bear','Bird','Cat','Dog','Leopard','Otter']].idxmax(axis=1)
    label_map_df['label'] = label_map_df['label'].map(label_map)
    fold_models = train_kfold(label_map_df, train_dir, n_splits=5, batch_size=32)

    test_df = pd.read_csv(sample_submission_path)
    final_preds = predict_ensemble(fold_models, test_df, test_dir, batch_size=32)

    sub_df = pd.DataFrame({
        "idx": range(1, len(final_preds)+1),
        "label": final_preds
    })
    sub_path = os.path.join(output_dir, "ensemble_submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"✅ Saved submission: {sub_path}")