import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm  

train_dir = "train_folder/train"
test_dir = "test_folder/test"
label_map_path = "train_folder/train/_classes.csv"
sample_submission_path = "animals_sample.csv"
output_dir = "submissions"
os.makedirs(output_dir, exist_ok=True)

label_map = {'Bear': 0, 'Bird': 1, 'Cat': 2, 'Dog': 3, 'Leopard': 4, 'Otter': 5}
label_map_df = pd.read_csv(label_map_path)
label_map_df.columns = label_map_df.columns.str.strip()
label_map_df['label'] = label_map_df[['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']].idxmax(axis=1)
label_map_df['label'] = label_map_df['label'].map(label_map)

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

train_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_aug = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
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

train_dataset = AnimalDataset(train_df, train_dir, transform=train_aug)
val_dataset = AnimalDataset(val_df, train_dir, transform=val_aug)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(input, target)
        pt = torch.exp(-ce_loss)
        return ((1-pt)**self.gamma * ce_loss).mean()

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)

def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss().to(device)

    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            images, targets_a, targets_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
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
                _, predicted = torch.max(outputs,1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        print(f"{model_name} Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{model_name}_best.pth")
            print("✅ Saved best model!")

        scheduler.step()

    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = [
    "tf_efficientnetv2_s_in21ft1k",
    "resnet50",
    "convnext_tiny"
]

results = {}
best_model_name = None
best_f1_score_val = -1

for model_name in model_names:
    print(f"\n🚀 Training model: {model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=6).to(device)
    model = train_model(model, train_loader, val_loader, epochs=20, lr=1e-4, model_name=model_name)

    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
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
    results[model_name] = val_f1

    if val_f1 > best_f1_score_val:
        best_f1_score_val = val_f1
        best_model_name = model_name

print("\n📊 Результаты моделей:")
for m, f in results.items():
    print(f"{m}: F1={f:.4f}")
print(f"\n🏆 Лучшая модель: {best_model_name} (F1={best_f1_score_val:.4f})")

best_model = timm.create_model(best_model_name, pretrained=False, num_classes=6).to(device)
best_model.load_state_dict(torch.load(f"{best_model_name}_best.pth"))
best_model.eval()

test_df = pd.read_csv(sample_submission_path)
test_dataset = AnimalDataset(test_df, test_dir, transform=val_aug, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_preds = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = best_model(images)
        best_preds.append(outputs.softmax(1).cpu().numpy())
best_preds = np.vstack(best_preds)
best_labels = np.argmax(best_preds, axis=1)

sub_best = pd.DataFrame({
    "idx": range(1, len(best_labels)+1),
    "label": best_labels
})
best_sub_path = os.path.join(output_dir, f"{best_model_name}_best_submission.csv")
sub_best.to_csv(best_sub_path, index=False)
print(f"✅ Saved best submission: {best_sub_path}")

ensemble_preds = []

for model_name in model_names:
    model = timm.create_model(model_name, pretrained=False, num_classes=6).to(device)
    model.load_state_dict(torch.load(f"{model_name}_best.pth"))
    model.eval()

    preds_model = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds_model.append(outputs.softmax(1).cpu().numpy())
    preds_model = np.vstack(preds_model)
    ensemble_preds.append(preds_model)

ensemble_mean = np.mean(ensemble_preds, axis=0)
final_preds = np.argmax(ensemble_mean, axis=1)

sub_ens = pd.DataFrame({
    "idx": range(1, len(final_preds)+1),
    "label": final_preds
})
ens_sub_path = os.path.join(output_dir, f"ensemble_submission.csv")
sub_ens.to_csv(ens_sub_path, index=False)
print(f"✅ Saved ensemble submission: {ens_sub_path}")