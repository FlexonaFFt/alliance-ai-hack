import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

train_dir = 'train_folder/train'
test_dir = 'test_folder/test'
label_map_path = '_classes.csv'
sample_submission_path = 'animals_sample.csv'

label_map_df = pd.read_csv(label_map_path)
label_map_df.columns = label_map_df.columns.str.strip()
label_map = {'Bear': 0, 'Bird': 1, 'Cat': 2, 'Dog': 3, 'Leopard': 4, 'Otter': 5}
label_map_df['label'] = label_map_df[['Bear', 'Bird', 'Cat', 'Dog', 'Leopard', 'Otter']].idxmax(axis=1)
label_map_df['label'] = label_map_df['label'].map(label_map)


strong_aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),  
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


medium_aug = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

light_aug = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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

train_df, val_df = train_test_split(
    label_map_df,
    test_size=0.2,
    stratify=label_map_df['label'],
    random_state=42
)

train_dataset = AnimalDataset(train_df, train_dir, transform=strong_aug)
val_dataset = AnimalDataset(val_df, train_dir, transform=light_aug)

class_counts = train_df['label'].value_counts().sort_index()
weights = 1. / class_counts
samples_weight = train_df['label'].map(weights).values
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 6)
model = model.to(device)

class_weights = compute_class_weight('balanced', classes=np.arange(6), y=label_map_df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
def train_phase(model, train_loader, val_loader, optimizer, scheduler, epochs, phase_name=""):
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
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
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        print(f"{phase_name} Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")
        if scheduler:
            scheduler.step()

for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
train_phase(model, train_loader, val_loader, optimizer, scheduler=None, epochs=5, phase_name="Phase1")

train_dataset.transform = medium_aug
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
train_phase(model, train_loader, val_loader, optimizer, scheduler, epochs=15, phase_name="Phase2")

train_dataset.transform = light_aug
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
train_phase(model, train_loader, val_loader, optimizer, scheduler=None, epochs=5, phase_name="Phase3")
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

tta_transforms = [
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
]

test_df = pd.read_csv(sample_submission_path)
test_dataset = AnimalDataset(test_df, test_dir, transform=light_aug, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
all_preds = []
with torch.no_grad():
    for images, _ in tqdm(test_loader, desc="Testing TTA"):
        images = images.to(device)
        outputs = model(images)
        all_preds.append(outputs.softmax(1).cpu().numpy())


all_preds = np.vstack(all_preds)  
final_preds = np.argmax(all_preds, axis=1)

submission = pd.DataFrame({
    'idx': range(1, len(final_preds)+1),
    'label': final_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")