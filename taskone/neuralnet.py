import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

train = pd.read_csv('ctr_train.csv', nrows=5_000_000)
test = pd.read_csv('ctr_test.csv')
features = [c for c in test.columns if c not in ['id']]

cat_maps = {col: {val: i for i, val in enumerate(train[col].astype(str).unique())} for col in features}
for col in features:
    train[col] = train[col].astype(str).map(cat_maps[col])
    test[col] = test[col].astype(str).map(lambda x: cat_maps[col].get(x, 0))

class CTRDataset(Dataset):
    def __init__(self, df, features, target=None):
        self.X = df[features].values
        self.y = df[target].values if target else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.long)
        if self.y is not None:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, y
        return x

class CTRNet(nn.Module):
    def __init__(self, cat_maps, emb_dim=16):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(len(cat_maps[col]), emb_dim) for col in features
        ])
        self.bn = nn.BatchNorm1d(len(features) * emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(len(features) * emb_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embs, dim=1)
        x = self.bn(x)
        return self.fc(x).squeeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CTRNet(cat_maps).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)  
loss_fn = nn.BCELoss()

dataset = CTRDataset(train, features, target='click')
loader = DataLoader(dataset, batch_size=8192, shuffle=True, pin_memory=True, num_workers=2)

for epoch in range(5):  
    model.train()
    losses = []
    for X_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch {epoch+1}, Loss: {np.mean(losses):.5f}')

model.eval()
with torch.no_grad():
    val_dataset = CTRDataset(train, features, target='click')
    val_loader = DataLoader(val_dataset, batch_size=8192, pin_memory=True, num_workers=2)
    y_true = []
    y_pred = []
    for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        y_pred.append(preds)
        y_true.append(y_batch.numpy())
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    print('Train ROC-AUC:', roc_auc_score(y_true, y_pred))

test_dataset = CTRDataset(test, features)
test_loader = DataLoader(test_dataset, batch_size=8192, pin_memory=True, num_workers=2)
test_preds = []
model.eval()
with torch.no_grad():
    for X_batch in tqdm(test_loader, desc="Test prediction"):
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        test_preds.append(preds)
test_preds = np.concatenate(test_preds)
sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = test_preds
sub.to_csv("submission_nn.csv", index=False)
print("submission_nn.csv сохранён.")

model.eval()
with torch.no_grad():
    for X_batch in tqdm(test_loader, desc="Test prediction"):
        X_batch = X_batch.to(device)
        preds = model(X_batch).cpu().numpy()
        test_preds.append(preds)
test_preds = np.concatenate(test_preds)
sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = test_preds
sub.to_csv("submission_nn.csv", index=False)
print("submission_nn.csv сохранён.")
