import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random

train = pd.read_csv('ctr_train.csv')  
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
    def __init__(self, cat_maps, emb_dim=16, hidden1=256, hidden2=128, dropout1=0.3, dropout2=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(len(cat_maps[col]), emb_dim) for col in features
        ])
        self.bn = nn.BatchNorm1d(len(features) * emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(len(features) * emb_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()
        )

    def forward(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embs, dim=1)
        x = self.bn(x)
        return self.fc(x).squeeze()

MODEL_CONFIGS = {
    "fast": dict(emb_dim=8, hidden1=128, hidden2=64, dropout1=0.2, dropout2=0.1, epochs=2, batch_size=16384, lr=3e-3, seed=42),
    "mid":  dict(emb_dim=16, hidden1=256, hidden2=128, dropout1=0.3, dropout2=0.2, epochs=3, batch_size=8192, lr=2e-3, seed=123),
    "pro":  dict(emb_dim=32, hidden1=512, hidden2=256, dropout1=0.4, dropout2=0.3, epochs=4, batch_size=4096, lr=1e-3, seed=2025),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs("models_nn", exist_ok=True)
test_dataset = CTRDataset(test, features)
test_loader = DataLoader(test_dataset, batch_size=16384, pin_memory=True, num_workers=4)
test_preds_all = []

def print_model_info(model, name, cfg):
    total_params = sum(p.numel() for p in model.parameters())
    emb_params = sum(p.numel() for n, p in model.named_parameters() if 'emb_layers' in n)
    print(f"\nИнформация о модели '{name}':")
    print(f"  emb_dim: {cfg['emb_dim']}")
    print(f"  hidden1: {cfg['hidden1']}, hidden2: {cfg['hidden2']}")
    print(f"  dropout1: {cfg['dropout1']}, dropout2: {cfg['dropout2']}")
    print(f"  batch_size: {cfg['batch_size']}, epochs: {cfg['epochs']}, lr: {cfg['lr']}")
    print(f"  Всего параметров: {total_params:,}")
    print(f"  Параметров в embedding-слоях: {emb_params:,}")
    print(f"  Параметров в полносвязных слоях: {total_params - emb_params:,}\n")

for name, cfg in MODEL_CONFIGS.items():
    print(f"\n=== Модель '{name}' | seed={cfg['seed']} ===")
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    model = CTRNet(cat_maps,
                   emb_dim=cfg['emb_dim'],
                   hidden1=cfg['hidden1'],
                   hidden2=cfg['hidden2'],
                   dropout1=cfg['dropout1'],
                   dropout2=cfg['dropout2']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.BCELoss()
    dataset = CTRDataset(train, features, target='click')
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    print_model_info(model, name, cfg)

    for epoch in range(cfg['epochs']):
        model.train()
        losses = []
        for X_batch, y_batch in tqdm(loader, desc=f"{name} Epoch {epoch+1}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"{name} Epoch {epoch+1}, Loss: {np.mean(losses):.5f}")

    torch.save(model.state_dict(), f"models_nn/model_{name}.pt")
    model.eval()
    with torch.no_grad():
        val_dataset = CTRDataset(train, features, target='click')
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4)
        y_true = []
        y_pred = []
        for X_batch, y_batch in tqdm(val_loader, desc=f"{name} Validation"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y_batch.numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        auc = roc_auc_score(y_true, y_pred)
        print(f"{name} Train ROC-AUC: {auc:.5f}")

    test_preds = []
    model.eval()
    with torch.no_grad():
        for X_batch in tqdm(test_loader, desc=f"{name} Test prediction"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            test_preds.append(preds)
    test_preds = np.concatenate(test_preds)
    test_preds_all.append(test_preds)
    sub = pd.read_csv("ctr_sample_submission.csv")
    sub['click'] = test_preds
    sub.to_csv(f"submission_nn_{name}.csv", index=False)
    print(f"submission_nn_{name}.csv сохранён.")

ensemble_preds = np.mean(test_preds_all, axis=0)
sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = ensemble_preds
sub.to_csv("submission_nn_ensemble.csv", index=False)
print("submission_nn_ensemble.csv сохранён.")
