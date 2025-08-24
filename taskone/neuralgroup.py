import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os
import random

train = pd.read_csv('ctr_train.csv', nrows=10_000_000)
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
    def __init__(self, cat_maps, emb_dim=16, hidden1=256, hidden2=128, dropout1=0.4, dropout2=0.3):
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

MODEL_CONFIGS = [
    dict(emb_dim=16, hidden1=256, hidden2=128, dropout1=0.4, dropout2=0.3, seed=42),
    dict(emb_dim=32, hidden1=256, hidden2=128, dropout1=0.5, dropout2=0.4, seed=123),
    dict(emb_dim=24, hidden1=384, hidden2=192, dropout1=0.3, dropout2=0.2, seed=2025),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs("models_nn", exist_ok=True)

test_dataset = CTRDataset(test, features)
test_loader = DataLoader(test_dataset, batch_size=4096)
test_preds_all = []

for i, cfg in enumerate(MODEL_CONFIGS):
    print(f"\n=== Модель {i+1} | seed={cfg['seed']} ===")
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    random.seed(cfg['seed'])

    model = CTRNet(cat_maps, **{k: cfg[k] for k in cfg if k != 'seed'}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    dataset = CTRDataset(train, features, target='click')
    loader = DataLoader(dataset, batch_size=4096, shuffle=True)

    for epoch in range(8):
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

    torch.save(model.state_dict(), f"models_nn/model_{i+1}.pt")
    model.eval()
    with torch.no_grad():
        X = torch.tensor(train[features].values, dtype=torch.long).to(device)
        y_true = train['click'].values
        y_pred = model(X).cpu().numpy()
        auc = roc_auc_score(y_true, y_pred)
        print(f"Model {i+1} Train ROC-AUC: {auc:.5f}")

    test_preds = []
    model.eval()
    with torch.no_grad():
        for X_batch in tqdm(test_loader, desc="Test prediction"):
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            test_preds.append(preds)
    test_preds = np.concatenate(test_preds)
    test_preds_all.append(test_preds)
    sub = pd.read_csv("ctr_sample_submission.csv")
    sub['click'] = test_preds
    sub.to_csv(f"submission_nn_model{i+1}.csv", index=False)
    print(f"submission_nn_model{i+1}.csv сохранён.")

ensemble_preds = np.mean(test_preds_all, axis=0)
sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = ensemble_preds
sub.to_csv("submission_nn_ensemble.csv", index=False)
print("submission_nn_ensemble.csv сохранён.")
