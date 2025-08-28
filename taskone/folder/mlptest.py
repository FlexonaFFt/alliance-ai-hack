import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

N_SAMPLE = 2_000_000  
N_SPLITS = 5
BATCH_SIZE = 4096
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_STATE = 42

train = pd.read_csv("ctr_train.csv", nrows=N_SAMPLE)
test = pd.read_csv("ctr_test.csv")
features = [c for c in test.columns if c not in ['id']]
y = train['click'].values

label_encoders = {}
for col in features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

class CTRDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.values.astype(np.int64)
        self.y = y if y is None else y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
        else:
            return torch.tensor(self.X[idx])

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

oof = np.zeros(len(train))
pred = np.zeros(len(test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    train_ds = CTRDataset(X_tr, y_tr)
    val_ds = CTRDataset(X_val, y_val)
    test_ds = CTRDataset(test[features])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = MLP(len(features)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    best_auc = 0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE).float(), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE).float()
                preds = model(xb).cpu().numpy()
                val_preds.append(preds)
        val_preds = np.concatenate(val_preds)
        auc = roc_auc_score(y_val, val_preds)
        print(f"Epoch {epoch+1} AUC: {auc:.5f}")
        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE).float()
            preds = model(xb).cpu().numpy()
            val_preds.append(preds)
    oof[val_idx] = np.concatenate(val_preds)
    
    test_preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(DEVICE).float()
            preds = model(xb).cpu().numpy()
            test_preds.append(preds)
    pred += np.concatenate(test_preds) / N_SPLITS
    
    del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"MLP CV AUC: {cv_auc:.6f}")

sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = pred
sub.to_csv("submission_mlp.csv", index=False)
print("submission_mlp.csv сохранён.")