import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import gc

N_SAMPLE = 3_000_000 
N_SPLITS = 5
RANDOM_STATE = 42

train = pd.read_csv("ctr_train.csv", nrows=N_SAMPLE)
test = pd.read_csv("ctr_test.csv")
features = [c for c in test.columns if c not in ['id']]
y = train['click'].values

for col in features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

oof = np.zeros(len(train))
pred = np.zeros(len(test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 128,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y)):
    print(f"Fold {fold+1}/{N_SPLITS}")
    X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
    y_tr, y_val = y[tr_idx], y[val_idx]
    
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_train, lgb_val],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    pred += model.predict(test[features], num_iteration=model.best_iteration) / N_SPLITS
    
    del model, X_tr, X_val, y_tr, y_val, lgb_train, lgb_val
    gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"LightGBM CV AUC: {cv_auc:.6f}")

sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = pred
sub.to_csv("submission_lgb.csv", index=False)
print("submission_lgb.csv сохранён.")