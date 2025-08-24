import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

N_SAMPLE = 5_000_000  
N_SPLITS = 5
def read_sample(path, n_rows):
    return pd.read_csv(path, nrows=n_rows)

train = read_sample("ctr_train.csv", N_SAMPLE)
test = pd.read_csv("ctr_test.csv")

features = [col for col in train.columns if col not in ['click', 'id']]
cat_features = features  

y_train = train['click'].values
for col in features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof = np.zeros(len(train))
preds = np.zeros(len(test))
models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y_train)):
    X_tr, X_val = train.iloc[tr_idx][features].copy(), train.iloc[val_idx][features].copy()
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    for col in features:
        X_tr[col] = X_tr[col].astype(str)
        X_val[col] = X_val[col].astype(str)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    test_pool = Pool(test[features], cat_features=cat_features)

    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        eval_metric='AUC',
        random_seed=42,
        task_type='GPU',
        verbose=100,
        early_stopping_rounds=30
    )

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    preds += model.predict_proba(test_pool)[:, 1] / N_SPLITS

    model.save_model(f'catboost_fold{fold}.cbm')
    models.append(model)

print("CV ROC-AUC:", roc_auc_score(y_train, oof))
sample_submission = pd.read_csv('ctr_sample_submission.csv')
sample_submission['click'] = preds
sample_submission.to_csv('submission.csv', index=False)