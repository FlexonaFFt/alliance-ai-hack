import pandas as pd, numpy as np, os, gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def target_encode(train, test, col, target, min_samples_leaf=100, smoothing=10):
    temp = pd.concat([train[[col, target]], test[[col]]])
    averages = train.groupby(col)[target].agg(['mean', 'count'])
    smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
    prior = train[target].mean()
    averages['smoothed'] = prior * (1 - smoothing) + averages['mean'] * smoothing
    ft_train = train[col].map(averages['smoothed']).fillna(prior)
    ft_test = test[col].map(averages['smoothed']).fillna(prior)
    return ft_train, ft_test

def freq_encode(train, test, col):
    freq = train[col].value_counts()
    ft_train = train[col].map(freq).fillna(0)
    ft_test = test[col].map(freq).fillna(0)
    return ft_train, ft_test

CFG = dict(
    iterations=1200,
    depth=12,
    learning_rate=0.025,
    l2_leaf_reg=5,
    random_strength=1,
    border_count=254,
    grow_policy='Lossguide',
    subsample=0.8,
    rsm=0.8,
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    verbose=100,
    early_stopping_rounds=60
)

N_SAMPLE = 10_000_000
N_SPLITS = 7

def read_sample(path, n_rows):
    return pd.read_csv(path, nrows=n_rows)

test = pd.read_csv("ctr_test.csv")
features = [c for c in test.columns if c not in ['id']]
for col in features:
    test[col] = test[col].astype(str)

train = read_sample("ctr_train.csv", N_SAMPLE)
y = train['click'].values
for col in features:
    train[col] = train[col].astype(str)
    # Target encoding
    tr_te, te_te = target_encode(train, test, col, 'click')
    train[f"{col}_te"] = tr_te
    test[f"{col}_te"] = te_te
    # Frequency encoding
    tr_fe, te_fe = freq_encode(train, test, col)
    train[f"{col}_fe"] = tr_fe
    test[f"{col}_fe"] = te_fe

# Взаимодействия признаков (пример для пар)
for i in range(len(features)):
    for j in range(i+1, len(features)):
        new_col = f"{features[i]}_{features[j]}"
        train[new_col] = train[features[i]] + "_" + train[features[j]]
        test[new_col] = test[features[i]] + "_" + test[features[j]]

all_features = [c for c in train.columns if c not in ['id', 'click']]
oof, pred = np.zeros(len(train)), np.zeros(len(test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(train[all_features], y)):
    X_tr, X_val = train.iloc[tr_idx][all_features], train.iloc[val_idx][all_features]
    y_tr, y_val = y[tr_idx], y[val_idx]
    model = CatBoostClassifier(**CFG)
    model.fit(
        Pool(X_tr, y_tr, cat_features=features),
        eval_set=Pool(X_val, y_val, cat_features=features),
        use_best_model=True
    )
    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    pred += model.predict_proba(Pool(test[all_features], cat_features=features))[:, 1] / N_SPLITS
    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/super_fold{fold}.cbm")
    del model; gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"SuperModel CV AUC: {cv_auc:.6f}")

np.save("models/super_oof.npy", oof)
np.save("models/super_pred.npy", pred)

sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = pred
sub.to_csv("submission_super.csv", index=False)
print("submission_super.csv сохранён.")