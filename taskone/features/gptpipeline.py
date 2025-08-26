import pandas as pd, numpy as np, os, gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CFG = dict(iterations=1500, depth=8, lr=0.03848888585447572, n_sample=5_000_000, n_splits=5)
COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    verbose=100,
    early_stopping_rounds=40,
    l2_leaf_reg=9.096315897868687,
    boosting_type="Plain",
    bootstrap_type="Bernoulli",
)

def add_frequency_encoding(df, cat_cols):
    for col in cat_cols:
        freq = df[col].value_counts()
        df[f"{col}_freq"] = df[col].map(freq)
    return df

# target encoding (CV)
def add_target_encoding(train, test, cat_cols, target, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for col in cat_cols:
        train[f"{col}_te"] = 0.0
        test[f"{col}_te"] = 0.0
        for tr_idx, val_idx in skf.split(train, train[target]):
            tr, val = train.iloc[tr_idx], train.iloc[val_idx]
            means = tr.groupby(col)[target].mean()
            train.loc[val_idx, f"{col}_te"] = val[col].map(means)
        global_means = train.groupby(col)[target].mean()
        test[f"{col}_te"] = test[col].map(global_means)
    return train, test

# генерация парных признаков
def add_pairwise_interactions(train, test, cat_cols, max_pairs=5):
    from itertools import combinations
    pairs = list(combinations(cat_cols, 2))[:max_pairs]
    for col1, col2 in pairs:
        new_col = f"{col1}_{col2}_int"
        train[new_col] = train[col1] + "_" + train[col2]
        test[new_col]  = test[col1]  + "_" + test[col2]
    return train, test

print("\n=== Читаем данные ===")
train = pd.read_csv("ctr_train.csv", nrows=CFG['n_sample'])
test  = pd.read_csv("ctr_test.csv")
sub   = pd.read_csv("ctr_sample_submission.csv")

features = [c for c in test.columns if c not in ['id']]

for col in features:
    train[col] = train[col].astype(str)
    test[col]  = test[col].astype(str)

print("Добавляем frequency encoding...")
train = add_frequency_encoding(train, features)
test  = add_frequency_encoding(test, features)

print("Добавляем target encoding (CV)...")
train, test = add_target_encoding(train, test, features, "click", n_splits=CFG['n_splits'])

print("Генерируем парные признаки...")
train, test = add_pairwise_interactions(train, test, features, max_pairs=10)

new_features = [c for c in train.columns if c not in ['id','click']]

X, y = train[new_features], train['click'].values
X_test = test[new_features]

print("=== Обучение CatBoost ===")
interaction_cols = [c for c in train.columns if c.endswith('_int')]
cat_cols = features + interaction_cols

new_features = [c for c in train.columns if c not in ['id','click']]
X, y = train[new_features], train['click'].values
X_test = test[new_features]

cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
num_like = [c for c in X.columns if c.endswith('_freq') or c.endswith('_te')]
X = X.copy()
X_test = X_test.copy()
for c in num_like:
    X[c] = X[c].astype('float32')
    X_test[c] = X_test[c].astype('float32')

oof, pred = np.zeros(len(train)), np.zeros(len(test))
skf = StratifiedKFold(n_splits=CFG['n_splits'], shuffle=True, random_state=42)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"FOLD {fold}")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = CatBoostClassifier(
        iterations=CFG['iterations'],
        depth=CFG['depth'],
        learning_rate=CFG['lr'],
        **COMMON
    )

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool   = Pool(X_val, y_val, cat_features=cat_idx)
    test_pool  = Pool(X_test, cat_features=cat_idx)

    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    oof[val_idx] = model.predict_proba(val_pool)[:, 1]
    pred += model.predict_proba(test_pool)[:, 1] / CFG['n_splits']

    os.makedirs("models", exist_ok=True)
    model.save_model(f"models/pro_adv_fold{fold}.cbm")
    del model; gc.collect()

cv_auc = roc_auc_score(y, oof)
print(f"CV AUC (freq + target encoding + pairwise): {cv_auc:.6f}")

sub['click'] = pred
sub.to_csv("submission_pro_adv.csv", index=False)
print("submission_pro_adv.csv сохранён.")
