import pandas as pd
import numpy as np
import os
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from eli5.sklearn import PermutationImportance
from sklearn.preprocessing import LabelEncoder
import json

best_params = {'iterations': 2733, 'depth': 8, 'learning_rate': 0.03848888585447572, 'l2_leaf_reg': 9.096315897868687}  # Placeholder

# Define CFG_LIST without duplicate keys
CFG_LIST = {
    "light": {**best_params, 'n_sample': 20_000_000, 'n_splits': 5},
    "mid": {**best_params, 'iterations': best_params['iterations'] + 500, 'depth': best_params['depth'] + 1, 'learning_rate': best_params['learning_rate'] * 0.8, 'n_sample': 20_000_000, 'n_splits': 5},
    "pro": {**best_params, 'iterations': best_params['iterations'] + 1000, 'depth': best_params['depth'] + 2, 'learning_rate': best_params['learning_rate'] * 0.6, 'n_sample': 20_000_000, 'n_splits': 5}
}
COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    verbose=100,
    early_stopping_rounds=100,
    auto_class_weights='Balanced'
)
BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)

def feature_engineering(df, y=None, is_train=True, top_pairs=None, features=None):
    if features is None:
        features = [c for c in df.columns if c.startswith('site_') or c.startswith('app_') or c.startswith('device_')]
    
    for col in features:
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq).fillna(0).astype(float)
    
    if is_train and y is not None:
        te = TargetEncoder(cols=features, smoothing=1.0)
        df[[col + '_te' for col in features]] = te.fit_transform(df[features], y)
    
    if top_pairs is None:
        top_pairs = []
    for pair in top_pairs:
        new_col = f"{pair[0]}_{pair[1]}"
        df[new_col] = df[pair[0]].astype(str) + '_' + df[pair[1]].astype(str)
    
    new_features = [c for c in df.columns if c not in ['id', 'click', 'idx'] and c not in features] + features + [f"{p[0]}_{p[1]}" for p in top_pairs]
    return df, new_features

def select_features(model, X, y, features, threshold=0.01):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cat_features = [c for c in features if not c.endswith('_freq') and not c.endswith('_te')]

    # Убедимся, что категориальные признаки — строки
    for col in cat_features:
        X_train[col] = X_train[col].astype(str).fillna('NaN')
        X_val[col] = X_val[col].astype(str).fillna('NaN')

    # Debug вывод
    print("Verifying X_val type:", type(X_val))
    print("X_val columns:", X_val.columns.tolist())
    print("Feature types in X_val before encoding:")
    for col in features:
        try:
            print(f"{col}: {X_val[col].dtype}")
        except Exception as e:
            print(f"Error accessing {col}: {e}")

    print("Features for Pool:", features)
    print("Categorical features for Pool:", cat_features)

    # Основная модель (с cat_features)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Подготовка данных для numeric-only версии
    X_train_numeric = X_train[features].copy()
    X_val_numeric = X_val[features].copy()
    label_encoders = {}

    for col in cat_features:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col]], axis=0).astype(str).fillna('NaN')
        le.fit(combined)
        X_train_numeric[col] = le.transform(X_train[col])
        X_val_numeric[col] = le.transform(X_val[col])
        label_encoders[col] = le

    # В numpy для PermutationImportance
    X_train_numeric = X_train_numeric.to_numpy()
    X_val_numeric = X_val_numeric.to_numpy()

    # Numeric CatBoost (без cat_features)
    model_numeric = CatBoostClassifier(
        iterations=model.get_param('iterations'),
        depth=model.get_param('depth'),
        learning_rate=model.get_param('learning_rate'),
        l2_leaf_reg=model.get_param('l2_leaf_reg'),
        **COMMON
    )
    model_numeric.fit(X_train_numeric, y_train, eval_set=(X_val_numeric, y_val), use_best_model=True)

    # Permutation Importance
    perm = PermutationImportance(model_numeric, random_state=42, n_iter=5).fit(X_val_numeric, y_val)
    importances = perm.feature_importances_

    # Отбор признаков
    selected = [features[i] for i in range(len(features)) if importances[i] > threshold]
    print(f"Selected {len(selected)} features out of {len(features)}")
    print("Selected features:", selected)
    print("Feature importances:", list(zip(features, importances)))

    return selected


test = pd.read_csv("ctr_test.csv")
orig_features = [c for c in test.columns if c.startswith('site_') or c.startswith('app_') or c.startswith('device_')]
numeric_features = ['hour', 'C1', 'banner_pos', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
for col in orig_features:
    test[col] = test[col].astype(str).fillna('NaN')

blend_pred, blend_oof = np.zeros(len(test)), None
top_pairs = None

for name, cfg in CFG_LIST.items():
    print(f"\n=== Training family '{name}' ===")
    train = pd.read_csv("ctr_train.csv", nrows=cfg['n_sample'])
    y = train['click'].values
    for col in orig_features:
        train[col] = train[col].astype(str).fillna('NaN')
    
    train, features = feature_engineering(train, y, is_train=True, top_pairs=top_pairs, features=orig_features)
    test, _ = feature_engineering(test, is_train=False, top_pairs=top_pairs, features=orig_features)
    
    if name == 'light':
        sub_train = train.sample(1_000_000, random_state=42)
        sub_y = y[sub_train.index]
        sub_model = CatBoostClassifier(**COMMON, iterations=500, depth=8, learning_rate=0.05)
        selected_features = select_features(sub_model, sub_train[features], sub_y, features)
        
        inter_import = sub_model.get_feature_importance(type='Interaction')
        top_pairs = [(features[int(i)], features[int(j)]) for i, j, score in inter_import[:5]]
        print("Top interaction pairs:", top_pairs)
        
        train, features = feature_engineering(train, y, is_train=True, top_pairs=top_pairs, features=orig_features)
        test, _ = feature_engineering(test, is_train=False, top_pairs=top_pairs, features=orig_features)
        features = selected_features + [c for c in train.columns if c.endswith('_freq') or c.endswith('_te') or '_' in c]
    
    cat_features = [c for c in features if not c.endswith('_freq') and not c.endswith('_te') and c not in numeric_features]
    
    oof, pred = np.zeros(len(train)), np.zeros(len(test))
    skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
    params = {k: cfg[k] for k in ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']}
    params.update(COMMON)
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train, y)):
        X_tr = train.iloc[tr_idx][features]
        X_val = train.iloc[val_idx][features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        for col in cat_features:
            X_tr[col] = X_tr[col].astype(str).fillna('NaN')
            X_val[col] = X_val[col].astype(str).fillna('NaN')
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        test_pool = Pool(test[features], cat_features=cat_features)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        pred += model.predict_proba(test[features])[:, 1] / cfg['n_splits']
        
        os.makedirs("models", exist_ok=True)
        model.save_model(f"models/{name}_fold{fold}.cbm")
        del model; gc.collect()
    
    cv_auc = roc_auc_score(y, oof)
    print(f"{name} CV AUC: {cv_auc:.6f}")
    
    np.save(f"models/{name}_oof.npy", oof)
    np.save(f"models/{name}_pred.npy", pred)
    
    sub_ind = pd.read_csv("ctr_sample_submission.csv")
    sub_ind['click'] = pred
    sub_ind.to_csv(f"submission_{name}.csv", index=False)
    print(f"submission_{name}.csv saved.")
    
    if blend_oof is None:
        blend_oof = np.zeros_like(oof)
    w = BLEND_W[name]
    blend_oof += w * oof
    blend_pred += w * pred
    
    del train, oof, pred; gc.collect()

overall_auc = roc_auc_score(pd.read_csv("ctr_train.csv", nrows=CFG_LIST['mid']['n_sample'])['click'].values, blend_oof)
print(f"\nBlended CV AUC: {overall_auc:.6f}")

sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = blend_pred
sub.to_csv("submission_blend.csv", index=False)
print("submission_blend.csv saved.")
