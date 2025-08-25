import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import optuna

COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    verbose=100,
    early_stopping_rounds=100,
    auto_class_weights='Balanced'
)

def feature_engineering(df, y=None, is_train=True, top_pairs=None, features=None):
    if features is None:
        features = [c for c in df.columns if c.startswith('ID_')]
    
    # Frequency encoding
    for col in features:
        freq = df[col].value_counts(normalize=True)
        df[col + '_freq'] = df[col].map(freq).fillna(0).astype(float)
    
    # Target encoding
    if is_train and y is not None:
        te = TargetEncoder(cols=features, smoothing=1.0)
        df[[col + '_te' for col in features]] = te.fit_transform(df[features], y)
    
    # Interactions
    if top_pairs is None:
        top_pairs = []
    for pair in top_pairs:
        new_col = f"{pair[0]}_{pair[1]}"
        df[new_col] = df[pair[0]].astype(str) + '_' + df[pair[1]].astype(str)
    
    new_features = [c for c in df.columns if c not in ['id', 'click'] and c not in features] + features + [f"{p[0]}_{p[1]}" for p in top_pairs]
    return df, new_features

def objective(trial, X, y, features, cat_features):
    params = {
        'iterations': trial.suggest_int('iterations', 500, 3000),
        'depth': trial.suggest_int('depth', 8, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        **COMMON
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx][features], X.iloc[val_idx][features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        for col in cat_features:
            X_tr[col] = X_tr[col].astype(str).fillna('NaN')
            X_val[col] = X_val[col].astype(str).fillna('NaN')
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        preds = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, preds))
    return np.mean(aucs)

if __name__ == "__main__":
    subsample_size = 3_000_000
    train = pd.read_csv("ctr_train.csv", nrows=subsample_size)
    y = train['click'].values
    orig_features = [c for c in train.columns if c.startswith('ID_')]
    
    for col in orig_features:
        train[col] = train[col].astype(str).fillna('NaN')
    
    top_pairs = []
    train, features = feature_engineering(train, y, is_train=True, top_pairs=top_pairs, features=orig_features)
    print("Feature types before Pool:")
    for col in features:
        print(f"{col}: {train[col].dtype}")
    
    cat_features = [c for c in features if not c.endswith('_freq') and not c.endswith('_te')]
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train, y, features, cat_features), n_trials=20)
    
    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)
    
    with open("best_params.txt", "w") as f:
        f.write(str(study.best_params))