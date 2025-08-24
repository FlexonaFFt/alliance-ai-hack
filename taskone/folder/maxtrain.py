import pandas as pd
import numpy as np
import os
import gc
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tqdm import tqdm

CFG_LIST = {
    "mid1": dict(iterations=1000, depth=6, lr=0.08, n_sample=5_000_000, 
                 n_splits=3, subsample=0.8, colsample_bylevel=0.8, l2_leaf_reg=3),
    "mid2":   dict(iterations=1500, depth=8, lr=0.05, n_sample=10_000_000,
                 n_splits=5, subsample=0.85, colsample_bylevel=0.85, l2_leaf_reg=2),
    "pro":   dict(iterations=2000, depth=10, lr=0.035, n_sample=10_000_000,
                 n_splits=5, subsample=0.9, colsample_bylevel=0.9,
                 l2_leaf_reg=1, random_strength=1)
}

COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    devices='0',  # Можно изменить на '0:1' для нескольких GPU
    verbose=100,
    early_stopping_rounds=50,
    od_type="Iter",
    border_count=254,
    loss_function='Logloss',
    bootstrap_type='Bernoulli'
)

BLEND_W = dict(light=0.2, mid=0.3, pro=0.5)

def read_sample(path, n_rows):
    return pd.read_csv(path, nrows=n_rows)

def preprocess_data(train, test, features):
    train_processed = train.copy()
    test_processed = test.copy()
    
    for df in [train_processed, test_processed]:
        for col in features:
            df[col] = df[col].fillna('MISSING').astype(str)
            if df[col].nunique() > 100:
                counts = df[col].value_counts()
                rare_categories = counts[counts < 10].index
                df.loc[df[col].isin(rare_categories), col] = 'RARE'
    
    return train_processed, test_processed

def create_features(df, base_features):
    df = df.copy()
    new_features = []
    interaction_pairs = [
        ('site_id', 'app_id'),
        ('site_id', 'device_id'),
        ('app_id', 'device_id'),
        ('site_category', 'app_category')
    ]
    
    for col1, col2 in interaction_pairs:
        if col1 in base_features and col2 in base_features:
            new_feature = f'{col1}_{col2}'
            df[new_feature] = df[col1] + '_' + df[col2]
            new_features.append(new_feature)
    
    for col in base_features:
        if df[col].nunique() < 1000: 
            freq = df[col].value_counts()
            freq_feature = f'{col}_freq'
            df[freq_feature] = df[col].map(freq)
            new_features.append(freq_feature)
    
    return df, new_features

def target_encoding_single_fold(X_tr, X_val, feature, y_tr, alpha=10):
    means = X_tr.copy()
    means['target'] = y_tr
    target_means = means.groupby(feature)['target'].mean()
    counts = means.groupby(feature)['target'].count()
    global_mean = y_tr.mean()
    smoothed_means = (target_means * counts + global_mean * alpha) / (counts + alpha)
    val_encoded = X_val[feature].map(smoothed_means)
    val_encoded.fillna(global_mean, inplace=True)
    
    return val_encoded.values

def target_encoding_test(X_train, X_test, feature, y_train, alpha=10):
    means = X_train.copy()
    means['target'] = y_train
    target_means = means.groupby(feature)['target'].mean()
    counts = means.groupby(feature)['target'].count()
    
    global_mean = y_train.mean()
    smoothed_means = (target_means * counts + global_mean * alpha) / (counts + alpha)
    
    test_encoded = X_test[feature].map(smoothed_means)
    test_encoded.fillna(global_mean, inplace=True)
    
    return test_encoded.values

def train_fold(X_tr, X_val, y_tr, y_val, test_data, features, params, te_features=None):
    if te_features is None:
        te_features = []
    
    X_tr_fold = X_tr.copy()
    X_val_fold = X_val.copy()
    test_fold = test_data.copy()
    
    for feature in te_features:
        if feature in features:
            X_val_fold[f'{feature}_te'] = target_encoding_single_fold(
                X_tr_fold, X_val_fold, feature, y_tr
            )
            
            test_fold[f'{feature}_te'] = target_encoding_test(
                X_tr_fold, test_fold, feature, y_tr
            )
            
            X_tr_fold[f'{feature}_te'] = target_encoding_test(
                X_tr_fold, X_tr_fold, feature, y_tr
            )
    
    all_cat_features = list(X_tr_fold.columns)
    train_pool = Pool(X_tr_fold, y_tr, cat_features=all_cat_features)
    val_pool = Pool(X_val_fold, y_val, cat_features=all_cat_features)
    test_pool = Pool(test_fold, cat_features=all_cat_features)
    model = CatBoostClassifier(**params)
    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        verbose_eval=100
    )
    
    val_pred = model.predict_proba(val_pool)[:, 1]
    test_pred = model.predict_proba(test_pool)[:, 1]
    return model, val_pred, test_pred

def feature_selection(X, y, features, k=50):
    if len(X) > 100000:
        X_encoded = X[features].apply(lambda x: x.astype('category').cat.codes)
        selector = SelectKBest(mutual_info_classif, k=min(k, len(features)))
        selector.fit(X_encoded, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [features[i] for i in selected_indices]
        print(f"Selected {len(selected_features)} features from {len(features)}")
        return selected_features
    else:
        return features

def post_process(predictions):
    return np.clip(predictions, 1e-6, 1 - 1e-6)

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)
    print("Loading test data...")
    test = pd.read_csv("ctr_test.csv")
    base_features = [c for c in test.columns if c not in ['id']]
    test_processed, _ = preprocess_data(test, test, base_features)
    
    blend_pred = np.zeros(len(test))
    blend_oof = None
    all_models_info = []
    for name, cfg in CFG_LIST.items():
        print(f"\n{'='*50}")
        print(f"Training '{name}' configuration")
        print(f"{'='*50}")
        print(f"Loading {cfg['n_sample']} samples...")
        train = read_sample("ctr_train.csv", cfg['n_sample'])
        y = train['click'].values
        

        print("Preprocessing data...")
        train_processed, test_processed_full = preprocess_data(
            train, test_processed, base_features
        )
        

        print("Selecting features...")
        selected_features = feature_selection(
            train_processed, y, base_features, k=50
        )
        

        print("Creating features...")
        train_extended, new_features = create_features(
            train_processed[selected_features], selected_features
        )
        test_extended, _ = create_features(
            test_processed_full[selected_features], selected_features
        )
        
        all_features = list(train_extended.columns)
        te_features = ['site_id', 'app_id', 'device_id', 'site_domain']
        te_features = [f for f in te_features if f in selected_features]
        params = {
            'iterations': cfg['iterations'],
            'depth': cfg['depth'],
            'learning_rate': cfg['lr'],
            'subsample': cfg.get('subsample', 0.8),
            'colsample_bylevel': cfg.get('colsample_bylevel', 0.8),
            'l2_leaf_reg': cfg.get('l2_leaf_reg', 3),
            'random_strength': cfg.get('random_strength', 0),
            **COMMON
        }
        
        print(f"Parameters: {params}")
        print(f"Training with {len(all_features)} features")
        
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_extended, y)):
            print(f"\n--- Fold {fold+1}/{cfg['n_splits']} ---")
            X_tr = train_extended.iloc[tr_idx]
            X_val = train_extended.iloc[val_idx]
            y_tr = y[tr_idx]
            y_val = y[val_idx]
            model, val_pred, test_pred = train_fold(
                X_tr, X_val, y_tr, y_val, test_extended, 
                all_features, params, te_features
            )
            
            oof[val_idx] = val_pred
            pred += test_pred / cfg['n_splits']
            model_path = f"models/{name}_fold{fold}.cbm"
            model.save_model(model_path)
            all_models_info.append({
                'name': name,
                'fold': fold,
                'path': model_path,
                'features': all_features
            })
            
            del model, X_tr, X_val
            gc.collect()
        
        cv_auc = roc_auc_score(y, oof)
        print(f"\n{name} CV AUC: {cv_auc:.6f}")
        np.save(f"predictions/{name}_oof.npy", oof)
        np.save(f"predictions/{name}_pred.npy", pred)
        sub = pd.read_csv("ctr_sample_submission.csv")
        sub['click'] = post_process(pred)
        sub.to_csv(f"submission_{name}.csv", index=False)
        print(f"Saved submission_{name}.csv")
        
        if blend_oof is None:
            blend_oof = np.zeros_like(oof)
        
        weight = BLEND_W[name]
        blend_oof += weight * oof
        blend_pred += weight * pred
        
        del train, train_processed, train_extended, oof
        gc.collect()
    
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    final_train = read_sample("ctr_train.csv", CFG_LIST['mid']['n_sample'])
    final_y = final_train['click'].values[:len(blend_oof)]
    
    final_auc = roc_auc_score(final_y, blend_oof)
    print(f"Blended CV AUC: {final_auc:.6f}")
    
    sub = pd.read_csv("ctr_sample_submission.csv")
    sub['click'] = post_process(blend_pred)
    sub.to_csv("submission_blend_final.csv", index=False)
    print("Saved submission_blend_final.csv")
    
    joblib.dump(all_models_info, "models/models_info.pkl")
    print("Saved models info")
    return final_auc

if __name__ == "__main__":
    auc_score = main()
    print(f"\nFinal AUC Score: {auc_score:.6f}")