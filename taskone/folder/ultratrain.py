import pandas as pd, numpy as np, os, gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CFG_LIST = {
    "strong20m": dict(iterations=1200, depth=10,  lr=0.025,  n_sample=20_000_000,  n_splits=7)
}
COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",
    verbose=100,
    early_stopping_rounds=80,  
    l2_leaf_reg=5,              
    border_count=254,           
    grow_policy='Lossguide',    
    subsample=0.8,              
    rsm=0.8                     
)
BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)

def read_sample(path, n_rows):
    return pd.read_csv(path, nrows=n_rows)

test = pd.read_csv("ctr_test.csv")
features = [c for c in test.columns if c not in ['id']]
for col in features:
    test[col] = test[col].astype(str)
test_pool_template = Pool(test[features], cat_features=features)

blend_pred, blend_oof = np.zeros(len(test)), None
for name, cfg in CFG_LIST.items():
    print(f"\n=== Training family '{name}' ===")
    train = read_sample("ctr_train.csv", cfg['n_sample'])
    y = train['click'].values
    for col in features:
        train[col] = train[col].astype(str)
    
    oof, pred = np.zeros(len(train)), np.zeros(len(test))
    skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
    params = dict(
        iterations=cfg['iterations'],
        depth=cfg['depth'],
        learning_rate=cfg['lr'],
        **COMMON
    )
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y)):
        X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            Pool(X_tr, y_tr, cat_features=features),
            eval_set=Pool(X_val, y_val, cat_features=features),
            use_best_model=True
        )
        
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        pred += model.predict_proba(test_pool_template)[:, 1] / cfg['n_splits']
        
        os.makedirs("models", exist_ok=True)
        model.save_model(f"models/{name}_fold{fold}.cbm")
        del model; gc.collect()
    
    cv_auc = roc_auc_score(y, oof)
    print(f"{name}  CV AUC: {cv_auc:.6f}")
    
    np.save(f"models/{name}_oof.npy",  oof)
    np.save(f"models/{name}_pred.npy", pred)

    sub_ind = pd.read_csv("ctr_sample_submission.csv")
    sub_ind['click'] = pred
    sub_ind.to_csv(f"submission_{name}.csv", index=False)
    print(f"submission_{name}.csv сохранён.")
    
    if blend_oof is None:
        blend_oof = np.zeros_like(oof)
    w = BLEND_W[name]
    blend_oof  += w * oof
    blend_pred += w * pred
    
    del train, oof, pred; gc.collect()

overall_auc = roc_auc_score(read_sample("ctr_train.csv", CFG_LIST['mid']['n_sample'])['click'].values, blend_oof)
print(f"\nBlended CV AUC: {overall_auc:.6f}")

sub = pd.read_csv("ctr_sample_submission.csv")
sub['click'] = blend_pred
sub.to_csv("submission_blend.csv", index=False)
print("submission_blend.csv сохранён.")