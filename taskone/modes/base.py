import pandas as pd
import numpy as np
import os
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

class CTRTrainer:
    def __init__(self, cfg_list, common_params, blend_weights, train_path, test_path, sample_sub_path):
        self.cfg_list = cfg_list
        self.common_params = common_params
        self.blend_weights = blend_weights
        self.train_path = train_path
        self.test_path = test_path
        self.sample_sub_path = sample_sub_path
        self.features = None
        self.test_pool_template = None
        self.blend_pred = None
        self.blend_oof = None
        self.max_sample = max(cfg['n_sample'] for cfg in cfg_list.values())  # Унифицируем размер сэмпла

    def read_sample(self, path, n_rows):
        return pd.read_csv(path, nrows=n_rows)

    def prepare_test_pool(self):
        test = pd.read_csv(self.test_path)
        self.features = [c for c in test.columns if c not in ['id']]
        for col in self.features:
            test[col] = test[col].astype(str)
        self.test_pool_template = Pool(test[self.features], cat_features=self.features)
        return self.test_pool_template.num_row()  

    def train_family(self, name, cfg):
        print(f"\n=== Training family '{name}' ===")
        train = self.read_sample(self.train_path, self.max_sample)  
        y = train['click'].values
        for col in self.features:
            train[col] = train[col].astype(str)
        
        oof = np.zeros(len(train))
        pred = np.zeros(self.test_pool_template.num_row())  
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
        params = {
            'iterations': cfg['iterations'],
            'depth': cfg['depth'],
            'learning_rate': cfg['lr'],
            **self.common_params
        }
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train[self.features], y)):
            X_tr, X_val = train.iloc[tr_idx][self.features], train.iloc[val_idx][self.features]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr, cat_features=self.features),
                eval_set=Pool(X_val, y_val, cat_features=self.features),
                use_best_model=True
            )
            
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            pred += model.predict_proba(self.test_pool_template)[:, 1] / cfg['n_splits']
            
            os.makedirs("models", exist_ok=True)
            model.save_model(f"models/{name}_fold{fold}.cbm")
            del model
            gc.collect()
        
        cv_auc = roc_auc_score(y, oof)
        print(f"{name}  CV AUC: {cv_auc:.6f}")
        
        np.save(f"models/{name}_oof.npy", oof)
        np.save(f"models/{name}_pred.npy", pred)

        sub_ind = pd.read_csv(self.sample_sub_path)
        sub_ind['click'] = pred
        sub_ind.to_csv(f"submission_{name}.csv", index=False)
        print(f"submission_{name}.csv сохранён.")
        
        return oof, pred

    def blend_all(self):
        test_len = self.prepare_test_pool()
        self.blend_pred = np.zeros(test_len)
        self.blend_oof = None
        
        for name, cfg in self.cfg_list.items():
            oof, pred = self.train_family(name, cfg)
            w = self.blend_weights[name]
            if self.blend_oof is None:
                self.blend_oof = np.zeros_like(oof)
            self.blend_oof += w * oof
            self.blend_pred += w * pred
            del oof, pred
            gc.collect()
        
        mid_sample = self.read_sample(self.train_path, self.max_sample)  
        overall_auc = roc_auc_score(mid_sample['click'].values, self.blend_oof)
        print(f"\nBlended CV AUC: {overall_auc:.6f}")
        
        sub = pd.read_csv(self.sample_sub_path)
        sub['click'] = self.blend_pred
        sub.to_csv("submission_blend.csv", index=False)
        print("submission_blend.csv сохранён.")

CFG_LIST = {
    "light": dict(iterations=500, depth=6,  lr=0.08,  n_sample=20_000_000,  n_splits=3), 
    "mid":   dict(iterations=800, depth=8,  lr=0.04,  n_sample=20_000_000, n_splits=5),
    "pro":   dict(iterations=2000, depth=10, lr=0.01,  n_sample=20_000_000, n_splits=7)
}
COMMON = dict(
    eval_metric="AUC",
    random_seed=42,
    task_type="GPU",     
    verbose=100,
    early_stopping_rounds=150,
    grow_policy="SymmetricTree",
    boosting_type="Ordered",  # Новый: ordered boosting для лучшей производительности
    l2_leaf_reg=5,  # Новый: усиленная регуляризация
    bagging_temperature=0.5,  # Новый: для разнообразия
    scale_pos_weight=10  # Новый: пример для дисбаланса; рассчитайте как len(negative)/len(positive)
)
BLEND_W = dict(light=0.2, mid=0.3, pro=0.5)
trainer = CTRTrainer(CFG_LIST, COMMON, BLEND_W, "ctr_train.csv", "ctr_test.csv", "ctr_sample_submission.csv")
trainer.blend_all()