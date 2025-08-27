import pandas as pd
import numpy as np
import os
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


class CTRConfig:
    def __init__(self):
        self.train_path = "ctr_train.csv"
        self.test_path = "ctr_test.csv"
        self.submission_path = "ctr_sample_submission.csv"
        self.target = "click"
        
        self.CFG_LIST = {
            "light": dict(iterations=500, depth=6,  lr=0.04,  n_sample=2_000_000,  n_splits=3),
            "mid":   dict(iterations=500, depth=8,  lr=0.03,  n_sample=6_000_000, n_splits=5),
            "pro":   dict(iterations=2733, depth=8, lr=0.033, n_sample=6_000_000, n_splits=5)
        }
        
        self.COMMON = dict(
            eval_metric="AUC",
            random_seed=42,
            task_type="GPU",     
            verbose=100,
            early_stopping_rounds=40
        )
        
        self.BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)
        self.features = [
            'hour', 'banner_pos', 'site_category', 'app_category',
            'device_model', 'device_type', 'device_conn_type',
            'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
        ]
        
        self.small_cat_features = ['hour', 'banner_pos', 'device_type', 'device_conn_type']
        self.large_cat_features = [
            'site_category', 'app_category', 'device_model',
            'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
        ]


class FeatureEngineer:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.freq_maps = {}
        self.target_maps = {}
        self.label_encoders = {}
        self.le_dict = {}  # Новый словарь для сопоставлений
    
    def fit(self, df: pd.DataFrame):
        print("Обучение маппингов для кодирования признаков...")
        y = df[self.cfg.target].values if self.cfg.target in df.columns else None
        for col in self.cfg.features:
            if col in df.columns:
                self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        
        if y is not None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for col in self.cfg.features:
                df[f"{col}_te"] = 0.0
                for tr_idx, val_idx in skf.split(df, y):
                    tr, val = df.iloc[tr_idx], df.iloc[val_idx]
                    means = tr.groupby(col)[self.cfg.target].mean()
                    df.loc[val_idx, f"{col}_te"] = val[col].map(means).fillna(0.0)
                self.target_maps[col] = df.groupby(col)[self.cfg.target].mean().to_dict()
        
        for col in self.cfg.features:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna("__NA__"))
            self.label_encoders[col] = le
            self.le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))  # Создаем словарь
        
        print("Маппинги для кодирования признаков обучены.")
    
    def transform(self, df: pd.DataFrame, is_train=True):
        print(f"Преобразование {'обучающих' if is_train else 'тестовых'} данных...")
        for col in self.cfg.features:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("__NA__")
        
        for col in self.cfg.features:
            if col in df.columns and col in self.freq_maps:
                df[f"{col}_freq"] = df[col].map(self.freq_maps[col]).fillna(0)
        
        if not is_train:  
            for col in self.cfg.features:
                if col in df.columns and col in self.target_maps:
                    df[f"{col}_te"] = df[col].map(self.target_maps[col]).fillna(0)
        
        for col in self.cfg.features:
            if col in df.columns and col in self.le_dict:
                df[f"{col}_le"] = df[col].map(self.le_dict[col]).fillna(-1)  # Обработка неизвестных значений
        
        print(f"Данные преобразованы. Новая размерность: {df.shape}")
        return df


class CTRTrainer:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.fe = FeatureEngineer(cfg)
        self.test = None
        self.test_processed = None
        self.blend_pred = None
        self.blend_oof = None
    
    def read_sample(self, path, n_rows):
        return pd.read_csv(path, nrows=n_rows)
    
    def load_test(self):
        print("Загрузка тестовых данных...")
        self.test = pd.read_csv(self.cfg.test_path)
        features_in_test = [col for col in self.cfg.features if col in self.test.columns]
        self.test = self.test[['id'] + features_in_test]
        print(f"Тестовые данные загружены, shape: {self.test.shape}")
    
    def train_family(self, name, cfg):
        print(f"\n=== Обучение семейства '{name}' ===")       
        train = self.read_sample(self.cfg.train_path, cfg['n_sample'])
        features_in_train = [col for col in self.cfg.features if col in train.columns]
        train = train[['id', self.cfg.target] + features_in_train]   
        y = train[self.cfg.target].values
        
        self.fe.fit(train)
        train_processed = self.fe.transform(train)
        if self.test_processed is None:
            self.test_processed = self.fe.transform(self.test.copy(), is_train=False)
        
        features = [col for col in train_processed.columns 
                   if col not in ['id', self.cfg.target]]
        cat_features = [col for col in features 
                       if not col.endswith('_freq') and 
                          not col.endswith('_te') and 
                          not col.endswith('_le')]
        
        print(f"Используем {len(features)} признаков, из них {len(cat_features)} категориальных")
        
        oof = np.zeros(len(train_processed))
        pred = np.zeros(len(self.test_processed))
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
        params = dict(
            iterations=cfg['iterations'],
            depth=cfg['depth'],
            learning_rate=cfg['lr'],
            **self.cfg.COMMON
        )
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(train_processed[features], y)):
            print(f"Обучение фолда {fold+1}/{cfg['n_splits']}...")
            
            X_tr = train_processed.iloc[tr_idx][features]
            X_val = train_processed.iloc[val_idx][features]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr, cat_features=cat_features),
                eval_set=Pool(X_val, y_val, cat_features=cat_features),
                use_best_model=True
            )
            
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            pred += model.predict_proba(self.test_processed[features])[:, 1] / cfg['n_splits']
            
            os.makedirs("models", exist_ok=True)
            model.save_model(f"models/{name}_fold{fold}.cbm")
            del model; gc.collect()
        
        cv_auc = roc_auc_score(y, oof)
        print(f"{name} CV AUC: {cv_auc:.6f}")
        
        np.save(f"models/{name}_oof.npy", oof)
        np.save(f"models/{name}_pred.npy", pred)
        
        sub_ind = pd.read_csv(self.cfg.submission_path)
        sub_ind['click'] = pred
        sub_ind.to_csv(f"submission_{name}.csv", index=False)
        print(f"submission_{name}.csv сохранён.")
        
        if self.blend_oof is None:
            self.blend_oof = np.zeros_like(oof)
        if self.blend_pred is None:
            self.blend_pred = np.zeros_like(pred)
            
        w = self.cfg.BLEND_W[name]
        self.blend_oof += w * oof
        self.blend_pred += w * pred
        
        del train, train_processed, oof, pred; gc.collect()
        
        return cv_auc
    
    def save_blend(self):
        mid_sample = self.read_sample(self.cfg.train_path, self.cfg.CFG_LIST['mid']['n_sample'])
        y_true = mid_sample[self.cfg.target].values
        
        overall_auc = roc_auc_score(y_true, self.blend_oof)
        print(f"\nBlended CV AUC: {overall_auc:.6f}")
        
        sub = pd.read_csv(self.cfg.submission_path)
        sub['click'] = self.blend_pred
        sub.to_csv("submission_blend.csv", index=False)
        print("submission_blend.csv сохранён.")
        
        return overall_auc


class Pipeline:
    def __init__(self):
        self.cfg = CTRConfig()
        self.trainer = CTRTrainer(self.cfg)
    
    def run(self):
        print("Запуск пайплайна обучения...")
        self.trainer.load_test()
        for name, cfg in self.cfg.CFG_LIST.items():
            self.trainer.train_family(name, cfg)
        overall_auc = self.trainer.save_blend()
        print(f"Пайплайн завершен. Итоговый AUC: {overall_auc:.6f}")
        return overall_auc


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()