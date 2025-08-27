import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder

# ================= CONFIG =================
class EnhancedConfig:
    def __init__(self):
        self.train_path = "ctr_train.csv"
        self.test_path = "ctr_test.csv"
        self.submission_path = "ctr_sample_submission.csv"
        self.target = "click"
        
        # Оптимальные параметры из tune.py
        self.best_params = {
            'iterations': 2733, 
            'depth': 10,  # Увеличиваем глубину
            'learning_rate': 0.03848888585447572, 
            'l2_leaf_reg': 9.096315897868687
        }
        
        # Конфигурации для разных моделей
        self.CFG_LIST = {
            "light": {**self.best_params, 'n_sample': 5_000_000, 'n_splits': 5},
            "mid": {**self.best_params, 'iterations': self.best_params['iterations'] + 500, 
                   'depth': self.best_params['depth'] + 1, 
                   'learning_rate': self.best_params['learning_rate'] * 0.8, 
                   'n_sample': 10_000_000, 'n_splits': 5},
            "pro": {**self.best_params, 'iterations': self.best_params['iterations'] + 1000, 
                   'depth': self.best_params['depth'] + 2, 
                   'learning_rate': self.best_params['learning_rate'] * 0.6, 
                   'n_sample': 20_000_000, 'n_splits': 5}
        }
        
        # Общие параметры
        self.COMMON = dict(
            eval_metric="AUC",
            random_seed=42,
            task_type="GPU",
            verbose=100,
            early_stopping_rounds=100,
            auto_class_weights='Balanced',
            boosting_type="Plain",
            bootstrap_type="Bernoulli"
        )
        
        # Веса для блендинга
        self.BLEND_W = dict(light=0.2, mid=0.3, pro=0.5)  # Увеличиваем вес pro модели
        
        # Признаки
        self.features = [
            'hour', 'banner_pos', 'site_category', 'app_category',
            'device_model', 'device_type', 'device_conn_type',
            'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
        ]
        
        # Категориальные признаки с малым количеством уникальных значений
        self.small_cat_features = ['hour', 'banner_pos', 'device_type', 'device_conn_type']
        
        # Категориальные признаки с большим количеством уникальных значений
        self.large_cat_features = [
            'site_category', 'app_category', 'device_model',
            'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
        ]

# ================= FEATURE ENGINEER =================
class EnhancedFeatureEngineer:
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
        self.freq_maps = {}
        self.target_encoders = {}
        self.label_encoders = {}

    def fit(self, df: pd.DataFrame):
        """Обучаем маппинги (частотное кодирование, target encoding)"""
        y = df[self.cfg.target].values if self.cfg.target in df.columns else None
        
        # Частотное кодирование
        for col in df.columns:
            if col not in [self.cfg.target, "id"]:
                self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        
        # Target encoding (с кросс-валидацией)
        if y is not None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for col in self.cfg.large_cat_features + self.cfg.small_cat_features:
                df[f"{col}_te"] = 0.0
                for tr_idx, val_idx in skf.split(df, y):
                    tr, val = df.iloc[tr_idx], df.iloc[val_idx]
                    means = tr.groupby(col)[self.cfg.target].mean()
                    df.loc[val_idx, f"{col}_te"] = val[col].map(means).fillna(0.0)
        
        # Label encoding для категориальных признаков
        for col in self.cfg.large_cat_features + self.cfg.small_cat_features:
            le = LabelEncoder()
            le.fit(df[col].astype(str).fillna("__NA__"))
            self.label_encoders[col] = le

    def transform(self, df: pd.DataFrame, is_train=True):
        """Применяем преобразования"""
        # Заполнение пропусков
        for col in df.columns:
            if col not in [self.cfg.target, "id"]:
                df[col] = df[col].astype(str).fillna("__NA__")
        
        # Частотное кодирование
        for col in df.columns:
            if col not in [self.cfg.target, "id"]:
                if col in self.freq_maps:
                    df[f"{col}_freq"] = df[col].map(self.freq_maps[col]).fillna(0)
        
        # Label encoding
        for col in self.cfg.large_cat_features + self.cfg.small_cat_features:
            if col in self.label_encoders:
                df[f"{col}_le"] = self.label_encoders[col].transform(df[col].astype(str).fillna("__NA__"))
        
        # Добавляем взаимодействия признаков
        self._add_interactions(df)
        
        return df
    
    def _add_interactions(self, df):
        """Добавляем взаимодействия между признаками"""
        # Взаимодействия между категориальными признаками
        important_cats = ['site_category', 'app_category', 'device_model', 'C1']
        for i, col1 in enumerate(important_cats):
            for col2 in important_cats[i+1:]:
                df[f"{col1}_{col2}_int"] = df[col1] + "_" + df[col2]
                # Добавляем частотное кодирование для взаимодействий
                if is_train:
                    freq = df[f"{col1}_{col2}_int"].value_counts(normalize=True)
                    df[f"{col1}_{col2}_int_freq"] = df[f"{col1}_{col2}_int"].map(freq).fillna(0)
        
        # Взаимодействия между числовыми признаками (если они есть)
        numeric_cols = [f"{col}_freq" for col in self.cfg.large_cat_features]
        for i, col1 in enumerate(numeric_cols[:3]):  # Ограничиваем количество взаимодействий
            for col2 in numeric_cols[i+1:4]:  # Ограничиваем количество взаимодействий
                if col1 in df.columns and col2 in df.columns:
                    df[f"{col1}_{col2}_prod"] = df[col1] * df[col2]
                    df[f"{col1}_{col2}_sum"] = df[col1] + df[col2]
                    df[f"{col1}_{col2}_diff"] = df[col1] - df[col2]

# ================= TRAINER =================
class EnhancedTrainer:
    def __init__(self, cfg: EnhancedConfig, fe: EnhancedFeatureEngineer):
        self.cfg = cfg
        self.fe = fe

    def read_sample(self, path, n_rows):
        return pd.read_csv(path, nrows=n_rows)

    def train_family(self, name, cfg, test):
        print(f"\n=== Training family '{name}' ===")

        # load train
        train = self.read_sample(self.cfg.train_path, cfg['n_sample'])
        y = train[self.cfg.target].values

        # feature engineering
        self.fe.fit(train)
        train = self.fe.transform(train)
        test = self.fe.transform(test.copy(), is_train=False)

        # features
        features = [c for c in train.columns if c not in ['id', self.cfg.target]]

        # категориальные признаки
        cat_features = [c for c in features if not c.endswith('_freq') and 
                                             not c.endswith('_te') and 
                                             not c.endswith('_le') and
                                             not c.endswith('_prod') and
                                             not c.endswith('_sum') and
                                             not c.endswith('_diff')]
        print(f"Using {len(features)} features, {len(cat_features)} categorical")

        # prepare
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)

        params = dict(
            iterations=cfg['iterations'],
            depth=cfg['depth'],
            learning_rate=cfg['learning_rate'],
            l2_leaf_reg=cfg['l2_leaf_reg'],
            **self.cfg.COMMON
        )

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y), 1):
            print(f"Training fold {fold}/{cfg['n_splits']}...")

            X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr, cat_features=cat_features),
                eval_set=Pool(X_val, y_val, cat_features=cat_features),
                use_best_model=True
            )

            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            pred += model.predict_proba(test[features])[:, 1] / cfg['n_splits']

            os.makedirs("models", exist_ok=True)
            model.save_model(f"models/{name}_enhanced_fold{fold}.cbm")
            del model; gc.collect()

        cv_auc = roc_auc_score(y, oof)
        print(f"{name} CV AUC: {cv_auc:.6f}")
        return oof, pred, features

# ================= BLENDER =================
class EnhancedBlender:
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg

    def blend(self, oof_dict, pred_dict, y_true):
        blend_oof = np.zeros_like(list(oof_dict.values())[0])
        blend_pred = np.zeros_like(list(pred_dict.values())[0])

        for name in oof_dict:
            w = self.cfg.BLEND_W[name]
            blend_oof += w * oof_dict[name]
            blend_pred += w * pred_dict[name]

        overall_auc = roc_auc_score(y_true, blend_oof)
        print(f"\nBlended CV AUC: {overall_auc:.6f}")
        return blend_pred

# ================= MAIN =================
def main():
    cfg = EnhancedConfig()
    fe = EnhancedFeatureEngineer(cfg)
    trainer = EnhancedTrainer(cfg, fe)
    blender = EnhancedBlender(cfg)

    # load test
    test = pd.read_csv(cfg.test_path)

    oof_dict, pred_dict = {}, {}
    for name, fam_cfg in cfg.CFG_LIST.items():
        oof, pred, features = trainer.train_family(name, fam_cfg, test)
        oof_dict[name] = oof
        pred_dict[name] = pred

        # сохраняем саб
        sub = pd.read_csv(cfg.submission_path)
        sub['click'] = pred
        sub.to_csv(f"submission_enhanced_{name}.csv", index=False)
        print(f"submission_enhanced_{name}.csv saved.")

    # blended
    y_true = trainer.read_sample(cfg.train_path, cfg.CFG_LIST['mid']['n_sample'])[cfg.target].values
    blend_pred = blender.blend(oof_dict, pred_dict, y_true)

    sub = pd.read_csv(cfg.submission_path)
    sub['click'] = blend_pred
    sub.to_csv("submission_enhanced_blend.csv", index=False)
    print("submission_enhanced_blend.csv saved.")

if __name__ == "__main__":
    main()