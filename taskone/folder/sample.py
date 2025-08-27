import os
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype


# ================= CONFIG =================
class CTRConfig:
    def __init__(self):
        self.train_path = "ctr_train.csv"
        self.test_path = "ctr_test.csv"
        self.submission_path = "ctr_sample_submission.csv"
        self.target = "click"

        # CatBoost configs
        self.CFG_LIST = {
            "light": dict(iterations=400, depth=8,  lr=0.03,  n_sample=500_000,  n_splits=3),
            "mid":   dict(iterations=500, depth=8,  lr=0.03,  n_sample=1_000_000, n_splits=5),
            "pro":   dict(iterations=800, depth=8, lr=0.035, n_sample=2_000_000, n_splits=5)
        }
        self.COMMON = dict(
            eval_metric="AUC",
            random_seed=42,
            task_type="GPU",
            verbose=100,
            early_stopping_rounds=40
        )
        self.BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)


# ================= FEATURE ENGINEER =================
class CTRFeatureEngineer:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.freq_maps = {}

    def fit(self, df: pd.DataFrame):
        """Обучаем маппинги (например, частотное кодирование)"""
        for col in df.columns:
            if col not in [self.cfg.target, "id"]:
                self.freq_maps[col] = df[col].value_counts(normalize=True).to_dict()

    def transform(self, df: pd.DataFrame, is_train=True):
        """Применяем преобразования"""
        for col in df.columns:
            if col not in [self.cfg.target, "id"]:
                # все категориальные храним строками
                df[col] = df[col].astype(str).fillna("__NA__")
                if col in self.freq_maps:
                    df[f"{col}_freq"] = df[col].map(self.freq_maps[col]).fillna(0)
        return df


# ================= TRAINER =================
class CTRTrainer:
    def __init__(self, cfg: CTRConfig, fe: CTRFeatureEngineer):
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

        # автоопределение категориальных
        cat_features = [c for c in features if not is_numeric_dtype(train[c])]
        print(f"Detected categorical features: {cat_features}")

        # prepare
        oof = np.zeros(len(train))
        pred = np.zeros(len(test))
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)

        params = dict(
            iterations=cfg['iterations'],
            depth=cfg['depth'],
            learning_rate=cfg['lr'],
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
            model.save_model(f"models/{name}_fold{fold}.cbm")
            del model; gc.collect()

        cv_auc = roc_auc_score(y, oof)
        print(f"{name} CV AUC: {cv_auc:.6f}")
        return oof, pred, features


# ================= BLENDER =================
class CTRBlender:
    def __init__(self, cfg: CTRConfig):
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
    cfg = CTRConfig()
    fe = CTRFeatureEngineer(cfg)
    trainer = CTRTrainer(cfg, fe)
    blender = CTRBlender(cfg)

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
        sub.to_csv(f"submission_{name}.csv", index=False)
        print(f"submission_{name}.csv saved.")

    # blended
    y_true = trainer.read_sample(cfg.train_path, cfg.CFG_LIST['mid']['n_sample'])[cfg.target].values
    blend_pred = blender.blend(oof_dict, pred_dict, y_true)

    sub = pd.read_csv(cfg.submission_path)
    sub['click'] = blend_pred
    sub.to_csv("submission_blend.csv", index=False)
    print("submission_blend.csv saved.")


if __name__ == "__main__":
    main()