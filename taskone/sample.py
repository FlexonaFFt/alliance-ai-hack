import os, gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class CTRConfig:
    train_path = "ctr_train.csv"
    test_path = "ctr_test.csv"
    submission_path = "ctr_sample_submission.csv"
    target = "click"

    features = [
        'hour', 'banner_pos', 'site_category', 'app_category',
        'device_model', 'device_type', 'device_conn_type',
        'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]

    small_cat_features = ['hour', 'banner_pos', 'device_type', 'device_conn_type']
    large_cat_features = [
        'site_category', 'app_category', 'device_model',
        'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
    ]

    CFG_LIST = {
        "light": dict(iterations=400, depth=8,  lr=0.03,  n_sample=500_000,  n_splits=3),
        "mid":   dict(iterations=600, depth=8,  lr=0.03,  n_sample=1_000_000, n_splits=5),
        "pro":   dict(iterations=800, depth=10, lr=0.033, n_sample=2_000_000, n_splits=5)
    }

    COMMON = dict(
        eval_metric="AUC",
        random_seed=42,
        task_type="GPU",
        verbose=100,
        early_stopping_rounds=40
    )

    BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)


class CTRFeatureEngineer:
    def __init__(self, config: CTRConfig):
        self.cfg = config

    def reduce_rare_categories(self, df, cols, min_freq=20):
        for c in cols:
            freq = df[c].value_counts()
            rare = freq[freq < min_freq].index
            df[c] = df[c].replace(rare, "__OTHER__")
        return df

    def frequency_encoding(self, df, col):
        freq = df[col].value_counts()
        df[col + "_freq"] = df[col].map(freq)
        return df

    def add_cross_features(self, df, col1, col2):
        name = f"{col1}_{col2}"
        df[name] = df[col1].astype(str) + "_" + df[col2].astype(str)
        return df

    def transform(self, df: pd.DataFrame, is_train=True):
        df = self.reduce_rare_categories(df, self.cfg.large_cat_features)

        for col in self.cfg.large_cat_features:
            df = self.frequency_encoding(df, col)

        df = self.add_cross_features(df, "hour", "device_type")
        df = self.add_cross_features(df, "banner_pos", "app_category")

        return df


class CTRTrainer:
    def __init__(self, config: CTRConfig, fe: CTRFeatureEngineer):
        self.cfg = config
        self.fe = fe

    def read_sample(self, path, n_rows):
        return pd.read_csv(path, nrows=n_rows)

    def train_family(self, name, cfg, test):
        print(f"\n=== Training family '{name}' ===")

        train = self.read_sample(self.cfg.train_path, cfg['n_sample'])
        y = train[self.cfg.target].values

        # feature engineering
        train = self.fe.transform(train)
        test = self.fe.transform(test.copy(), is_train=False)
        features = [c for c in train.columns if c not in ['id', self.cfg.target]]

        oof, pred = np.zeros(len(train)), np.zeros(len(test))
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
        params = dict(
            iterations=cfg['iterations'],
            depth=cfg['depth'],
            learning_rate=cfg['lr'],
            **self.cfg.COMMON
        )

        for fold, (tr_idx, val_idx) in enumerate(skf.split(train[features], y)):
            print(f"Training fold {fold+1}/{cfg['n_splits']}...")

            X_tr, X_val = train.iloc[tr_idx][features], train.iloc[val_idx][features]
            y_tr, y_val = y[tr_idx], y[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_tr, y_tr, cat_features=self.cfg.small_cat_features + self.cfg.large_cat_features),
                eval_set=Pool(X_val, y_val, cat_features=self.cfg.small_cat_features + self.cfg.large_cat_features),
                use_best_model=True
            )

            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            pred += model.predict_proba(test[features])[:, 1] / cfg['n_splits']

            os.makedirs("models", exist_ok=True)
            model.save_model(f"models/{name}_fold{fold}.cbm")
            del model; gc.collect()

        cv_auc = roc_auc_score(y, oof)
        print(f"{name} CV AUC: {cv_auc:.6f}")

        return oof, pred


class CTRBlender:
    def __init__(self, config: CTRConfig):
        self.cfg = config

    def blend(self, oof_dict, pred_dict, y_true):
        blend_oof, blend_pred = np.zeros(len(y_true)), np.zeros(len(list(pred_dict.values())[0]))

        for name, w in self.cfg.BLEND_W.items():
            blend_oof += w * oof_dict[name]
            blend_pred += w * pred_dict[name]

        auc = roc_auc_score(y_true, blend_oof)
        print(f"\nBlended CV AUC: {auc:.6f}")
        return blend_oof, blend_pred


def main():
    cfg = CTRConfig()
    fe = CTRFeatureEngineer(cfg)
    trainer = CTRTrainer(cfg, fe)
    blender = CTRBlender(cfg)

    test = pd.read_csv(cfg.test_path)
    oof_dict, pred_dict = {}, {}

    for name, fam_cfg in cfg.CFG_LIST.items():
        oof, pred = trainer.train_family(name, fam_cfg, test)
        oof_dict[name], pred_dict[name] = oof, pred

    y_true = trainer.read_sample(cfg.train_path, cfg.CFG_LIST['mid']['n_sample'])[cfg.target].values
    blend_oof, blend_pred = blender.blend(oof_dict, pred_dict, y_true)

    sub = pd.read_csv(cfg.submission_path)
    sub['click'] = blend_pred
    sub.to_csv("submission_blend.csv", index=False)
    print("submission_blend.csv saved!")


if __name__ == "__main__":
    main()
