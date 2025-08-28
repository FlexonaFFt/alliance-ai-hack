import pandas as pd
import numpy as np
import os
import gc
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher

class AvazuModelTrainer:
    def __init__(self):
        self.CFG_LIST = {
            "light": dict(iterations=350, depth=6, lr=0.08, n_sample=3_000_000, n_splits=3),
            "mid": dict(iterations=700, depth=8, lr=0.05, n_sample=3_000_000, n_splits=5),
            "pro": dict(iterations=1000, depth=10, lr=0.035, n_sample=6_000_000, n_splits=5)
        }
        self.COMMON = dict(
            eval_metric="AUC",
            random_seed=42,
            task_type="GPU",
            verbose=100,
            early_stopping_rounds=40,
            class_weights='balanced'
        )
        self.BLEND_W = dict(light=0.2, mid=0.4, pro=0.4)
        self.CAT_FEATURES = [
            'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
            'app_category', 'device_id', 'device_ip', 'device_model', 'device_conn_type'
        ]
        self.TOP_CAT_FEATURES = ['site_domain', 'site_id', 'device_id']
        self.NUM_FEATURES = [
            'C1', 'banner_pos', 'hour', 'device_type', 'C14', 'C15', 'C16', 'C17',
            'C18', 'C19', 'C20'
        ]
        self.TARGET = 'click'
        self.DROP_COLS = ['idx', 'id', 'device_conn_type', 'C21']
        self.scaler = StandardScaler()
        self.hasher = FeatureHasher(n_features=100, input_type='string')
        self.models = {config: [] for config in self.CFG_LIST.keys()}
        self.oof_preds = {}

    def preprocess_data(self, df, n_sample=None, is_test=False):
        if n_sample and not is_test:
            df = df.sample(n=n_sample, random_state=42)
        df.replace(-1, np.nan, inplace=True)
        df[self.NUM_FEATURES] = self.scaler.fit_transform(df[self.NUM_FEATURES].fillna(df[self.NUM_FEATURES].median()))
        for col in [c for c in self.CAT_FEATURES if c not in self.TOP_CAT_FEATURES]:
            if col in df.columns:
                hashed = self.hasher.transform(df[col].astype(str).values.reshape(-1, 1))
                df[f'{col}_hashed'] = hashed.toarray().sum(axis=1)
                df.drop(col, axis=1, errors='ignore', inplace=True)
        for col in self.TOP_CAT_FEATURES:
            if col in df.columns:
                freq = df[col].value_counts()
                df[f'{col}_freq'] = df[col].map(freq).fillna(0).astype(int)
                df.drop(col, axis=1, errors='ignore', inplace=True)
        if is_test:
            return df
        else:
            df = df.drop(self.DROP_COLS, axis=1, errors='ignore')
            return df

    def train_model(self, config_name, df, cfg, common):
        print(f"\nTraining {config_name} configuration...")
        X = df.drop(self.TARGET, axis=1)
        y = df[self.TARGET]
        skf = StratifiedKFold(n_splits=cfg['n_splits'], shuffle=True, random_state=42)
        oof_preds = np.zeros(len(df))
        auc_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{cfg['n_splits']}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            train_pool = Pool(X_train, y_train, cat_features=[f'{col}_hashed' for col in self.CAT_FEATURES if col not in self.TOP_CAT_FEATURES] + [f'{col}_freq' for col in self.TOP_CAT_FEATURES])
            val_pool = Pool(X_val, y_val, cat_features=[f'{col}_hashed' for col in self.CAT_FEATURES if col not in self.TOP_CAT_FEATURES] + [f'{col}_freq' for col in self.TOP_CAT_FEATURES])
            model = CatBoostClassifier(**common, **{k: v for k, v in cfg.items() if k in ['iterations', 'depth', 'lr']})
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            val_preds = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_preds)
            auc_scores.append(auc)
            oof_preds[val_idx] = val_preds
            self.models[config_name].append(model)
            print(f"Fold {fold + 1} AUC: {auc:.4f}")
            gc.collect()
        oof_auc = roc_auc_score(y, oof_preds)
        print(f"OOF AUC for {config_name}: {oof_auc:.4f}")
        self.oof_preds[config_name] = oof_preds
        return oof_auc

    def predict_test(self, test_df):
        print("\nPredicting on test data...")
        test_preds = {}
        for config_name in self.CFG_LIST.keys():
            fold_preds = np.zeros(len(test_df))
            for fold, model in enumerate(self.models[config_name]):
                test_pool = Pool(test_df, cat_features=[f'{col}_hashed' for col in self.CAT_FEATURES if col not in self.TOP_CAT_FEATURES] + [f'{col}_freq' for col in self.TOP_CAT_FEATURES])
                fold_preds += model.predict_proba(test_pool)[:, 1] / len(self.models[config_name])
            test_preds[config_name] = fold_preds
        final_preds = sum(self.BLEND_W[cfg] * test_preds[cfg] for cfg in self.CFG_LIST.keys())
        return final_preds

    def run(self, train_file_path, test_file_path):
        print("Loading dataset...")
        df_train = pd.read_csv(train_file_path, nrows=self.CFG_LIST['pro']['n_sample'])
        print("Preprocessing training dataset...")
        df_train = self.preprocess_data(df_train)
        for config_name, cfg in self.CFG_LIST.items():
            self.train_model(config_name, df_train, cfg, self.COMMON)
        final_oof_preds = sum(self.BLEND_W[cfg] * self.oof_preds[cfg] for cfg in self.CFG_LIST.keys())
        final_oof_auc = roc_auc_score(df_train[self.TARGET], final_oof_preds)
        print(f"Final Blended OOF AUC: {final_oof_auc:.4f}")
        print("Loading and preprocessing test dataset...")
        df_test = pd.read_csv(test_file_path)
        df_test = df_test.rename(columns={'idx': 'id'})
        df_test = pd.merge(df_test, df_train.drop(self.TARGET, axis=1), on='id', how='left')
        df_test = self.preprocess_data(df_test, is_test=True)
        final_test_preds = self.predict_test(df_test)
        submission = pd.DataFrame({
            'id': df_test['id'],
            'click': final_test_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("Predictions saved to 'submission.csv'")

if __name__ == "__main__":
    trainer = AvazuModelTrainer()
    trainer.run(
        train_file_path="ctr_train.csv",
        test_file_path="ctr_sample_submission.csv"
    )