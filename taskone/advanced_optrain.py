import pandas as pd
import numpy as np
import os
import gc
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from itertools import combinations


class CTRConfig:
    def __init__(self):
        self.train_path = "ctr_train.csv"
        self.test_path = "ctr_test.csv"
        self.submission_path = "ctr_sample_submission.csv"
        self.target = "click"
        
        self.CFG_LIST = {
            "light": dict(iterations=700, depth=7,  lr=0.04,  n_sample=2_000_000,  n_splits=3),
            "mid":   dict(iterations=1000, depth=8,  lr=0.03,  n_sample=6_000_000, n_splits=5),
            "pro":   dict(iterations=3000, depth=9, lr=0.025, n_sample=6_000_000, n_splits=5)
        }
        
        self.COMMON = dict(
            eval_metric="AUC",
            random_seed=42,
            task_type="GPU",     
            verbose=100,
            early_stopping_rounds=50,
            l2_leaf_reg=3.0,
            bootstrap_type="Bayesian",
            random_strength=1.0
        )
        
        self.BLEND_W = dict(light=0.15, mid=0.35, pro=0.5)
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
        
        self.interaction_params = {
            'max_cat_interactions': 3,
            'max_num_interactions': 5,
            'top_k_interactions': 20,
        }


class FeatureEngineer:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.freq_maps = {}
        self.target_maps = {}
        self.label_encoders = {}
        self.interaction_features = []
        self.interaction_freq_maps = {}
    
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
        
        self._create_interactions(df, y)
        
        print("Маппинги для кодирования признаков обучены.")
    
    def _create_interactions(self, df, y=None):
        print("Создание взаимодействий признаков...")
        
        cat_features = self.cfg.small_cat_features + self.cfg.large_cat_features
        cat_pairs = list(combinations(cat_features[:10], 2))
        
        for col1, col2 in cat_pairs[:self.cfg.interaction_params['max_cat_interactions']]:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_{col2}_concat"
                df[interaction_name] = df[col1].astype(str) + "_" + df[col2].astype(str)
                self.interaction_features.append(interaction_name)
                
                self.interaction_freq_maps[interaction_name] = df[interaction_name].value_counts(normalize=True).to_dict()
        
        num_features = [f"{col}_freq" for col in self.cfg.features if f"{col}_freq" in df.columns]
        num_features += [f"{col}_te" for col in self.cfg.features if f"{col}_te" in df.columns]
        
        if len(num_features) >= 2:
            num_pairs = list(combinations(num_features[:10], 2))
            
            for col1, col2 in num_pairs[:self.cfg.interaction_params['max_num_interactions']]:
                interaction_name = f"{col1}_{col2}_prod"
                df[interaction_name] = df[col1] * df[col2]
                self.interaction_features.append(interaction_name)
                
                interaction_name = f"{col1}_{col2}_sum"
                df[interaction_name] = df[col1] + df[col2]
                self.interaction_features.append(interaction_name)
                
                interaction_name = f"{col1}_{col2}_diff"
                df[interaction_name] = df[col1] - df[col2]
                self.interaction_features.append(interaction_name)
        
        if y is not None and len(self.interaction_features) > self.cfg.interaction_params['top_k_interactions']:
            correlations = []
            for feature in self.interaction_features:
                if feature in df.columns:
                    corr = np.abs(np.corrcoef(df[feature].fillna(0).values, y)[0, 1])
                    correlations.append((feature, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            self.interaction_features = [x[0] for x in correlations[:self.cfg.interaction_params['top_k_interactions']]]
        
        print(f"Создано {len(self.interaction_features)} взаимодействий признаков")
    
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
            if col in df.columns and col in self.label_encoders:
                df[f"{col}_le"] = self.label_encoders[col].transform(df[col].astype(str).fillna("__NA__"))
        
        for feature in self.interaction_features:
            if "_concat" in feature:
                col1, col2 = feature.split("_concat")[0].split("_", 1)
                if col1 in df.columns and col2 in df.columns:
                    df[feature] = df[col1].astype(str) + "_" + df[col2].astype(str)
                    
                    if feature in self.interaction_freq_maps:
                        df[f"{feature}_freq"] = df[feature].map(self.interaction_freq_maps[feature]).fillna(0)
        
        for feature in self.interaction_features:
            if "_prod" in feature or "_sum" in feature or "_diff" in feature:
                parts = feature.rsplit("_", 1)[0].split("_", 1)
                if len(parts) == 2:
                    col1, col2 = parts
                    if col1 in df.columns and col2 in df.columns:
                        if "_prod" in feature:
                            df[feature] = df[col1] * df[col2]
                        elif "_sum" in feature:
                            df[feature] = df[col1] + df[col2]
                        elif "_diff" in feature:
                            df[feature] = df[col1] - df[col2]
        
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
        self.feature_importances = {}
    
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
                          not col.endswith('_le') and
                          not ("_prod" in col or "_sum" in col or "_diff" in col)]
        
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
        
        if name == "pro":
            params.update({
                "colsample_bylevel": 0.7,
                "subsample": 0.9,
            })
        
        self.feature_importances[name] = {}
        
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
            
            fold_importances = model.get_feature_importance(Pool(X_val, y_val, cat_features=cat_features))
            fold_importance_dict = dict(zip(features, fold_importances))
            for feature, importance in fold_importance_dict.items():
                if feature not in self.feature_importances[name]:
                    self.feature_importances[name][feature] = []
                self.feature_importances[name][feature].append(importance)
            
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
            pred += model.predict_proba(self.test_processed[features])[:, 1] / cfg['n_splits']
            
            os.makedirs("models", exist_ok=True)
            model.save_model(f"models/{name}_fold{fold}.cbm")
            del model; gc.collect()
        
        cv_auc = roc_auc_score(y, oof)
        print(f"{name} CV AUC: {cv_auc:.6f}")
        
        avg_importances = {}
        for feature, importances in self.feature_importances[name].items():
            avg_importances[feature] = np.mean(importances)
        
        print(f"\nТоп-20 важных признаков для {name}:")
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:20]
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.6f}")
        
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


class HyperparamOptimizer:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.best_params = {}
        self.best_score = 0.0
    
    def objective(self, trial, X, y, cat_features):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 3000),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
            'eval_metric': 'AUC',
            'task_type': 'GPU',
            'random_seed': 42,
            'verbose': 0
        }
        
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        elif params['bootstrap_type'] == 'Bernoulli':
            params['subsample'] = trial.suggest_float('subsample', 0.1, 1.0)
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                Pool(X_train, y_train, cat_features=cat_features),
                eval_set=Pool(X_val, y_val, cat_features=cat_features),
                early_stopping_rounds=50,
                verbose=0
            )
            
            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    def optimize(self, name="mid", n_trials=20):
        print(f"\n=== Оптимизация гиперпараметров для '{name}' ===")
        
        cfg = self.cfg.CFG_LIST[name]
        sample_size = min(cfg['n_sample'], 1_000_000)
        
        train = pd.read_csv(self.cfg.train_path, nrows=sample_size)
        features_in_train = [col for col in self.cfg.features if col in train.columns]
        train = train[['id', self.cfg.target] + features_in_train]
        
        fe = FeatureEngineer(self.cfg)
        fe.fit(train)
        train_processed = fe.transform(train)
        
        y = train_processed[self.cfg.target]
        X = train_processed.drop(['id', self.cfg.target], axis=1)
        
        cat_features = [col for col in X.columns 
                       if not col.endswith('_freq') and 
                          not col.endswith('_te') and 
                          not col.endswith('_le') and
                          not ("_prod" in col or "_sum" in col or "_diff" in col)]
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: self.objective(trial, X, y, cat_features), n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"Лучший AUC: {self.best_score:.6f}")
        print(f"Лучшие параметры: {self.best_params}")
        
        os.makedirs("params", exist_ok=True)
        with open(f"params/{name}_best_params.txt", "w") as f:
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
        
        return self.best_params, self.best_score


class StackingEnsemble:
    def __init__(self, cfg: CTRConfig):
        self.cfg = cfg
        self.base_models = {}
        self.meta_model = None
        self.base_oof_preds = {}
        self.base_test_preds = {}
        self.meta_features_train = None
        self.meta_features_test = None
    
    def train_base_models(self, trainer):
        print("\n=== Обучение базовых моделей для стекинга ===")
        
        for name in self.cfg.CFG_LIST.keys():
            try:
                oof_preds = np.load(f"models/{name}_oof.npy")
                test_preds = np.load(f"models/{name}_pred.npy")
                self.base_oof_preds[name] = oof_preds
                self.base_test_preds[name] = test_preds
                print(f"Загружены предсказания для модели {name}")
            except FileNotFoundError:
                print(f"Предсказания для модели {name} не найдены. Сначала обучите базовые модели.")
                return False
        
        return True
    
    def add_extra_models(self):
        print("\n=== Добавление дополнительных моделей для стекинга ===")
        
        train_sample = pd.read_csv(self.cfg.train_path, nrows=self.cfg.CFG_LIST['mid']['n_sample'])
        test = pd.read_csv(self.cfg.test_path)
        
        features_in_train = [col for col in self.cfg.features if col in train_sample.columns]
        train_sample = train_sample[['id', self.cfg.target] + features_in_train]
        
        features_in_test = [col for col in self.cfg.features if col in test.columns]
        test = test[['id'] + features_in_test]
        
        y = train_sample[self.cfg.target].values
        
        fe = FeatureEngineer(self.cfg)
        fe.fit(train_sample)
        train_processed = fe.transform(train_sample)
        test_processed = fe.transform(test.copy(), is_train=False)
        
        features = [col for col in train_processed.columns 
                   if col not in ['id', self.cfg.target]]
        
        cat_features = [col for col in features 
                       if not col.endswith('_freq') and 
                          not col.endswith('_te') and 
                          not col.endswith('_le') and
                          not ("_prod" in col or "_sum" in col or "_diff" in col)]
        
        try:
            from lightgbm import LGBMClassifier
            
            print("Обучение LightGBM модели...")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lgb_oof = np.zeros(len(train_processed))
            lgb_pred = np.zeros(len(test_processed))
            
            for fold, (tr_idx, val_idx) in enumerate(skf.split(train_processed[features], y)):
                print(f"LightGBM: Обучение фолда {fold+1}/5...")
                
                X_tr = train_processed.iloc[tr_idx][features]
                X_val = train_processed.iloc[val_idx][features]
                y_tr, y_val = y[tr_idx], y[val_idx]
                
                for col in cat_features:
                    if col in X_tr.columns:
                        X_tr[col] = X_tr[col].astype('category')
                        X_val[col] = X_val[col].astype('category')
                
                model = LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.03,
                    max_depth=8,
                    num_leaves=128,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
                
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    early_stopping_rounds=50,
                    verbose=0
                )
                
                lgb_oof[val_idx] = model.predict_proba(X_val)[:, 1]
                lgb_pred += model.predict_proba(test_processed[features])[:, 1] / 5
            
            lgb_auc = roc_auc_score(y, lgb_oof)
            print(f"LightGBM CV AUC: {lgb_auc:.6f}")
            
            np.save("models/lgb_oof.npy", lgb_oof)
            np.save("models/lgb_pred.npy", lgb_pred)
            
            self.base_oof_preds["lgb"] = lgb_oof
            self.base_test_preds["lgb"] = lgb_pred
            
            sub_lgb = pd.read_csv(self.cfg.submission_path)
            sub_lgb['click'] = lgb_pred
            sub_lgb.to_csv("submission_lgb.csv", index=False)
            print("submission_lgb.csv сохранён.")
            
        except ImportError:
            print("LightGBM не установлен. Пропускаем добавление LightGBM модели.")
        
        return True
    
    def prepare_meta_features(self):
        print("\n=== Подготовка мета-признаков для стекинга ===")
        
        train_sample = pd.read_csv(self.cfg.train_path, nrows=self.cfg.CFG_LIST['mid']['n_sample'])
        y = train_sample[self.cfg.target].values if self.cfg.target in train_sample.columns else None
        
        meta_features_list = []
        meta_features_names = []
        
        for name, oof_preds in self.base_oof_preds.items():
            if len(oof_preds) > len(train_sample):
                oof_preds = oof_preds[:len(train_sample)]
            elif len(oof_preds) < len(train_sample):
                print(f"Предупреждение: размер OOF предсказаний для {name} ({len(oof_preds)}) меньше, чем размер train_sample ({len(train_sample)})")
                continue
            
            meta_features_list.append(oof_preds.reshape(-1, 1))
            meta_features_names.append(name)
        
        if meta_features_list:
            self.meta_features_train = np.hstack(meta_features_list)
            print(f"Созданы мета-признаки для обучения размерностью {self.meta_features_train.shape}")
        else:
            print("Не удалось создать мета-признаки для обучения")
            return False
        
        test_meta_features_list = []
        
        for name in meta_features_names:
            test_preds = self.base_test_preds[name]
            test_meta_features_list.append(test_preds.reshape(-1, 1))
        
        self.meta_features_test = np.hstack(test_meta_features_list)
        print(f"Созданы мета-признаки для теста размерностью {self.meta_features_test.shape}")
        
        return True
    
    def train_meta_model(self):
        print("\n=== Обучение мета-модели для стекинга ===")
        
        train_sample = pd.read_csv(self.cfg.train_path, nrows=self.cfg.CFG_LIST['mid']['n_sample'])
        y = train_sample[self.cfg.target].values
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_oof = np.zeros(len(self.meta_features_train))
        meta_pred = np.zeros(len(self.meta_features_test))
        
        for fold, (tr_idx, val_idx) in enumerate(skf.split(self.meta_features_train, y)):
            print(f"Мета-модель: Обучение фолда {fold+1}/5...")
            
            X_tr = self.meta_features_train[tr_idx]
            X_val = self.meta_features_train[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            
            meta_model = CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                eval_metric="AUC",
                random_seed=42,
                task_type="GPU",
                verbose=0
            )
            
            meta_model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=0
            )
            
            meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]
            meta_pred += meta_model.predict_proba(self.meta_features_test)[:, 1] / 5
            
            if fold == 0:
                self.meta_model = meta_model
        
        meta_auc = roc_auc_score(y, meta_oof)
        print(f"Мета-модель CV AUC: {meta_auc:.6f}")
        
        np.save("models/meta_oof.npy", meta_oof)
        np.save("models/meta_pred.npy", meta_pred)
        
        sub_meta = pd.read_csv(self.cfg.submission_path)
        sub_meta['click'] = meta_pred
        sub_meta.to_csv("submission_stacking.csv", index=False)
        print("submission_stacking.csv сохранён.")
        
        return meta_auc
    
    def run(self, trainer):
        print("\n=== Запуск стекинга моделей ===")
        
        if not self.train_base_models(trainer):
            print("Не удалось загрузить предсказания базовых моделей. Сначала обучите базовые модели.")
            return None
        
        self.add_extra_models()
        
        if not self.prepare_meta_features():
            print("Не удалось подготовить мета-признаки для стекинга.")
            return None
        
        meta_auc = self.train_meta_model()
        
        print(f"Стекинг завершен. Итоговый AUC мета-модели: {meta_auc:.6f}")
        return meta_auc


class Pipeline:
    def __init__(self):
        self.cfg = CTRConfig()
        self.trainer = CTRTrainer(self.cfg)
        self.optimizer = HyperparamOptimizer(self.cfg)
        self.stacking = StackingEnsemble(self.cfg)
    
    def run(self, optimize_hyperparams=False, use_stacking=True):
        print("Запуск пайплайна обучения...")
        self.trainer.load_test()
        
        if optimize_hyperparams:
            print("\n=== Запуск оптимизации гиперпараметров ===")
            for name in ["mid", "pro"]:
                best_params, _ = self.optimizer.optimize(name, n_trials=20)
                
                for param, value in best_params.items():
                    if param in ["iterations", "depth", "learning_rate"]:
                        if param == "iterations":
                            self.cfg.CFG_LIST[name]["iterations"] = value
                        elif param == "depth":
                            self.cfg.CFG_LIST[name]["depth"] = value
                        elif param == "learning_rate":
                            self.cfg.CFG_LIST[name]["lr"] = value
                    else:
                        self.cfg.COMMON[param] = value
        
        for name, cfg in self.cfg.CFG_LIST.items():
            self.trainer.train_family(name, cfg)
        
        blend_auc = self.trainer.save_blend()
        print(f"Блендинг завершен. AUC блендинга: {blend_auc:.6f}")
        
        stacking_auc = None
        if use_stacking:
            stacking_auc = self.stacking.run(self.trainer)
            if stacking_auc is not None:
                print(f"Стекинг улучшил AUC с {blend_auc:.6f} до {stacking_auc:.6f}")
        
        final_auc = stacking_auc if stacking_auc is not None else blend_auc
        print(f"Пайплайн завершен. Итоговый AUC: {final_auc:.6f}")
        return final_auc


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run(optimize_hyperparams=False, use_stacking=True)  # Включаем стекинг