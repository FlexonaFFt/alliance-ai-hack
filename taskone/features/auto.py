import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc

class Config:
    def __init__(self):
        self.train_path = "ctr_train.csv"
        self.test_path = "ctr_test.csv"
        self.submission_path = "ctr_sample_submission.csv"
        self.target = "click"
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
        self.sample_size = 2_000_000
        self.n_trials = 5
        self.n_splits = 5

class Preprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.freq_encodings = {}
        self.test_ids = None  # Для хранения оригинальных id из тестового набора

    def load_data(self):
        use_cols = self.config.features + [self.config.target, 'id']
        print(f"Loading {self.config.sample_size} samples...")
        train = pd.read_csv(self.config.train_path, usecols=use_cols, nrows=self.config.sample_size)
        test = pd.read_csv(self.config.test_path, usecols=self.config.features + ['id'])

        # Сохраняем оригинальные id из тестового набора данных
        self.test_ids = test['id'].copy()

        print(f"Train shape: {train.shape}, Test shape: {test.shape}")
        return train, test

    def preprocess(self, train: pd.DataFrame, test: pd.DataFrame):
        X_train = train[self.config.features].copy()
        y_train = train[self.config.target].copy().astype('int8')
        X_test = test[self.config.features].copy()

        del train, test
        gc.collect()

        X_train = X_train.fillna("NA")
        X_test = X_test.fillna("NA")

        print("Applying frequency encoding...")
        for col in self.config.large_cat_features:
            freq = X_train[col].value_counts().to_dict()
            self.freq_encodings[col] = freq
            X_train[col] = X_train[col].map(freq).astype("float32")
            X_test[col] = X_test[col].map(freq).fillna(0).astype("float32")

        print("Applying OHE...")
        for col in self.config.small_cat_features:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        self.ohe.fit(X_train[self.config.small_cat_features])
        ohe_train = self.ohe.transform(X_train[self.config.small_cat_features])
        ohe_test = self.ohe.transform(X_test[self.config.small_cat_features])
        ohe_cols = self.ohe.get_feature_names_out(self.config.small_cat_features)
        ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_cols, index=X_train.index).astype('float32')
        ohe_test_df = pd.DataFrame(ohe_test, columns=ohe_cols, index=X_test.index).astype('float32')

        X_train = X_train.drop(columns=self.config.small_cat_features)
        X_test = X_test.drop(columns=self.config.small_cat_features)
        X_train = pd.concat([X_train, ohe_train_df], axis=1)
        X_test = pd.concat([X_test, ohe_test_df], axis=1)

        for col in X_train.columns:
            if X_train[col].dtype == 'float64':
                X_train[col] = X_train[col].astype('float32')
            if col in X_test.columns and X_test[col].dtype == 'float64':
                X_test[col] = X_test[col].astype('float32')

        print(f"Final train shape: {X_train.shape}, test shape: {X_test.shape}")
        return X_train, y_train, X_test

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config
        self.best_params = None

    def objective(self, trial, X, y):
        params = {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03),
            "num_leaves": trial.suggest_int("num_leaves", 32, 128),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 5),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 5),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "max_bin": trial.suggest_int("max_bin", 50, 150),
            "verbose": -1
        }

        cv = StratifiedKFold(n_splits=self.config.n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, valid_idx in cv.split(X, y):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            train_set = lgb.Dataset(X_train, y_train)
            valid_set = lgb.Dataset(X_valid, y_valid, reference=train_set)
            model = lgb.train(
                params, train_set,
                num_boost_round=1000,
                valid_sets=[valid_set],
                callbacks=[lgb.callback.reset_parameter()]
            )
            preds = model.predict(X_valid, num_iteration=model.best_iteration)
            score = roc_auc_score(y_valid, preds)
            scores.append(score)

            del model, train_set, valid_set
            gc.collect()

        return np.mean(scores)

    def tune(self, X, y):
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X, y), n_trials=self.config.n_trials)
        self.best_params = study.best_params
        print("Best params:", self.best_params)
        return self.best_params

class CTRModel:
    def __init__(self, params, preprocessor: Preprocessor):
        self.params = params
        self.model = None
        self.preprocessor = preprocessor

    def fit(self, X, y):
        dtrain = lgb.Dataset(X, y)
        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=1000
        )

    def predict(self, X):
        return self.model.predict(X)

    def save_submission(self, preds, path):
        sub = pd.DataFrame({"id": self.preprocessor.test_ids, "click": preds})
        sub.to_csv(path, index=False)

class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.prep = Preprocessor(config)
        self.trainer = ModelTrainer(config)
        self.model = None

    def run(self):
        print("Starting pipeline...")
        train, test = self.prep.load_data()
        gc.collect()

        X_train, y_train, X_test = self.prep.preprocess(train, test)
        gc.collect()

        print("Tuning hyperparameters...")
        best_params = self.trainer.tune(X_train, y_train)
        best_params.update({
            "objective": "binary",
            "metric": "auc",
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "verbose": -1
        })

        print("Training final model...")
        self.model = CTRModel(best_params, self.prep)
        self.model.fit(X_train, y_train)

        print("Making predictions...")
        preds = self.model.predict(X_test)
        self.model.save_submission(preds, self.config.submission_path)

        print("Pipeline completed successfully!")

if __name__ == "__main__":
    cfg = Config()
    pipeline = Pipeline(cfg)
    pipeline.run()