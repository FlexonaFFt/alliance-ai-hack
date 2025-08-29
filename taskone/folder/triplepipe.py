import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib


class DataPreprocessor:
    def __init__(self, cat_cols=None, num_cols=None):
        if cat_cols is None:
            self.cat_cols = [
                'site_id', 'site_domain', 'site_category',
                'app_id', 'app_domain', 'app_category',
                'device_id', 'device_ip', 'device_model'
            ]
        else:
            self.cat_cols = cat_cols
        if num_cols is None:
            self.num_cols = [
                'hour', 'C1', 'banner_pos', 'device_type', 'device_conn_type',
                'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'
            ]
        else:
            self.num_cols = num_cols
        self.label_encoders = {col: LabelEncoder() for col in self.cat_cols}

    def load_data(self, path, is_train=True, nrows=None):
        df = pd.read_csv(path, nrows=nrows)
        df.columns = df.columns.str.strip()

        if is_train:
            y = df['click'].values
            X = df[self.cat_cols + self.num_cols].copy()
            return X, y
        else:
            X = df[self.cat_cols + self.num_cols].copy()
            if 'idx' in df.columns:
                idx = df['idx'].astype(str).values
            elif 'id' in df.columns:
                idx = df['id'].astype(str).values
            else:
                raise KeyError("В тестовом наборе нет столбца 'id' или 'idx'.")
            return X, idx

    def fit_transform_cat(self, X):
        X_cat = X[self.cat_cols].copy()
        for col in self.cat_cols:
            X_cat[col] = self.label_encoders[col].fit_transform(X_cat[col].astype(str))
        return X_cat

    def transform_cat(self, X):
        X_cat = X[self.cat_cols].copy()
        for col in self.cat_cols:
            le = self.label_encoders[col]
            mapping = {cat: idx for idx, cat in enumerate(le.classes_)}
            X_cat[col] = X_cat[col].astype(str).map(mapping).fillna(-1).astype(int)
        return X_cat

    def preprocess_train(self, X):
        X_cat = self.fit_transform_cat(X)
        X_num = X[self.num_cols].copy()
        X_all = pd.concat([X_cat, X_num], axis=1)
        return X_all.values

    def preprocess_test(self, X):
        X_cat = self.transform_cat(X)
        X_num = X[self.num_cols].copy()
        X_all = pd.concat([X_cat, X_num], axis=1)
        return X_all.values

    def balance_data(self, X, y):
        Xy = pd.DataFrame(X)
        Xy['click'] = y
        df_majority = Xy[Xy['click'] == 0]
        df_minority = Xy[Xy['click'] == 1]
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority], ignore_index=True)
        y_balanced = df_balanced['click'].values
        X_balanced = df_balanced.drop('click', axis=1).values
        return X_balanced, y_balanced


class ClickPredictor:
    def __init__(self, n_estimators=100, random_state=42, verbose=True):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            n_jobs=-1,
            random_state=random_state,
            class_weight='balanced_subsample',
            verbose=1 if verbose else 0
        )

    def cross_validate(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f'\nTraining fold {fold}/{n_splits}...')
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
            scores.append(score)
            print(f'Fold {fold} ROC-AUC: {score:.4f}')
        print(f'Mean ROC-AUC: {np.mean(scores):.4f}')
        return scores

    def fit(self, X, y):
        print('\nFinal training on balanced data...')
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save_model(self, path):
        joblib.dump(self.model, path)

    def load_model(self, path):
        self.model = joblib.load(path)


def main():
    train_path = 'datasearch/ctr_train.csv'
    test_path = 'datasearch/ctr_test.csv'
    submission_path = 'datasearch/ctr_sample_submission.csv'
    output_path = 'datasearch/ctr_submission_fixed.csv'

    preprocessor = DataPreprocessor()
    X_train_raw, y_train = preprocessor.load_data(train_path, is_train=True, nrows=5_000_000)
    X_train_processed = preprocessor.preprocess_train(X_train_raw)
    X_train_balanced, y_train_balanced = preprocessor.balance_data(X_train_processed, y_train)

    predictor = ClickPredictor(n_estimators=100, verbose=True)
    predictor.cross_validate(X_train_balanced, y_train_balanced, n_splits=5)
    predictor.fit(X_train_balanced, y_train_balanced)

    sample_df = pd.read_csv(submission_path, dtype={'idx': str})
    sample_idx = sample_df['idx'].astype(str)

    X_test_raw, test_ids = preprocessor.load_data(test_path, is_train=False)
    X_test_processed = preprocessor.preprocess_test(X_test_raw)

    y_test_pred = predictor.predict_proba(X_test_processed)
    preds_df = pd.DataFrame({
        'idx': pd.Series(test_ids, dtype=str),
        'click': y_test_pred
    })


    submission = sample_df[['idx']].merge(preds_df, on='idx', how='left')
    missing = submission['click'].isna().sum()
    if missing > 0:
        print(f'Внимание: для {missing} idx нет предсказаний. Проверь соответствие id/idx в тесте и sample.')

    submission.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')


if __name__ == '__main__':
    main()
