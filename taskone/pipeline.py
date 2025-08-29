import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import category_encoders as ce

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CTRModelTrainer:
    FEATURES = [
        'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
        'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type',
        'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'month', 'dayofweek', 'day', 'hour_time'
    ]
    TARGET = 'click'

    def __init__(self, train_path, test_path, submission_path, sample_size=5_000_000, sample_frac=0.1, random_state=42):
        self.train_path = train_path
        self.test_path = test_path
        self.submission_path = submission_path
        self.sample_size = sample_size
        self.sample_frac = sample_frac
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1, class_weight='balanced')
        self.target_encoder = ce.TargetEncoder()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.test = None

    def load_data(self):
        logging.info("Загрузка данных...")
        train = pd.read_csv(self.train_path, nrows=self.sample_size)
        test = pd.read_csv(self.test_path)
        submission = pd.read_csv(self.submission_path)
        logging.info(f"Train dataset: {train.shape}")
        logging.info(f"Test dataset: {test.shape}")
        logging.info(f"Submission: {submission.shape}")
        return train, test, submission

    def preprocess(self, train, test):
        logging.info("Предобработка данных...")
        features = [f for f in self.FEATURES if f in train.columns and f in test.columns]
        target = self.TARGET

        if target not in train.columns:
            raise ValueError(f"Целевая переменная '{target}' отсутствует в train")

        train = train[features + [target]]
        test = test[features]

        sampled_data = train.sample(frac=self.sample_frac, random_state=self.random_state)
        X = sampled_data[features]
        y = sampled_data[target]

        X_encoded = self.target_encoder.fit_transform(X, y)
        test_encoded = self.target_encoder.transform(test)
        return X_encoded, y, test_encoded

    def balance_classes(self, X, y):
        logging.info("Балансировка классов...")
        ros = RandomOverSampler(random_state=self.random_state)
        X_res, y_res = ros.fit_resample(X, y)
        return X_res, y_res

    def split_data(self, X, y):
        logging.info("Разделение данных на train/test...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.3, random_state=self.random_state
        )
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def train_model(self):
        logging.info("Обучение модели Random Forest...")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        logging.info("Оценка модели...")
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        y_pred = (y_pred_proba > 0.5).astype(int)
        cnf_matrix = confusion_matrix(self.y_test, y_pred)
        logging.info(f"ROC AUC: {roc_auc:.6f}")
        logging.info(f"Confusion Matrix:\n{cnf_matrix}")
        print(f"\nBlended CV AUC: {roc_auc:.6f}")
        print("Confusion Matrix:")
        print(cnf_matrix)
        return roc_auc, cnf_matrix

    def predict_test(self, test_encoded):
        logging.info("Предсказание на тестовых данных...")
        y_test_pred_proba = self.model.predict_proba(test_encoded)[:, 1]
        return y_test_pred_proba

    def save_submission(self, submission, predictions, output_path="submission_blend.csv"):
        logging.info("Сохранение submission...")
        submission['click'] = predictions
        submission.to_csv(output_path, index=False)
        print(f"{output_path} сохранён.")

    def run(self):
        train, test, submission = self.load_data()
        X_encoded, y, test_encoded = self.preprocess(train, test)
        X_res, y_res = self.balance_classes(X_encoded, y)
        self.split_data(X_res, y_res)
        self.train_model()
        self.evaluate()
        test_pred = self.predict_test(test_encoded)
        self.save_submission(submission, test_pred)

if __name__ == "__main__":
    trainer = CTRModelTrainer(
        train_path='datasearch/ctr_train.csv',
        test_path='datasearch/ctr_test.csv',
        submission_path='datasearch/ctr_sample_submission.csv'
    )
    trainer.run()
