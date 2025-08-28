import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ydata_profiling import ProfileReport
import shap
import matplotlib.pyplot as plt
from time import time

class SimpleDatasetAnalyzer:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        self.scaler = StandardScaler()
        self.categorical_cols = ['site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 
                                'app_category', 'device_id', 'device_ip', 'device_model', 'device_conn_type']
        self.numeric_cols = ['C1', 'banner_pos', 'hour', 'device_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']

    def load_data(self, file_path, nrows=None):
        start_time = time()
        self.df = pd.read_csv(file_path, nrows=nrows)
        print(f"Данные загружены. Размер: {self.df.shape}, Время: {time() - start_time:.2f} сек")

    def preprocess(self):
        start_time = time()
        self.df.replace(-1, pd.NA, inplace=True)
        for col in self.categorical_cols:
            if col in self.df.columns:
                self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))
        for col in self.numeric_cols:
            if col in self.df.columns and self.df[col].isna().any():
                self.df[col] = self.df[col].fillna(self.df[col].median())
        self.X = self.df.drop(['idx', 'id', 'click'], axis=1, errors='ignore')
        self.y = self.df['click']
        self.X[self.numeric_cols] = self.scaler.fit_transform(self.X[self.numeric_cols].astype(float))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        print(f"Предобработка завершена. Время: {time() - start_time:.2f} сек")

    def train_model(self):
        start_time = time()
        self.model = LogisticRegression(solver='saga', max_iter=100, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        print(f"Модель обучена. Время: {time() - start_time:.2f} сек")

    def generate_eda_report(self, output_file):
        start_time = time()
        profile = ProfileReport(self.df, minimal=True)
        profile.to_file(output_file)
        print(f"EDA-отчёт сохранён в {output_file}. Время: {time() - start_time:.2f} сек")

    def explain_with_shap(self, sample_size=100, label=""):
        start_time = time()
        X_sample = self.X_test.sample(min(sample_size, len(self.X_test)), random_state=42).astype(float)
        explainer = shap.LinearExplainer(self.model, self.X_train)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False, max_display=10)
        plt.savefig(os.path.join("graphs", f"shap_summary_{label}.png"))
        plt.close()
        print(f"SHAP-анализ сохранён в graphs/shap_summary_{label}.png. Время: {time() - start_time:.2f} сек")

    def analyze_subsamples(self, file_path, sample_sizes=[3000000, 6000000, 10000000, 20000000, 30000000, None]):
        if not os.path.exists("graphs"):
            os.makedirs("graphs")
        full_df = pd.read_csv(file_path)
        for size in sample_sizes:
            start_time = time()
            if size is None:
                self.df = full_df
                label = "full_40M"
            else:
                self.df = full_df.sample(n=size, random_state=42)
                label = f"{size // 1000000}M"
            print(f"\nАнализ подвыборки: {label} ({len(self.df)} строк)")
            self.preprocess()
            self.generate_eda_report(output_file=os.path.join("graphs", f"eda_report_{label}.html"))
            self.train_model()
            self.explain_with_shap(sample_size=100, label=label)
            print(f"Обработка подвыборки завершена. Общее время: {time() - start_time:.2f} сек")


if __name__ == "__main__":
    file_path = "ctr_train.csv"
    analyzer = SimpleDatasetAnalyzer()
    analyzer.analyze_subsamples(file_path)