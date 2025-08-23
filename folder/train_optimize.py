import joblib 
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import category_encoders as ce

target, models = 'click', []
train, test = pd.read_csv("ctr_train.csv"), pd.read_csv("ctr_test.csv")
features = [col for col in train.columns if col not in ['click', 'id']]
train_encoded, test_encoded, encoder = train.copy(), test.copy(), ce.TargetEncoder(cols=features)
train_encoded[features] = encoder.fit_transform(train[features], train[target])
test_encoded[features] = encoder.transform(test[features])

'''кросс-валидация'''
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof, preds = np.zeros(len(train)), np.zeros(len(test))



for fold, (tr_idx, val_idx) in enumerate(skf.split(train_encoded, train[target])):
    X_tr, X_val = train_encoded.iloc[tr_idx][features], train_encoded.iloc[val_idx][features]
    y_tr, y_val = train[target].iloc[tr_idx], train[target].iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.05,
        num_leaves=64, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    )

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    oof[val_idx] = model.predict_proba(X_val)[:, 1]
    preds += model.predict_proba(test_encoded[features])[:, 1] / skf.n_splits

    joblib.dump(model, f'lgbm_fold{fold}.pkl')
    models.append(model)

print("CV ROC-AUC:", roc_auc_score(train[target], oof))
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission['click'] = preds
sample_submission.to_csv('my_submission.csv', index=False)