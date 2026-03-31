# =============================================================
# Diabetes Prediction — Ensemble Learning Benchmark
# Author: Omkar Pallerla | MS Business Analytics, ASU
# Snowflake + dbt integration layer included
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               BaggingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
COLORS = ['#4f9cf9','#06d6a0','#7c3aed','#f59e0b','#ef4444','#ec4899','#14b8a6','#f97316','#84cc16','#a855f7']

# ── 1. LOAD DATA ────────────────────────────────────────────
df = pd.read_csv('diabetes.csv')
print(f"Shape: {df.shape}")
print(f"Target balance:\n{df['Outcome'].value_counts()}")

# ── 2. DATA CLEANING — fix physiologically impossible zeros ─
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].median(), inplace=True)

# Feature engineering
df['BMI_Age']        = df['BMI'] * df['Age']
df['Glucose_Insulin'] = df['Glucose'] / (df['Insulin'] + 1)
df['Risk_Score']     = (df['Glucose'] / 100) + (df['BMI'] / 25) + (df['Age'] / 50)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                      stratify=y, random_state=42)

# ── 3. DEFINE 10+ MODELS ────────────────────────────────────
models = {
    'XGBoost':            XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                         subsample=0.8, random_state=42, eval_metric='logloss'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting':  GradientBoostingClassifier(n_estimators=150, random_state=42),
    'Bagging':            BaggingClassifier(n_estimators=100, random_state=42),
    'Voting Ensemble':    VotingClassifier(estimators=[
                              ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                              ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
                              ('lr', LogisticRegression(max_iter=1000))
                          ], voting='soft'),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM':                SVC(probability=True, kernel='rbf', random_state=42),
    'KNN':                KNeighborsClassifier(n_neighbors=7),
    'Naive Bayes':        GaussianNB(),
    'MLP Neural Net':     MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
}

# ── 4. TRAIN & BENCHMARK ────────────────────────────────────
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc  = roc_auc_score(y_test, y_prob)
    acc  = (y_pred == y_test).mean()
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='roc_auc').mean()
    results[name] = {'model': model, 'auc': auc, 'acc': acc, 'cv_auc': cv,
                     'preds': y_pred, 'probs': y_prob}
    print(f"{name:22s} AUC={auc:.3f}  Acc={acc:.3f}  CV_AUC={cv:.3f}")

# ── 5. EXPORT RISK SCORES (Snowflake → Power BI) ────────────
best_model = results['XGBoost']['model']
probs = best_model.predict_proba(X_test)[:, 1]
score_df = X_test.copy()
score_df['patient_id']   = range(len(score_df))
score_df['actual']       = y_test.values
score_df['risk_score']   = probs
score_df['risk_tier']    = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                    labels=['Low', 'Medium', 'High'])
score_df.to_csv('outputs/patient_risk_scores.csv', index=False)
print("\nExported: outputs/patient_risk_scores.csv")

# ── 6. DBT MODEL STUB ───────────────────────────────────────
dbt_model = """-- models/marts/patient_risk_scores.sql
-- Runs daily in Snowflake via dbt
SELECT
    patient_id,
    glucose,
    bmi,
    age,
    risk_score,
    risk_tier,
    CURRENT_TIMESTAMP() AS scored_at
FROM {{ ref('stg_patient_vitals') }}
LEFT JOIN {{ ref('ml_scores') }} USING (patient_id)
"""
with open('dbt_models/patient_risk_scores.sql', 'w') as f:
    f.write(dbt_model)

# ── 7. VISUALIZATIONS ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')

# AUC comparison
ax = axes[0, 0]
sorted_r = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
names_s  = [r[0] for r in sorted_r]
aucs_s   = [r[1]['auc'] for r in sorted_r]
colors_bar = ['#06d6a0' if i == 0 else '#4f9cf9' for i in range(len(names_s))]
ax.barh(names_s[::-1], aucs_s[::-1], color=colors_bar[::-1])
ax.set_xlim(0.5, 1.0)
ax.set_title('AUC-ROC Benchmark — 10 Models', color='white', pad=12)
ax.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='0.80 threshold')

# ROC Curves — top 4
ax = axes[0, 1]
top4 = [r[0] for r in sorted_r[:4]]
for i, name in enumerate(top4):
    fpr, tpr, _ = roc_curve(y_test, results[name]['probs'])
    ax.plot(fpr, tpr, color=COLORS[i], label=f"{name} (AUC={results[name]['auc']:.2f})", lw=2)
ax.plot([0,1],[0,1],'--', color='gray', alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Top 4 Models', color='white', pad=12)
ax.legend(fontsize=8)

# Feature importance (XGBoost)
ax = axes[1, 0]
xgb = results['XGBoost']['model']
feat_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color='#7c3aed')
ax.set_title('Feature Importance — XGBoost', color='white', pad=12)

# Risk tier distribution
ax = axes[1, 1]
tier_c = score_df['risk_tier'].value_counts()
colors_t = {'Low': '#06d6a0', 'Medium': '#f59e0b', 'High': '#ef4444'}
ax.pie(tier_c, labels=tier_c.index,
       colors=[colors_t.get(str(t), '#4f9cf9') for t in tier_c.index],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Patient Risk Tier Distribution', color='white', pad=12)

plt.tight_layout()
plt.savefig('outputs/diabetes_benchmark.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Saved: outputs/diabetes_benchmark.png")
plt.show()
