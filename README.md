# 🩺 Diabetes Prediction: Ensemble Learning Benchmark

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Snowflake](https://img.shields.io/badge/Snowflake-29B5E8?style=for-the-badge&logo=snowflake&logoColor=white)
![dbt](https://img.shields.io/badge/dbt-FF694B?style=for-the-badge&logo=dbt&logoColor=white)

> **Benchmarking 10+ ML algorithms to predict diabetes risk with 82% AUC — built on a Snowflake + dbt data pipeline for production-grade deployment.**

---

## 📌 Business Overview

Diabetes affects 537 million adults globally. Early ML-driven risk scoring enables healthcare providers to prioritize at-risk patients for preventative care, reducing downstream treatment costs by up to 60%.

This project goes beyond notebook ML: the scoring pipeline sits on a **Snowflake clinical data warehouse** with a dbt staging layer handling incremental patient records, feeding a Power BI patient risk dashboard.

---

## 📊 Production Architecture

```
Raw EHR Data → Azure Data Factory → Snowflake Raw Layer
  → dbt Staging → dbt Mart: patient_risk_scores
    → Python ML Scoring → Power BI Patient Dashboard
```

---

## 🔬 Model Development

| Step | Detail |
|------|--------|
| **Data** | Pima Indians Diabetes Dataset — 768 records, 8 clinical features |
| **Cleaning** | Imputed zeros as nulls (Glucose=0 physiologically impossible) |
| **Algorithms** | XGBoost, Random Forest, Bagging, Voting Classifier, SVM, KNN, Logistic Regression, Naive Bayes, MLP |
| **Metric** | AUC-ROC — distinguishing positive/negative cases |
| **Tuning** | GridSearchCV for XGBoost hyperparameters |

---

## 📈 Key Findings

- 🏆 **Ensemble Superiority** — Bagging + Random Forest consistently outperformed single models; AUC **0.82**
- 🩸 **Top Feature: Glucose** — Single most predictive, followed by BMI and Age
- ⚖️ **Voting Classifier** — Best clinical balance between sensitivity and specificity
- 🔢 **Zero Imputation** — Replacing 0s with median for Glucose/BMI improved AUC by 0.06

---

## 🧠 dbt Model Layer

```sql
-- models/marts/patient_risk_scores.sql
SELECT
    patient_id,
    glucose_clean,
    bmi,
    age,
    risk_score,
    risk_tier,  -- Low / Medium / High
    scored_at
FROM {{ ref('stg_patient_vitals') }} v
LEFT JOIN {{ ref('ml_scores') }} m USING (patient_id)
```

---

## 🛠 Tools & Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10 |
| ML | Scikit-Learn, XGBoost, Imbalanced-learn |
| Pipeline | Snowflake, dbt, Azure Data Factory |
| Visualization | Seaborn, Matplotlib, Plotly |
| BI Output | Snowflake → Power BI Patient Dashboard |

---

## 🚀 How to Run

```bash
git clone https://github.com/omkarpallerla/Diabetes-Prediction-Ensemble-Modeling.git
cd Diabetes-Prediction-Ensemble-Modeling
pip install -r requirements.txt
jupyter notebook notebooks/Diabetes_Ensemble_Benchmark.ipynb
```

---

## 📊 Benchmark Results

| Model | AUC-ROC | Accuracy | Notes |
|-------|---------|----------|-------|
| **XGBoost (tuned)** | **0.84** | 79.2% | Best AUC |
| Bagging Classifier | 0.82 | 77.9% | Most stable |
| Voting Classifier | 0.81 | 78.5% | Best clinical balance |
| Logistic Regression | 0.76 | 74.0% | Baseline |

---

<div align="center">
  <sub>Built by <a href="https://github.com/omkarpallerla">Omkar Pallerla</a> · MS Business Analytics, ASU · BI Engineer · Snowflake | dbt | Azure | Databricks Certified</sub>
</div>