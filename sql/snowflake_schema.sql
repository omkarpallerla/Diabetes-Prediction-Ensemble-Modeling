-- ============================================================
-- Snowflake Schema: Diabetes Risk Scoring Pipeline
-- Author: Omkar Pallerla | MS Business Analytics, ASU
-- ============================================================

-- Raw EHR staging layer
CREATE OR REPLACE TABLE diabetes_raw.ehr_records (
    patient_id        VARCHAR(20) PRIMARY KEY,
    pregnancies       INT,
    glucose           FLOAT,
    blood_pressure    FLOAT,
    skin_thickness    FLOAT,
    insulin           FLOAT,
    bmi               FLOAT,
    diabetes_pedigree FLOAT,
    age               INT,
    outcome           INT,  -- 1 = diabetic, 0 = non-diabetic
    loaded_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Cleaned staging table (zero imputation applied)
CREATE OR REPLACE TABLE diabetes_staging.stg_patient_vitals AS
SELECT
    patient_id,
    pregnancies,
    -- Replace physiologically impossible zeros with column median
    CASE WHEN glucose = 0 THEN MEDIAN(glucose) OVER () ELSE glucose END AS glucose_clean,
    CASE WHEN blood_pressure = 0 THEN MEDIAN(blood_pressure) OVER () ELSE blood_pressure END AS blood_pressure_clean,
    CASE WHEN skin_thickness = 0 THEN MEDIAN(skin_thickness) OVER () ELSE skin_thickness END AS skin_thickness_clean,
    CASE WHEN insulin = 0 THEN MEDIAN(insulin) OVER () ELSE insulin END AS insulin_clean,
    CASE WHEN bmi = 0 THEN MEDIAN(bmi) OVER () ELSE bmi END AS bmi_clean,
    diabetes_pedigree,
    age,
    outcome,
    loaded_at
FROM diabetes_raw.ehr_records;

-- ML scores mart (populated by Python scoring job)
CREATE OR REPLACE TABLE diabetes_mart.patient_risk_scores (
    patient_id        VARCHAR(20),
    glucose_clean     FLOAT,
    bmi_clean         FLOAT,
    age               INT,
    risk_score        FLOAT,        -- XGBoost probability output
    risk_tier         VARCHAR(10),  -- Low / Medium / High
    actual_outcome    INT,
    model_version     VARCHAR(20),
    scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Power BI summary view — aggregated risk metrics
CREATE OR REPLACE VIEW diabetes_mart.vw_risk_dashboard AS
SELECT
    risk_tier,
    COUNT(*) AS patient_count,
    ROUND(AVG(risk_score), 3) AS avg_risk_score,
    ROUND(AVG(glucose_clean), 1) AS avg_glucose,
    ROUND(AVG(bmi_clean), 1) AS avg_bmi,
    ROUND(AVG(age), 1) AS avg_age,
    SUM(actual_outcome) AS confirmed_diabetic,
    ROUND(SUM(actual_outcome) / COUNT(*), 3) AS actual_positive_rate
FROM diabetes_mart.patient_risk_scores
GROUP BY risk_tier
ORDER BY CASE risk_tier WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END;

-- Monthly scoring trend (for Power BI time series)
CREATE OR REPLACE VIEW diabetes_mart.vw_scoring_trend AS
SELECT
    DATE_TRUNC('month', scored_at) AS score_month,
    risk_tier,
    COUNT(*) AS patients_scored,
    ROUND(AVG(risk_score), 3) AS avg_risk_score
FROM diabetes_mart.patient_risk_scores
GROUP BY 1, 2
ORDER BY 1, 2;
