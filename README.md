# NovaMind-AI-Churn-Prediction-Model

# NovaMind AI — Churn EDA, Data Quality Checks, and Random Forest Model (Databricks)

This project analyzes NovaMind AI customer data in **Databricks** to (1) validate data quality, (2) perform **SQL-based exploratory data analysis (EDA)** on churn behavior, and (3) train a **Python (scikit-learn) churn prediction model** that outputs churn likelihood for current users.

> Primary notebook: `novamind_ai_DQ_EDA_Model.ipynb`

---

## What this notebook does

### 1) Data Quality (DQ) checks (SQL)
Runs validation queries against the historical table to ensure the dataset is usable for analysis/modeling, including:

- **Null checks** across key columns (e.g., `user_id`, region/country, plan, usage, payments, support, activity, churn label).
- **Uniqueness check** for `user_id` (detect duplicates).
- **Domain/value checks** (e.g., expected `subscription_plan` values).
- **Freshness checks** using `last_active_days_ago` (min/max/avg inactivity).

### 2) SQL EDA on churn drivers
Explores churn distribution and key slices to understand where churn concentrates:

- Overall **churn rate** (`churned` distribution).
- **Churn rate by region**.
- **Churn rate by subscription plan**.
- **Churn rate by renewal behavior**, plus renewal behavior × plan.
- **Churn rate by channel** (`api_or_app`), plus api/app × plan.
- Identifies **high-risk segments** with simple rules (e.g., failed payments + inactivity bands).
- Additional country-level churn breakdowns (including a dedicated look at Africa in one section).

### 3) Modeling churn (Python / scikit-learn)
Builds a baseline production-style model using historical data:

- Loads historical data from Unity Catalog via Spark:
  - `novamind_ai.default.novamind_ai_historical`
- Converts Spark DataFrame → **Pandas** for scikit-learn.
- Encodes categorical features using **LabelEncoder** (adds `*_encoded` columns):
  - `subscription_plan`, `api_or_app`, `renewal_behavior`, `region`
- Uses a **stratified train/test split** to preserve churn class ratio.
- Trains a **RandomForestClassifier** with:
  - `class_weight="balanced"` (handles churn imbalance)
  - `n_estimators=100`, `max_depth=10`, `random_state=42`
- Evaluates using:
  - Accuracy
  - ROC-AUC
  - Classification report (precision/recall/F1)
  - Confusion matrix
- Prints **feature importances** to highlight strongest churn predictors in this dataset.

### 4) Scoring “current users” (inference)
Scores active users with unknown outcomes to produce churn probability estimates:

- Loads current users table:
  - `novamind_ai.default.novamind_ai_current_users`
- Applies the same feature set and model to generate churn risk scores.

---

## Data & key fields

The notebook expects a historical dataset containing (at minimum) fields like:

- Identifiers/demographics: `user_id`, `region`, `country`
- Plan & lifecycle: `subscription_plan`, `months_subscribed`, `renewal_behavior`
- Engagement: `usage_frequency`, `feature_adoption`, `prompt_volume`, `api_or_app`
- Risk signals: `failed_payments`, `support_tickets`, `last_active_days_ago`
- Label (historical only): `churned` (0/1)

---

## How to run

### Option A — Run in Databricks (recommended)
1. Import `novamind_ai_DQ_EDA_Model.ipynb` into your Databricks workspace.
2. Attach the notebook to a cluster with:
   - Spark access to Unity Catalog
   - Python environment with `scikit-learn` and `pandas`
3. Ensure these tables exist and are readable:
   - `novamind_ai.default.novamind_ai_historical`
   - `novamind_ai.default.novamind_ai_current_users`
4. Run cells top-to-bottom.

### Option B — Run locally (not the primary target)
This notebook mixes `%sql` and Databricks/Spark calls (`spark.table(...)`), so local execution will require:
- A Spark environment configured similarly, or
- Refactoring `%sql` cells and replacing Unity Catalog reads with CSV/parquet inputs.

---

## Notes / best-practice considerations

- **Categorical encoding:** the notebook uses `LabelEncoder` for multiple columns. For strict ML best practices, consider `OneHotEncoder` (or target encoding) and a `Pipeline`/`ColumnTransformer` to prevent training/serving skew.
- **Reproducibility:** random seeds are set for splitting and the Random Forest (`random_state=42`).
- **Imbalanced target:** churn is imbalanced; `class_weight="balanced"` is used and ROC-AUC is reported.
- **Model persistence:** if you plan to operationalize this, add model + encoder persistence (e.g., `joblib`) and a consistent preprocessing pipeline.

---

## Repository contents

- `novamind_ai_DQ_EDA_Model.ipynb` — end-to-end DQ checks, SQL EDA, churn modeling, and current-user scoring.

---

## License
Add a `LICENSE` file if you plan to reuse or distribute this project.
