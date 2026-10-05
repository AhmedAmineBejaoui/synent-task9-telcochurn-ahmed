# Synent Task 9 – Telco Customer Churn Prediction

End-to-end Data Science project (Synent Technologies Data Science Internship).
Predict whether a telecom customer is likely to churn from customer, service, contract and billing information.

## Dataset

- **File:** `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (also at repo root for VS Code run)
- **Source:** Kaggle – Telco Customer Churn (blastchar) – https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Shape:** 7043 rows, 21 columns
- **Target:** `Churn` (Yes = churned, No = stayed) – 26.54% Yes / 73.46% No
- Columns: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn

## Workflow

1. Data collection (local CSV)
2. Data cleaning: `TotalCharges` to numeric (11 NaN filled with median), 0 duplicates, `customerID` dropped
3. EDA: churn distribution, tenure/charges, contract, internet service, correlation
4. Preprocessing: `SimpleImputer(median)+StandardScaler` for numeric, `SimpleImputer(most_frequent)+OneHotEncoder` for categoricals (ColumnTransformer Pipeline)
5. Train/test split: 80/20, `stratify=y`, `random_state=42`
6. Model: `LogisticRegression(max_iter=1000)`
7. Evaluation: Accuracy, Precision, Recall, F1, ROC AUC + Confusion Matrix + ROC Curve
8. Saving: `model/churn_model.joblib` + `model/model_metadata.json`
9. App: `app.py` (Streamlit, user input → prediction + probability)
10. Deployment: Streamlit Community Cloud from GitHub

## Results (this run)

From `outputs/model_metrics.csv`:

| Metric | Score |
|---|---|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 Score | 0.604 |
| ROC AUC | 0.8419 |

Key insights (see notebook §24):
- Stayed median tenure 38 months vs churned 10 months; stayed median MonthlyCharges 64.43 vs churned 79.65.
- Contract churn: Month-to-month 42.71%, One year 11.27%, Two year 2.83%.
- InternetService churn: Fiber optic 41.89%, DSL 18.96%, No internet 7.40%.
- AUC 0.84 = good ranking, but recall 0.56 means ~44% of real churners missed.
- Correlations are observed patterns only, not causation.

## Run locally (VS Code, Python 3.13 tested)

```powershell
pip install -r requirements.txt
# option 1: notebook
# open telco_churn_task9_synent.ipynb (root) or notebooks/telco_churn_analysis.ipynb and Run All
# option 2: app
streamlit run app.py
# open http://localhost:8501
```

Test profiles verified:
- High-risk (tenure 2, MonthlyCharges 85, Fiber optic, Month-to-month) → Churn, ~54.7%
- Low-risk (tenure 60, MonthlyCharges 45, DSL, Two year) → Stay, churn proba ~1.2%

## Project Structure

```text
synent-task9-telcochurn-ahmed/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── telco_churn_analysis.ipynb
├── telco_churn_task9_synent.ipynb
├── model/
│   ├── churn_model.joblib
│   └── model_metadata.json
├── outputs/
│   ├── model_metrics.csv
│   └── figures/ (12 PNG)
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Deployment

- GitHub: push this folder as public `synent-task9-telcochurn-ahmed`
- Streamlit Cloud: `share.streamlit.io` → Create app → repo / branch `main` / `app.py` → Deploy → live app below
- **Live App:** [synent-task9-telcochurn-ahmed.streamlit.app](https://synent-task9-telcochurn-ahmed.streamlit.app/)
- App expects `model/churn_model.joblib` relative to `app.py`.

## Demo Video & Links

- **GitHub:** [synent-task9-telcochurn-ahmed](https://github.com/AhmedAmineBejaoui/synent-task9-telcochurn-ahmed)
- **Live App:** [synent-task9-telcochurn-ahmed.streamlit.app](https://synent-task9-telcochurn-ahmed.streamlit.app/)
- **Dataset:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) + local `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Video (1–3 min):** [Google Drive — Task 9 Demo](https://drive.google.com/file/d/15RRbMFwFUu1bxcKkwbDWIZwohnK6o4uP/view?usp=sharing)
- **LinkedIn post:** [LinkedIn — Telco Churn post](https://lnkd.in/p/ekFqE_gJ)

## Author

Ahmed Amin Bejaoui – Synent Task 9
