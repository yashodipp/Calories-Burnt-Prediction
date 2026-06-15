# Calories Burnt Prediction

Production-ready Streamlit dashboard for predicting calories burned during exercise using biometric and workout signals.

## Project Files

- `app.py` - Streamlit SaaS-style dashboard
- `calories.csv` - bundled dataset
- `random_forest_regressor_model.pkl` - bundled Random Forest model
- `RandomForest.ipynb` - original notebook
- `requirements.txt` - deployment dependencies

## Features

- Home page with KPI cards and project overview
- Prediction page with dynamic inputs, confidence score, gauge, recommendations, and downloadable JSON report
- Analytics dashboard with Plotly charts
- Data insights with missing values, duplicates, schema, statistics, and outlier analysis
- Model performance page with regression metrics, residual diagnostics, and feature importance
- Dark and light theme aware premium UI

## Input Features

- Gender
- Age
- Height
- Weight
- Duration
- Heart_Rate
- Body_Temp

## Prediction Target

- Calories

## Algorithm

Random Forest Regressor

## Run Locally

```bash
cd "/Users/yashodip/Documents/New project/Calories_Burnt_Prediction"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Deploy

Upload the full folder with all bundled artifacts to Streamlit Community Cloud or any Python hosting environment, then run:

```bash
streamlit run app.py
```
