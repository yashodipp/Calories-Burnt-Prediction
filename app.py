from __future__ import annotations

import base64
import io
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


APP_TITLE = "Calories Burnt Prediction"
APP_SUBTITLE = "AI-powered exercise energy expenditure intelligence"
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "calories.csv"
MODEL_PATH = BASE_DIR / "random_forest_regressor_model.pkl"
TARGET = "Calories"
ID_COLUMNS = ["User_ID"]
FEATURE_COLUMNS = ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
NUMERIC_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
GENDER_MAP = {"male": 0, "female": 1}
GENDER_REVERSE_MAP = {0: "male", 1: "female"}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=".",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --primary: #ff5c35;
            --secondary: #0ea5e9;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --ink: #0f172a;
            --muted: #64748b;
            --glass: rgba(255,255,255,0.72);
            --border: rgba(148,163,184,0.22);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 92, 53, 0.18), transparent 34rem),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.14), transparent 36rem),
                linear-gradient(135deg, #f8fafc 0%, #eef2ff 48%, #fff7ed 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(30,41,59,0.94));
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem;
            border: 1px solid var(--border);
            border-radius: 22px;
            background:
                linear-gradient(135deg, rgba(15,23,42,0.92), rgba(255,92,53,0.82)),
                url("https://images.unsplash.com/photo-1517836357463-d25dfeac3438?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.20);
            color: #fff;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 4rem);
            line-height: 1.02;
            letter-spacing: 0;
        }
        .hero p {
            max-width: 780px;
            font-size: 1.06rem;
            color: rgba(255,255,255,0.90);
        }
        .glass-card {
            padding: 1.15rem;
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--glass);
            backdrop-filter: blur(20px);
            box-shadow: 0 14px 36px rgba(15,23,42,0.10);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            height: 100%;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            border-color: rgba(255,92,53,0.40);
            box-shadow: 0 20px 48px rgba(15,23,42,0.15);
        }
        .metric-card {
            padding: 1rem;
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(255,255,255,0.84), rgba(255,255,255,0.48));
            border: 1px solid var(--border);
            box-shadow: 0 12px 30px rgba(15,23,42,0.08);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: var(--ink);
            margin: 1.2rem 0 0.65rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(34,197,94,0.12);
            color: #15803d;
            border: 1px solid rgba(34,197,94,0.22);
            font-weight: 700;
            font-size: 0.85rem;
        }
        .stButton > button, .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 0;
            background: linear-gradient(135deg, #ff5c35, #f97316);
            color: white;
            font-weight: 800;
            padding: 0.75rem 1rem;
            box-shadow: 0 12px 28px rgba(249,115,22,0.28);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(249,115,22,0.36);
        }
        div[data-testid="stMetric"] {
            padding: 0.85rem;
            border-radius: 14px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.64);
        }
        @media (prefers-color-scheme: dark) {
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 92, 53, 0.16), transparent 34rem),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 36rem),
                    linear-gradient(135deg, #020617 0%, #111827 52%, #1e293b 100%);
                color: #e5e7eb;
            }
            .glass-card, .metric-card, div[data-testid="stMetric"] {
                background: rgba(15,23,42,0.72);
                color: #e5e7eb;
                border-color: rgba(148,163,184,0.18);
            }
            .metric-value, .section-title {
                color: #f8fafc;
            }
            .metric-label {
                color: #94a3b8;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    expected = set(FEATURE_COLUMNS + [TARGET])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def preprocess_features(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if "Gender" in prepared.columns:
        prepared["Gender"] = (
            prepared["Gender"]
            .replace(GENDER_MAP)
            .replace(GENDER_REVERSE_MAP)
            .replace(GENDER_MAP)
            .astype(float)
        )
    return prepared[FEATURE_COLUMNS].astype(float)


def model_predict(model: Any, features: pd.DataFrame) -> np.ndarray:
    if getattr(model, "feature_names_in_", None) is None:
        return model.predict(features[FEATURE_COLUMNS].to_numpy())
    return model.predict(features[FEATURE_COLUMNS])


@st.cache_resource(show_spinner=False)
def load_model_and_metadata() -> dict[str, Any]:
    df = load_data()
    X = preprocess_features(df[FEATURE_COLUMNS])
    y = df[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_status = "Bundled model loaded"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
    except Exception as exc:
        model_status = f"Fallback model trained from CSV because pickle loading failed: {exc}"
        model = RandomForestRegressor(max_depth=10, n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

    try:
        sample_prediction = model_predict(model, X_test)
    except Exception:
        model_status = "Fallback model trained because bundled model was incompatible with the feature schema"
        model = RandomForestRegressor(max_depth=10, n_estimators=300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        sample_prediction = model_predict(model, X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, sample_prediction)))
    mae = float(mean_absolute_error(y_test, sample_prediction))
    r2 = float(r2_score(y_test, sample_prediction))
    mape = float(np.mean(np.abs((y_test - sample_prediction) / np.maximum(y_test, 1))) * 100)
    residuals = y_test.to_numpy() - sample_prediction

    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(FEATURE_COLUMNS))

    return {
        "model": model,
        "model_status": model_status,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": sample_prediction,
        "residuals": residuals,
        "metrics": {"R2 Score": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape},
        "feature_importance": pd.DataFrame(
            {"Feature": FEATURE_COLUMNS, "Importance": np.asarray(importances, dtype=float)}
        ).sort_values("Importance", ascending=False),
    }


def sidebar_navigation() -> str:
    st.sidebar.markdown("## CaloriesAI")
    st.sidebar.caption("Fitness intelligence workspace")
    return st.sidebar.radio(
        "Navigation",
        ["Home", "Prediction", "Analytics Dashboard", "Data Insights", "Model Performance", "About Project"],
        label_visibility="collapsed",
    )


def metric_card(label: str, value: str, helper: str | None = None) -> None:
    helper_html = f"<div style='color:#64748b;font-size:0.82rem;margin-top:0.35rem'>{helper}</div>" if helper else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {helper_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str) -> None:
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def make_gauge(value: float, title: str, max_value: float, color: str = "#ff5c35") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " kcal", "font": {"size": 30}},
            title={"text": title, "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, max_value]},
                "bar": {"color": color},
                "bgcolor": "rgba(255,255,255,0.35)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_value * 0.35], "color": "rgba(34,197,94,0.20)"},
                    {"range": [max_value * 0.35, max_value * 0.70], "color": "rgba(245,158,11,0.20)"},
                    {"range": [max_value * 0.70, max_value], "color": "rgba(239,68,68,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=15, r=15, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def prediction_confidence(input_row: pd.DataFrame, metadata: dict[str, Any], prediction: float) -> tuple[float, float]:
    X_train = metadata["X_train"]
    normalized_distance = []
    for col in NUMERIC_FEATURES:
        std = float(X_train[col].std()) or 1.0
        normalized_distance.append(abs(float(input_row[col].iloc[0]) - float(X_train[col].mean())) / std)
    profile_score = max(0.0, 1.0 - min(np.mean(normalized_distance) / 3.0, 1.0))

    model = metadata["model"]
    tree_std = 0.0
    if hasattr(model, "estimators_"):
        tree_predictions = np.array([tree.predict(input_row[FEATURE_COLUMNS].to_numpy())[0] for tree in model.estimators_])
        tree_std = float(np.std(tree_predictions))
    stability = max(0.0, 1.0 - min(tree_std / max(prediction, 1.0), 1.0))
    confidence = float(np.clip((profile_score * 0.55 + stability * 0.45) * 100, 50, 98))
    return confidence, tree_std


def recommendation_text(prediction: float, inputs: dict[str, Any], confidence: float) -> list[str]:
    recs = []
    duration = float(inputs["Duration"])
    heart_rate = float(inputs["Heart_Rate"])
    body_temp = float(inputs["Body_Temp"])
    if prediction >= 180:
        recs.append("High burn session detected. Prioritize hydration, cooldown, and recovery tracking.")
    elif prediction >= 90:
        recs.append("Moderate burn profile. This is suitable for steady cardio or daily fitness goals.")
    else:
        recs.append("Low burn session. Increase duration or intensity gradually if the goal is higher expenditure.")
    if heart_rate > 110:
        recs.append("Heart rate is elevated. Keep the session controlled and monitor fatigue.")
    if duration < 10:
        recs.append("Duration is short. Extending the session can improve calorie expenditure reliability.")
    if body_temp >= 40.8:
        recs.append("Body temperature is high. Add cooldown time and avoid heat stress.")
    if confidence < 70:
        recs.append("Prediction confidence is moderate because the profile is less common in the training data.")
    return recs


def build_prediction_report(inputs: dict[str, Any], prediction: float, confidence: float, recs: list[str]) -> bytes:
    payload = {
        "project": APP_TITLE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prediction": {"calories_burnt": round(float(prediction), 2), "confidence_percent": round(float(confidence), 2)},
        "inputs": inputs,
        "recommendations": recs,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def home_page(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <span class="status-pill">Production ML Dashboard</span>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}. Estimate calories burned from biometric and workout signals using a Random Forest regression model trained on exercise performance data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Dataset Rows", f"{len(df):,}", "Exercise sessions")
    with c2:
        metric_card("Avg Calories", f"{df[TARGET].mean():.1f}", "kcal per session")
    with c3:
        metric_card("Model R2", f"{metadata['metrics']['R2 Score']:.3f}", "Holdout validation")
    with c4:
        metric_card("RMSE", f"{metadata['metrics']['RMSE']:.2f}", "Prediction error")

    section_title("Platform Modules")
    cards = st.columns(3)
    modules = [
        ("Prediction Studio", "Interactive calorie estimation with confidence scoring and downloadable reports."),
        ("Fitness Analytics", "Workout distribution, biometric trends, category analysis, and correlation views."),
        ("Model Governance", "Regression metrics, residual diagnostics, feature importance, and model status."),
    ]
    for col, (title, body) in zip(cards, modules):
        with col:
            st.markdown(f"<div class='glass-card'><h3>{title}</h3><p>{body}</p></div>", unsafe_allow_html=True)

    section_title("Technology Stack")
    st.markdown(
        """
        <div class="glass-card">
            <b>Streamlit</b> dashboard interface | <b>Plotly</b> analytics | <b>scikit-learn</b> Random Forest Regressor |
            <b>pandas</b> data processing | <b>joblib</b> model loading | <b>JSON</b> prediction reports
        </div>
        """,
        unsafe_allow_html=True,
    )


def prediction_page(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    section_title("Prediction Studio")
    st.caption("Enter exercise and biometric details to estimate calories burned.")

    defaults = df[FEATURE_COLUMNS].copy()
    left, right = st.columns([1.05, 1])
    with left:
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["male", "female"], index=0)
                age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
                height = st.slider("Height (cm)", float(df["Height"].min()), float(df["Height"].max()), float(df["Height"].median()), step=1.0)
                weight = st.slider("Weight (kg)", float(df["Weight"].min()), float(df["Weight"].max()), float(df["Weight"].median()), step=1.0)
            with c2:
                duration = st.slider("Duration (minutes)", float(df["Duration"].min()), float(df["Duration"].max()), float(df["Duration"].median()), step=1.0)
                heart_rate = st.slider("Heart Rate (bpm)", float(df["Heart_Rate"].min()), float(df["Heart_Rate"].max()), float(df["Heart_Rate"].median()), step=1.0)
                body_temp = st.slider("Body Temperature (C)", float(df["Body_Temp"].min()), float(df["Body_Temp"].max()), float(df["Body_Temp"].median()), step=0.1)

            b1, b2 = st.columns(2)
            predict_clicked = b1.form_submit_button("Predict Calories")
            reset_clicked = b2.form_submit_button("Reset")

    input_payload = {
        "Gender": gender,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
    }
    input_df = pd.DataFrame([input_payload])
    model_input = preprocess_features(input_df)

    if reset_clicked:
        st.session_state.pop("last_prediction", None)
        st.rerun()

    if predict_clicked or "last_prediction" in st.session_state:
        if predict_clicked:
            prediction = float(model_predict(metadata["model"], model_input)[0])
            confidence, uncertainty = prediction_confidence(model_input, metadata, prediction)
            recs = recommendation_text(prediction, input_payload, confidence)
            st.session_state["last_prediction"] = {
                "prediction": prediction,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "inputs": input_payload,
                "recommendations": recs,
            }
        result = st.session_state["last_prediction"]
        prediction = result["prediction"]
        confidence = result["confidence"]
        uncertainty = result["uncertainty"]
        recs = result["recommendations"]

        with right:
            st.plotly_chart(make_gauge(prediction, "Predicted Calories Burned", max(float(df[TARGET].max()) * 1.1, 320)), use_container_width=True)
            st.metric("Confidence Score", f"{confidence:.1f}%", f"Ensemble spread: {uncertainty:.2f} kcal")
            st.progress(min(confidence / 100, 1.0))

        p1, p2 = st.columns([0.95, 1.05])
        with p1:
            fig = go.Figure(
                go.Indicator(
                    mode="number+gauge",
                    value=confidence,
                    number={"suffix": "%"},
                    title={"text": "Prediction Confidence"},
                    gauge={"shape": "bullet", "axis": {"range": [0, 100]}, "bar": {"color": "#0ea5e9"}},
                )
            )
            fig.update_layout(height=170, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with p2:
            st.markdown("<div class='glass-card'><h3>Recommendation System</h3>", unsafe_allow_html=True)
            for rec in recs:
                st.markdown(f"- {rec}")
            st.markdown("</div>", unsafe_allow_html=True)

        report = build_prediction_report(result["inputs"], prediction, confidence, recs)
        st.download_button(
            "Download Prediction Report",
            data=report,
            file_name="calories_burnt_prediction_report.json",
            mime="application/json",
        )
    else:
        with right:
            st.info("Submit the form to generate a prediction, confidence score, and recommendation report.")
            st.dataframe(defaults.describe().T, use_container_width=True)


def analytics_dashboard(df: pd.DataFrame) -> None:
    section_title("Analytics Dashboard")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Peak Calories", f"{df[TARGET].max():.0f} kcal")
    k2.metric("Median Duration", f"{df['Duration'].median():.0f} min")
    k3.metric("Avg Heart Rate", f"{df['Heart_Rate'].mean():.1f} bpm")
    k4.metric("Avg Body Temp", f"{df['Body_Temp'].mean():.1f} C")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x=TARGET, nbins=35, marginal="box", title="Calories Distribution", color_discrete_sequence=["#ff5c35"])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            df,
            x="Duration",
            y=TARGET,
            color="Heart_Rate",
            size="Weight",
            hover_data=["Gender", "Age", "Body_Temp"],
            color_continuous_scale="Turbo",
            title="Duration vs Calories",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        gender_summary = df.groupby("Gender", as_index=False)[TARGET].mean()
        fig = px.bar(gender_summary, x="Gender", y=TARGET, color="Gender", title="Average Calories by Gender", color_discrete_sequence=["#0ea5e9", "#ff5c35"])
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.pie(df, names="Gender", hole=0.55, title="Gender Mix", color_discrete_sequence=["#0ea5e9", "#ff5c35"])
        st.plotly_chart(fig, use_container_width=True)

    corr_df = df.drop(columns=ID_COLUMNS, errors="ignore").copy()
    corr_df["Gender"] = corr_df["Gender"].map(GENDER_MAP)
    fig = px.imshow(
        corr_df.corr(numeric_only=True),
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
    st.plotly_chart(fig, use_container_width=True)

    df_line = df.sort_values("Duration").groupby("Duration", as_index=False)[TARGET].mean()
    fig = px.line(df_line, x="Duration", y=TARGET, markers=True, title="Average Calories Trend by Exercise Duration")
    fig.update_traces(line_color="#ff5c35")
    st.plotly_chart(fig, use_container_width=True)


def data_insights(df: pd.DataFrame) -> None:
    section_title("Data Insights")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Duplicate Records", f"{int(df.duplicated().sum()):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Schema", "Statistical Summary", "Outliers"])
    with tab1:
        st.dataframe(df.head(50), use_container_width=True)
    with tab2:
        schema = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes.astype(str), "Missing": df.isna().sum().values})
        st.dataframe(schema, use_container_width=True)
    with tab3:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with tab4:
        numeric = df.select_dtypes(include=np.number).drop(columns=ID_COLUMNS, errors="ignore")
        outlier_rows = []
        for col in numeric.columns:
            q1 = numeric[col].quantile(0.25)
            q3 = numeric[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((numeric[col] < lower) | (numeric[col] > upper)).sum())
            outlier_rows.append({"Feature": col, "Lower Bound": lower, "Upper Bound": upper, "Outliers": count})
        outliers = pd.DataFrame(outlier_rows)
        st.dataframe(outliers, use_container_width=True)
        fig = px.bar(outliers, x="Feature", y="Outliers", title="IQR Outlier Count by Feature", color="Outliers", color_continuous_scale="Oranges")
        st.plotly_chart(fig, use_container_width=True)


def model_performance(metadata: dict[str, Any]) -> None:
    section_title("Model Performance")
    st.markdown(f"<span class='status-pill'>{metadata['model_status']}</span>", unsafe_allow_html=True)
    st.write("")

    metrics = metadata["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2 Score", f"{metrics['R2 Score']:.4f}")
    c2.metric("MAE", f"{metrics['MAE']:.2f}")
    c3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    c4.metric("MAPE", f"{metrics['MAPE']:.2f}%")

    st.info("This is a regression project. Accuracy, precision, recall, F1 score, ROC curve, and confusion matrix are classification metrics, so the dashboard displays regression equivalents and residual diagnostics.")

    y_test = metadata["y_test"]
    y_pred = metadata["y_pred"]
    residuals = metadata["residuals"]

    c5, c6 = st.columns(2)
    with c5:
        fig = px.scatter(x=y_test, y=y_pred, labels={"x": "Actual Calories", "y": "Predicted Calories"}, title="Actual vs Predicted Calories")
        min_v = min(float(np.min(y_test)), float(np.min(y_pred)))
        max_v = max(float(np.max(y_test)), float(np.max(y_pred)))
        fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", name="Ideal", line=dict(color="#ef4444", dash="dash")))
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        fig = px.histogram(x=residuals, nbins=35, title="Residual Distribution", labels={"x": "Residual"})
        fig.update_traces(marker_color="#0ea5e9")
        st.plotly_chart(fig, use_container_width=True)

    importance = metadata["feature_importance"]
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Feature Importance", color="Importance", color_continuous_scale="Oranges")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def about_project(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    section_title("About Project")
    st.markdown(
        """
        <div class="glass-card">
            <h3>Calories Burnt Prediction</h3>
            <p>This application estimates calories burned during exercise using user biometrics, activity duration, heart rate, and body temperature.
            The model is designed for wellness analytics, fitness dashboards, coaching tools, and exercise planning workflows.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="glass-card">
                <h4>Prediction Target</h4>
                <p><b>Calories</b> burned during an exercise session.</p>
                <h4>Algorithm</h4>
                <p>Random Forest Regressor with model fallback training from the bundled CSV.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="glass-card">
                <h4>Input Features</h4>
                <p>Gender, Age, Height, Weight, Duration, Heart Rate, Body Temperature.</p>
                <h4>Dataset</h4>
                <p>15,000 exercise records with no missing values in the bundled file.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.dataframe(
        pd.DataFrame(
            {
                "Artifact": ["app.py", "requirements.txt", "README.md", "RandomForest.ipynb", "random_forest_regressor_model.pkl", "calories.csv"],
                "Status": ["Generated", "Generated", "Generated", "Bundled", "Bundled", "Bundled"],
            }
        ),
        use_container_width=True,
    )


def main() -> None:
    inject_css()
    try:
        df = load_data()
        metadata = load_model_and_metadata()
    except Exception as exc:
        st.error(f"Application startup failed: {exc}")
        st.stop()

    page = sidebar_navigation()
    if page == "Home":
        home_page(df, metadata)
    elif page == "Prediction":
        prediction_page(df, metadata)
    elif page == "Analytics Dashboard":
        analytics_dashboard(df)
    elif page == "Data Insights":
        data_insights(df)
    elif page == "Model Performance":
        model_performance(metadata)
    elif page == "About Project":
        about_project(df, metadata)


if __name__ == "__main__":
    main()
