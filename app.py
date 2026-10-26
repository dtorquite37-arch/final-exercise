import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path
from datetime import datetime

from modules.data_processing import make_sample_data, validate_and_clean
from modules.exploratory_analysis import show_eda
from modules.prediction import predict_one, MODEL_FEATURES, TARGET
from modules.dashboard import show_dashboard

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.joblib"

st.set_page_config(page_title="Campus Facility Reservation Monitor", page_icon="🏫", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
st.sidebar.title("🏫 Facility Reservation System")
menu = st.sidebar.selectbox("Navigate to", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Midterm model: Decision Tree")
st.sidebar.caption("Model file: models/trained_model.joblib")

if "active_df" not in st.session_state:
    st.session_state.active_df = make_sample_data()
if "data_source" not in st.session_state:
    st.session_state.data_source = "Built-in sample dataset"

df = st.session_state.active_df

if menu == "Home":
    st.title("🏫 Campus Facility Reservation Monitoring")
    st.subheader("Cancellation / No-Show Decision Support")
    st.write("This Streamlit app adapts the midterm notebook's campus facility reservation analysis and Decision Tree classifier.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reservations in current data", len(df))
    c2.metric("Facility types", df["Facility_Type"].nunique() if "Facility_Type" in df else 0)
    c3.metric("Data source", st.session_state.data_source)
    st.info("Predictions support an administrator's review. They must not automatically approve or reject a reservation.")
    st.markdown("### What you can do")
    st.markdown("- **Dataset:** upload a CSV or use the built-in sample, check columns, missing values, and duplicates.")
    st.markdown("- **Data Analysis:** review statistics and interactive charts.")
    st.markdown("- **Prediction:** estimate Completed vs Not Completed using the saved Decision Tree pipeline.")
    st.markdown("- **Dashboard:** review reservation KPIs and model-generated risk ratings.")
    st.markdown("- **System Design:** view functional requirements, non-functional requirements, actors, and workflow.")
    st.caption("Important limitation: the notebook's built-in sample is small and contains few non-completed examples. Treat outputs as educational demonstrations, not validated operational forecasts.")

elif menu == "Dataset":
    st.title("📂 Dataset Upload and Validation")
    st.write("Upload a CSV with the midterm reservation fields. Or use the built-in sample dataset.")
    uploaded = st.file_uploader("Upload reservation dataset (CSV)", type=["csv"])
    col1, col2 = st.columns(2)
    if col1.button("Use built-in sample dataset", use_container_width=True):
        st.session_state.active_df = make_sample_data()
        st.session_state.data_source = "Built-in sample dataset"
        st.rerun()
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            cleaned, issues = validate_and_clean(raw)
            if issues:
                st.error("Dataset validation failed:")
                for issue in issues:
                    st.write(f"- {issue}")
            elif cleaned.empty:
                st.error("The uploaded dataset has no usable rows.")
            else:
                st.session_state.active_df = cleaned
                st.session_state.data_source = uploaded.name
                st.success(f"Loaded {len(cleaned)} rows from {uploaded.name}.")
                df = cleaned
        except Exception as exc:
            st.error(f"Could not read this CSV. Check that it is a valid CSV file. Details: {exc}")
    df = st.session_state.active_df
    st.markdown("### Data preview")
    st.dataframe(df.head(20), use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Rows", len(df))
    b.metric("Columns", len(df.columns))
    c.metric("Exact duplicate rows", int(df.duplicated().sum()))
    st.markdown("### Missing values by column")
    st.dataframe(df.isna().sum().rename("Missing values").to_frame(), use_container_width=True)
    st.markdown("### Required fields")
    st.write(", ".join(["Facility_Type", "User_Category", "Lead_Time_Days", "Participants", "Usage_Status"]))
    st.caption("For predictions, the app uses Facility_Type, User_Category, Lead_Time_Days, and Participants. Usage_Status is the observed outcome label, not a prediction input.")

elif menu == "Data Analysis":
    show_eda(df)

elif menu == "Prediction":
    st.title("🔮 Reservation Outcome Prediction")
    st.write("Enter reservation details available at booking time. Do not enter actual attendance or the observed usage status.")
    facility_options = sorted(df["Facility_Type"].dropna().astype(str).unique().tolist()) or ["Computer Lab", "Gymnasium", "Library", "Auditorium"]
    category_options = sorted(df["User_Category"].dropna().astype(str).unique().tolist()) or ["Student", "Faculty", "Organization"]
    with st.form("prediction_form"):
        facility = st.selectbox("Facility type", facility_options)
        category = st.selectbox("User category", category_options)
        lead = st.number_input("Lead time (days)", min_value=0, max_value=365, value=7, step=1)
        participants = st.number_input("Number of participants", min_value=1, max_value=10000, value=30, step=1)
        submitted = st.form_submit_button("Predict outcome")
    if submitted:
        try:
            result, probability = predict_one(model, facility, category, int(lead), int(participants))
            st.subheader(f"Predicted outcome: {result}")
            if probability is not None:
                risk = probability * 100
                rating = "HIGH" if risk >= 70 else ("MEDIUM" if risk >= 40 else "LOW")
                c1, c2 = st.columns(2)
                c1.metric("Estimated Not Completed score", f"{risk:.1f}%")
                c2.metric("Risk rating", rating)
                st.progress(min(max(risk / 100, 0.0), 1.0))
                st.caption("This score is a model output and may not be a calibrated real-world probability.")
            st.write(f"Prediction timestamp: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}")
            st.warning("Human review required. Do not automatically reject or deny facility access based only on this prediction.")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
    st.markdown("### Model evaluation from midterm split")
    metrics_path = BASE_DIR / "models" / "model_metrics.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        m2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        m4.metric("F1-score", f"{metrics.get('f1_score', 0):.3f}")
    else:
        st.info("Evaluation metrics file is not available.")

elif menu == "Dashboard":
    show_dashboard(df, model)

elif menu == "System Design":
    st.title("🧩 System Design and SAD Integration")
    st.markdown("### Actors")
    st.markdown("- **Facility Administrator:** reviews utilization, risk ratings, and reservations; records the final decision.")
    st.markdown("- **Reservation User (Student/Faculty/Organization):** submits reservation details and receives the system's decision through the existing reservation process.")
    st.markdown("### Functional requirements")
    requirements = [
        ("FR-DM-01", "The system shall summarize reservations by facility type."),
        ("FR-DM-02", "The system shall show cancellation/no-show counts and percentages by user category."),
        ("FR-DM-03", "The system shall predict the reservation outcome using the saved Decision Tree pipeline."),
        ("FR-DM-04", "The system shall validate uploaded CSV columns and display missing-value and duplicate summaries."),
        ("FR-DM-05", "The system shall show risk indicators and recommendations for administrator review.")
    ]
    for code, desc in requirements:
        st.write(f"**{code}:** {desc}")
    st.markdown("### Non-functional requirements")
    st.markdown("- **Usability:** navigation and charts should be understandable to facility administrators.")
    st.markdown("- **Reliability:** invalid files and missing required fields should produce clear messages.")
    st.markdown("- **Privacy and security:** use synthetic or authorized data; do not upload sensitive student records publicly.")
    st.markdown("### Analytical decision workflow")
    st.code("Reservation data → Validate and clean → Explore patterns → Saved Decision Tree predicts outcome → Show risk/review queue → Administrator reviews details → Authorized human decision", language="text")
    diagram = BASE_DIR / "assets" / "use_case_diagram.png"
    if diagram.exists():
        st.image(str(diagram), caption="Use case diagram from the midterm notebook")
    else:
        st.info("Use case diagram image can be added to assets/use_case_diagram.png before deployment.")
    st.markdown("### Model limitations")
    st.write("The original midterm dataset is a small synthetic practice dataset. Performance may be unstable and should not be generalized to real campus reservations without more representative data and validation.")
