from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from custom_transformers import (
    AgeImputer,
    CabinImputer,
    EmbarkedImputer,
    LogFareTransformer,
    TicketImputer,
    UniversalBackupImputer,
)

st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Enter passenger details below to predict whether they would have survived the Titanic disaster."
)


@st.cache_resource
def load_pipeline():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "model" / "titanic_pipeline.joblib"
    return joblib.load(model_path)


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        index=2,
        help="1 = 1st Class (Upper), 2 = 2nd Class (Middle), 3 = 3rd Class (Lower)",
    )
    name = st.text_input("Full Name", value="Braund, Mr. Owen Harris")
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.number_input(
        "Age", min_value=0.0, max_value=100.0, value=22.0, step=0.5
    )
    sibsp = st.number_input(
        "Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=10, value=1
    )

with col2:
    parch = st.number_input(
        "Parents / Children Aboard (Parch)", min_value=0, max_value=10, value=0
    )
    ticket = st.text_input("Ticket Number", value="A/5 21171")
    fare = st.number_input(
        "Fare Paid ($)", min_value=0.0, max_value=600.0, value=7.25, step=1.0
    )
    cabin = st.text_input(
        "Cabin Number (Optional)",
        value="",
        placeholder="e.g. C85 or leave blank",
    )
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton (S)",
            "C": "Cherbourg (C)",
            "Q": "Queenstown (Q)",
        }[x],
    )

st.markdown("---")

if st.button("Predict Survival Status", type="primary", use_container_width=True):
    raw_passenger_data = pd.DataFrame(
        [
            {
                "Pclass": pclass,
                "Name": name,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Ticket": ticket,
                "Fare": fare,
                "Cabin": cabin.strip() if cabin.strip() != "" else None,
                "Embarked": embarked,
            }
        ]
    )

    prediction = pipeline.predict(raw_passenger_data)[0]
    probabilities = pipeline.predict_proba(raw_passenger_data)[0]

    survival_prob = probabilities[1]
    perish_prob = probabilities[0]

    # Render Visual Output
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"### 🎉 Result: Survived!")
        st.metric(label="Survival Confidence", value=f"{survival_prob:.1%}")
        st.progress(float(survival_prob))
    else:
        st.error(f"### ⚠️ Result: Did Not Survive")
        st.metric(label="Likelihood of Perishing", value=f"{perish_prob:.1%}")
        st.progress(float(perish_prob))