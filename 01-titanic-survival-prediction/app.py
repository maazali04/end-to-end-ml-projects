import os
import requests
import streamlit as st

import streamlit as st
import requests

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "https://maazali04-titanic-api.vercel.app/predict"
    
st.set_page_config(
    page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered"
)

st.title("Titanic Passenger Survival Predictor")
st.write(
    "Enter passenger details below to predict whether they would have survived the Titanic disaster."
)

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
    payload = {
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

    try:
        with st.spinner("Connecting to FastAPI backend..."):
            response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            prediction = result["survived"]
            survival_prob = result["survival_probability"]
            perish_prob = 1.0 - survival_prob

            st.subheader("Prediction Result")

            if prediction == 1:
                st.success("### 🎉 Result: Survived!")
                st.metric(label="Survival Confidence", value=f"{survival_prob:.1%}")
                st.progress(float(survival_prob))
            else:
                st.error("### ⚠️ Result: Did Not Survive")
                st.metric(label="Likelihood of Perishing", value=f"{perish_prob:.1%}")
                st.progress(float(perish_prob))

        else:
            st.error(f"⚠️ API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Could not connect to FastAPI backend server. "
            "Please ensure Uvicorn is running on port 8000 (`uvicorn main:app --reload`)."
        )
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. Backend server took too long to respond.")