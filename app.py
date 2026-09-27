
import json

import joblib
import pandas as pd
import streamlit as st


MODEL_PATH = "model/churn_model.joblib"
METADATA_PATH = "model/model_metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


model = load_model()
metadata = load_metadata()

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Enter customer information to estimate the probability "
    "that the customer will churn."
)

st.markdown("---")

input_data = {}

for column in metadata["feature_columns"]:

    if column in metadata["numeric_features"]:

        if column == "SeniorCitizen":
            input_data[column] = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

        elif column == "tenure":
            input_data[column] = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12,
                step=1
            )

        elif column == "MonthlyCharges":
            input_data[column] = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=50.0,
                step=1.0
            )

        elif column == "TotalCharges":
            input_data[column] = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=500.0,
                step=10.0
            )

        else:
            input_data[column] = st.number_input(
                column,
                value=0.0
            )

    else:
        options = metadata["categories"].get(column, [])

        input_data[column] = st.selectbox(
            column.replace("_", " "),
            options
        )


st.markdown("---")


if st.button(
    "🔮 Predict Churn",
    use_container_width=True
):

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    churn_probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(
            f"⚠️ High churn risk — probability: "
            f"{churn_probability:.2%}"
        )
        st.write(
            "The model predicts that this customer is likely to churn."
        )
    else:
        st.success(
            f"✅ Low churn risk — probability: "
            f"{1 - churn_probability:.2%}"
        )
        st.write(
            "The model predicts that this customer is likely to stay."
        )

    st.progress(float(churn_probability))

    st.caption(
        "This prediction is an estimate produced by the machine learning model "
        "and is not a guarantee of future customer behavior."
    )
