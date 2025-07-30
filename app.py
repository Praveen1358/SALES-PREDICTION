import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Smart Salary Predictor", layout="centered")

# Custom style (ensure style.css is in the same folder and correctly referenced)
st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)

# Load model and label encoders
model = joblib.load("model.pkl")
encoders = joblib.load("label_encoders.pkl")

# App title
st.title("💼 Salary Prediction App")
st.write("Enter the required details below to predict whether the salary is greater than 50K or not.")

# User input form
with st.form("prediction_form"):
    st.subheader("Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
        education = st.selectbox("Education", encoders["education"].classes_)
        occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
        race = st.selectbox("Race", encoders["race"].classes_)

    with col2:
        marital_status = st.selectbox("Marital Status", encoders["marital-status"].classes_)
        relationship = st.selectbox("Relationship", encoders["relationship"].classes_)
        sex = st.selectbox("Sex", encoders["sex"].classes_)
        hours = st.number_input("Hours Per Week", min_value=1, max_value=100, value=40)
        native_country = st.selectbox("Native Country", encoders["native-country"].classes_)

    submit = st.form_submit_button("Predict Salary")

    if submit:
        # Prepare input for model (encoded)
        input_dict = {
            "age": age,
            "workclass": encoders["workclass"].transform([workclass])[0],
            "education": encoders["education"].transform([education])[0],
            "marital-status": encoders["marital-status"].transform([marital_status])[0],
            "occupation": encoders["occupation"].transform([occupation])[0],
            "relationship": encoders["relationship"].transform([relationship])[0],
            "race": encoders["race"].transform([race])[0],
            "sex": encoders["sex"].transform([sex])[0],
            "hours-per-week": hours,
            "native-country": encoders["native-country"].transform([native_country])[0],
        }

        # Model prediction
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        salary_range = encoders["income"].inverse_transform([prediction])[0]

        # 👇 Build readable summary table like screenshot
        readable_data = {
            "Age": age,
            "Workclass": workclass,
            "Education": education,
            "Marital Status": marital_status,
            "Occupation": occupation,
            "Relationship": relationship,
            "Race": race,
            "Sex": sex,
            "Hours per Week": hours,
            "Native Country": native_country,
        }

        result_df = pd.DataFrame([readable_data])
        result_df["Predicted Salary Range"] = salary_range  # add final column

        # Display results
        st.markdown(f"### 💰 <span style='color:#00bcd4'>Predicted Salary Range: {salary_range}</span>", unsafe_allow_html=True)
        st.markdown("### 🧾 Prediction Summary")
        st.dataframe(result_df, use_container_width=True)

