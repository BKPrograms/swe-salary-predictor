import streamlit as st
import pickle
import numpy as np


def load_model():
    try:
        with open("saved_model.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Please run train_and_save.py")
        exit(0)


data = load_model()
regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]


def show_predict_page():
    st.title("SWE Salary Predictor")
    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States of America",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    educationlvl = st.selectbox("Education", education)

    experience = st.slider("Years of Experience", 0, 50, 1)

    clicked = st.button("Calculate Salary")

    if clicked:
        X = np.array([[country, educationlvl, experience]])
        X[:, 0] = le_country.transform(X[:, 0])
        X[:, 1] = le_education.transform(X[:, 1])
        X = X.astype(float)

        salary = regressor_loaded.predict(X)
        st.subheader(f"Estimated salary is ${format(round(salary[0]), ',')} USD")
