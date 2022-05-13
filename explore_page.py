import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from train_and_save import shorten_categories, year_to_int, reduce_education
import pickle


@st.cache
def load_data():
    try:
        with open("saved_df.pkl", "rb") as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print("Please run train_and_save.py")
        exit(0)


df = load_data()


def show_explore_page():
    st.title("Explore Data")

    st.write("""### Stack Overflow Survey Data 2021""")

    data = df["Country"].value_counts()

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.pie(data, autopct="%1.1f%%", shadow=True, startangle=90)
    fig.legend(data.index, loc="upper left")
    st.write("""#### Data by country""")
    st.pyplot(fig)

    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["CompTotal"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["CompTotal"].mean().sort_values(ascending=True)
    st.line_chart(data)
