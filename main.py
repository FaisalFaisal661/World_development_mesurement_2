# main.py
import streamlit as st

import pandas as pd
import pickle as plk

from data_loader import load_data_from_drive, load_models
st.set_page_config(page_title="ML Project Dashboard", layout="wide")

st.title("Welcome to the ML Analysis Dashboard")
st.markdown("""
This app integrates data from Google Drive to perform:
1. **Exploratory Data Analysis** (EDA)
2. **Predictive Modeling** (ML Predictor)
""")

st.write("Highlight of the dataset")


df =load_data_from_drive()
st.write(df)
# Navigation
st.sidebar.title("Navigation")

