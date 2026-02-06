import pandas as pd
import numpy as np
import sklearn
import pickle as plk
import gdown
import streamlit as st

st.title('Hello')

#-- Load the dataset --
# data_loader.py


def load_data_from_drive(url):
    # Extract the file ID from the sharing link
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    output = 'data.csv'
    gdown.download(download_url, output, quiet=False)
    return pd.read_csv(output)

# @st.cache_data
# def load_data_from_drive(url):
#     # Extract the file ID from the sharing link
#     file_id = url.split('/1HtRgaj_ETt-3G8h0aoIpIzk2RuHjijUV')[-2]
#     download_url = f'https://drive.google.com/file/uc?id={file_id}'
#     output = 'steamlit_finalized.csv'
#     gdown.download(download_url, output, quiet=False)
    

df = load_data_from_drive('https://drive.google.com/file/d/1HtRgaj_ETt-3G8h0aoIpIzk2RuHjijUV/view?usp=drive_link')

st.write(df.head())

# -- Loading all Models --
@st.cache_resource
def load_models():
    kmeans = plk.load(open('kmeans_model.pkl', 'rb'))
    divisive = plk.load(open('divisive_model.pkl', 'rb'))
    gmm = plk.load(open('gmm_model.pkl', 'rb'))
    hdbscan = plk.load(open('hdbscan_model.pkl', 'rb'))
    hierarchical = plk.load(open('hierarchical_model.pkl', 'rb'))
    som = plk.load(open('som_model.pkl', 'rb'))
    spectral = plk.load(open('spectral_clustering_model.pkl', 'rb'))
    vectorizer = plk.load(open('vectorizer.pkl', 'rb'))
    return kmeans, divisive, gmm,hdbscan,hierarchical,som,spectral, vectorizer

kmeans, divisive, gmm,hdbscan,hierarchical,som,spectral, vectorizer = load_models()
st.write("All models loaded successfully!")   