import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle as plk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, BisectingKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
# import hdbscan 
from minisom import MiniSom 
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Development Clustering Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #2c3e50; text-align: center; }
    .stButton>button { background-color: #2980b9; color: white; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file):
    try:
        return pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df.copy()
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])
    return X_scaled, df_clean, numeric_cols

def load_pickled_model(model_name):
    """Attempts to load a model from a pickle file."""
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return plk.load(f)
    return None

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/earth-planet.png", width=100)
    st.title("Project Controls")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])
    
    model_choice = st.selectbox(
        "Choose Clustering Algorithm:",
        ["K-Means", "Hierarchical (Agglomerative)", "DBSCAN", "HDBSCAN", "GMM", "Divisive (Bisecting)", "Spectral Clustering", "Self-Organizing Maps (SOM)"]
    )
    
    use_pickle = st.checkbox("Try loading model from Pickle (.pkl)", value=False)

    if uploaded_file:
        df_temp = load_data(uploaded_file)
        if df_temp is not None:
            numeric_cols_temp = df_temp.select_dtypes(include=[np.number]).columns.tolist()
            x_axis = st.selectbox("X-Axis Feature", numeric_cols_temp, index=min(1, len(numeric_cols_temp)-1))
            y_axis = st.selectbox("Y-Axis Feature", numeric_cols_temp, index=min(2, len(numeric_cols_temp)-1))
    else:
        x_axis, y_axis = None, None

# --- MAIN APP LOGIC ---
st.title("🌍 Global Development Clustering Analysis")

if uploaded_file is None:
    st.warning("Please upload your dataset in the sidebar to begin.")
else:
    df_raw = load_data(uploaded_file)
    X_scaled, df_clean, numeric_cols = preprocess_data(df_raw)
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔬 Model Building", "📈 Comparative Analysis"])

    with tab1:
        st.header("Exploratory Data Analysis")
        col1, col2 = st.columns(2)
        with col1: st.dataframe(df_raw.head())
        with col2: st.dataframe(df_raw.describe())
        
        selected_dist_col = st.selectbox("Distribution View:", numeric_cols)
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.histplot(df_clean[selected_dist_col], kde=True, color='teal', ax=ax)
        st.pyplot(fig)

    with tab2:
        st.header(f"Model: {model_choice}")
        labels, score, model = None, None, None

        # Try loading from pickle if checked
        if use_pickle:
            model = load_pickled_model(model_choice)
            if model:
                st.success(f"Loaded {model_choice} from pickle file!")

        # --- MODEL LOGIC ---
        if model_choice == "K-Means":
            k_val = st.slider("Clusters (K)", 2, 10, 3)
            if not model: model = KMeans(n_clusters=k_val, random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        elif model_choice == "Hierarchical (Agglomerative)":
            k_val = st.slider("Clusters", 2, 10, 3)
            if not model: model = AgglomerativeClustering(n_clusters=k_val)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        elif model_choice == "DBSCAN":
            eps_val = st.slider("Epsilon", 0.1, 5.0, 0.5)
            if not model: model = DBSCAN(eps=eps_val, min_samples=5)
            labels = model.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                mask = labels != -1
                score = silhouette_score(X_scaled[mask], labels[mask]) if any(mask) else -1

        elif model_choice == "Spectral Clustering":
            k_val = st.slider("Clusters", 2, 10, 3)
            if not model: model = SpectralClustering(n_clusters=k_val, affinity='nearest_neighbors', random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        elif model_choice == "HDBSCAN":
            min_c = st.slider("Min Cluster Size", 2, 20, 5)
            if not model: model = hdbscan.HDBSCAN(min_cluster_size=min_c)
            labels = model.fit_predict(X_scaled)
            mask = labels != -1
            score = silhouette_score(X_scaled[mask], labels[mask]) if len(set(labels[mask])) > 1 else -1

        elif model_choice == "GMM":
            n_comp = st.slider("Components", 2, 10, 3)
            if not model: model = GaussianMixture(n_components=n_comp, random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        elif model_choice == "Divisive (Bisecting)":
            k_val = st.slider("Clusters", 2, 10, 3)
            if not model: model = BisectingKMeans(n_clusters=k_val, random_state=42)
            labels = model.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)

        elif model_choice == "Self-Organizing Maps (SOM)":
            grid = st.slider("Grid Size", 5, 20, 10)
            som = MiniSom(grid, grid, X_scaled.shape[1])
            som.train_random(X_scaled, 100)
            fig_som, ax_som = plt.subplots()
            ax_som.pcolor(som.distance_map().T, cmap='bone_r')
            st.pyplot(fig_som)

        # Plotting
        if labels is not None:
            df_plot = df_clean.copy()
            df_plot['Cluster'] = labels
            fig_scat, ax_scat = plt.subplots(figsize=(10, 5))
            sns.scatterplot(data=df_plot, x=x_axis, y=y_axis, hue='Cluster', palette='viridis', ax=ax_scat)
            st.pyplot(fig_scat)
            if score and score != -1: st.metric("Silhouette Score", f"{score:.3f}")

    with tab3:
        st.header("Comparative Analysis")
        if st.button("Run Comparison"):
            results = []
            models_to_test = {
                "K-Means": KMeans(n_clusters=3, random_state=42),
                "Hierarchical": AgglomerativeClustering(n_clusters=3),
                "GMM": GaussianMixture(n_components=3, random_state=42),
                "Spectral": SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42),
                "Divisive": BisectingKMeans(n_clusters=3, random_state=42)
            }
            for name, m in models_to_test.items():
                lbls = m.fit_predict(X_scaled)
                results.append({'Model': name, 'Score': silhouette_score(X_scaled, lbls)})
            
            res_df = pd.DataFrame(results).sort_values(by='Score', ascending=False)
            st.table(res_df)
            st.success(f"Best Model: {res_df.iloc[0]['Model']}")