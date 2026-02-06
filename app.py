import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, BisectingKMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import hdbscan
from minisom import MiniSom

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Development Clustering | Professional Edition",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLING & CSS ---
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'model' not in st.session_state:
    st.session_state.model = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file):
    """Loads CSV or Excel data efficiently."""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

def preprocess_data(df):
    """Handles missing values and scaling."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_clean = df.copy()
    
    # Fill missing values with median
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])
    return X_scaled, df_clean, numeric_cols

def save_model_to_pickle(model, model_name):
    """Saves the trained model to a pickle file."""
    filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    os.makedirs("models", exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    return filename

def load_model_from_pickle(model_name):
    """Loads a model from a pickle file."""
    filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🎛️ Project Controls")
    
    uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx"])
    
    st.divider()
    
    mode = st.radio("Operation Mode", ["Train New Model", "Load Saved Model"])
    
    model_choice = st.selectbox(
        "Select Algorithm:",
        ["K-Means", "Hierarchical", "DBSCAN", "HDBSCAN", "GMM", "Spectral", "Bisecting K-Means", "SOM"]
    )
    
    # Dynamic Hyperparameters based on model choice
    params = {}
    if mode == "Train New Model":
        with st.expander("⚙️ Hyperparameters", expanded=True):
            if model_choice in ["K-Means", "Hierarchical", "Spectral", "Bisecting K-Means", "GMM"]:
                params['n_clusters'] = st.slider("Number of Clusters (k)", 2, 15, 3)
            elif model_choice == "DBSCAN":
                params['eps'] = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                params['min_samples'] = st.slider("Min Samples", 2, 20, 5)
            elif model_choice == "HDBSCAN":
                params['min_cluster_size'] = st.slider("Min Cluster Size", 2, 50, 5)
            elif model_choice == "SOM":
                params['grid_size'] = st.slider("Grid Size (N x N)", 5, 30, 10)

# --- MAIN APPLICATION ---

col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🌍 Global Development Clustering")
    st.caption("Advanced Unsupervised Learning Analysis Platform")

if uploaded_file is None:
    st.info("👋 Welcome! Please upload your dataset in the sidebar to begin analysis.")
    # Create dummy data for visualization demo (optional)
else:
    df_raw = load_data(uploaded_file)
    X_scaled, df_clean, numeric_cols = preprocess_data(df_raw)
    st.session_state.df_clean = df_clean

    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Intelligence", "🧠 Model Engine", "📉 Comparative Lab"])

    # TAB 1: DATA INTELLIGENCE
    with tab1:
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df_raw.shape[0])
        col2.metric("Features", df_raw.shape[1])
        col3.metric("Missing Values", df_raw.isna().sum().sum())
        
        with st.expander("🔍 View Raw Data & Statistics"):
            st.dataframe(df_raw.head())
            st.dataframe(df_raw.describe())

        st.subheader("Feature Distribution Analysis")
        c1, c2 = st.columns([1, 3])
        with c1:
            dist_col = st.selectbox("Select Feature:", numeric_cols)
        with c2:
            fig = px.histogram(df_clean, x=dist_col, marginal="box", color_discrete_sequence=['#00CC96'])
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # TAB 2: MODEL ENGINE
    with tab2:
        col_m1, col_m2 = st.columns([3, 1])
        
        with col_m1:
            st.subheader(f"Active Model: {model_choice}")
        
        # --- MODEL TRAINING / LOADING LOGIC ---
        if st.button("🚀 Run Analysis", type="primary"):
            model = None
            labels = None
            
            try:
                if mode == "Load Saved Model":
                    model = load_model_from_pickle(model_choice)
                    if model:
                        st.success(f"✅ Successfully loaded {model_choice} from storage.")
                        # Prediction Logic for Loaded Models
                        if hasattr(model, 'fit_predict'):
                            labels = model.fit_predict(X_scaled)
                        elif hasattr(model, 'predict'):
                            labels = model.predict(X_scaled)
                        else:
                            # Fallback for models that don't store state nicely for predict (like DBSCAN)
                            st.warning("Loaded model object doesn't support direct prediction on new data. Retraining recommended.")
                    else:
                        st.error(f"⚠️ No saved model found for {model_choice}. Please train one first.")

                else:  # Train New Model
                    with st.spinner(f"Training {model_choice}..."):
                        if model_choice == "K-Means":
                            model = KMeans(n_clusters=params['n_clusters'], random_state=42)
                        elif model_choice == "Hierarchical":
                            model = AgglomerativeClustering(n_clusters=params['n_clusters'])
                        elif model_choice == "DBSCAN":
                            model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
                        elif model_choice == "HDBSCAN":
                            model = hdbscan.HDBSCAN(min_cluster_size=params['min_cluster_size'])
                        elif model_choice == "GMM":
                            model = GaussianMixture(n_components=params['n_clusters'], random_state=42)
                        elif model_choice == "Spectral":
                            model = SpectralClustering(n_clusters=params['n_clusters'], affinity='nearest_neighbors', random_state=42)
                        elif model_choice == "Bisecting K-Means":
                            model = BisectingKMeans(n_clusters=params['n_clusters'], random_state=42)
                        elif model_choice == "SOM":
                            st.info("SOM visualization is generated directly.")
                            
                        # Fit and Predict
                        if model_choice != "SOM":
                            labels = model.fit_predict(X_scaled)
                            
                            # Save the trained model
                            saved_path = save_model_to_pickle(model, model_choice)
                            st.success(f"Model trained & saved to {saved_path}")

                # Store in session state
                if labels is not None:
                    st.session_state.labels = labels
                    st.session_state.model = model

            except Exception as e:
                st.error(f"An error occurred during modeling: {str(e)}")

        # --- VISUALIZATION LOGIC ---
        if st.session_state.labels is not None:
            labels = st.session_state.labels
            df_viz = df_clean.copy()
            df_viz['Cluster'] = labels.astype(str)
            
            # Metric Calculation
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(labels).count(-1)
            
            # Calculate Silhouette Score (ignore if only 1 cluster or all noise)
            score_display = "N/A"
            if len(unique_labels) > 1:
                mask = labels != -1
                if mask.sum() > 0:
                    score = silhouette_score(X_scaled[mask], labels[mask])
                    score_display = f"{score:.3f}"

            # Results Dashboard
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters Found", n_clusters)
            m2.metric("Noise Points", n_noise, delta_color="inverse")
            m3.metric("Silhouette Coefficient", score_display)

            # Interactive Plotly Scatter
            st.subheader("Cluster Visualization")
            c_x, c_y, c_z = st.columns(3)
            x_ax = c_x.selectbox("X Axis", numeric_cols, index=0)
            y_ax = c_y.selectbox("Y Axis", numeric_cols, index=1)
            
            # 2D Scatter
            fig = px.scatter(
                df_viz, x=x_ax, y=y_ax, color='Cluster',
                hover_data=df_viz.columns[:3], # Show first 3 cols on hover
                title=f"{model_choice} Clustering Results",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # SOM Special Case
        if model_choice == "SOM" and mode == "Train New Model" and 'params' in locals():
             if st.button("Generate SOM Map"):
                som = MiniSom(params['grid_size'], params['grid_size'], X_scaled.shape[1])
                som.train_random(X_scaled, 100)
                plt.figure(figsize=(8, 8))
                plt.pcolor(som.distance_map().T, cmap='bone_r')
                plt.colorbar()
                st.pyplot(plt)

    # TAB 3: COMPARATIVE LAB
    with tab3:
        st.subheader("🏆 Model Benchmarking")
        st.write("Compare the performance of different algorithms on your dataset.")
        
        if st.button("Run Comprehensive Benchmark"):
            with st.spinner("Running tournament..."):
                results = []
                models_to_test = {
                    "K-Means (k=3)": KMeans(n_clusters=3, random_state=42),
                    "Agglomerative (k=3)": AgglomerativeClustering(n_clusters=3),
                    "GMM (k=3)": GaussianMixture(n_components=3, random_state=42),
                    "Bisecting KM (k=3)": BisectingKMeans(n_clusters=3, random_state=42)
                }
                
                for name, m in models_to_test.items():
                    try:
                        lbls = m.fit_predict(X_scaled)
                        s_score = silhouette_score(X_scaled, lbls)
                        results.append({'Algorithm': name, 'Silhouette Score': s_score})
                    except:
                        continue
                
                res_df = pd.DataFrame(results).sort_values(by='Silhouette Score', ascending=False)
                
                # Plotly Bar Chart for Comparison
                fig_comp = px.bar(
                    res_df, x='Silhouette Score', y='Algorithm', 
                    orientation='h', 
                    color='Silhouette Score',
                    color_continuous_scale='Viridis',
                    text_auto='.3f',
                    title="Algorithm Performance Ranking"
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                best_model = res_df.iloc[0]
                st.success(f"🏅 Best Performer: **{best_model['Algorithm']}** with a score of **{best_model['Silhouette Score']:.3f}**")
