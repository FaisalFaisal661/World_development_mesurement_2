import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# import hdbscan 
from minisom import MiniSom 
import streamlit as st
import requests
import io



# --- CONFIGURATION ---
GITHUB_URL = "https://raw.githubusercontent.com/FaisalFaisal661/World_development_mesurement_2/refs/heads/main/finalized.csv"
GITHUB_MODEL_BASE_URL = (
    "https://raw.githubusercontent.com/FaisalFaisal661/World_development_mesurement_2/main/models"
)

@st.cache_data
def load_data(url):
    try:
        
        if "github.com" in url and "raw" not in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"🌐 **GitHub Connection Error:** {e}")
        return None
# Execution
df = load_data(GITHUB_URL)
if df is not None:
    st.success("✅ Data loaded successfully!")
    st.dataframe(df.head())



df_scaled = df.select_dtypes(include=np.number)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Clustering & Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)
df_scaled_data = df.select_dtypes(include=np.number).values
# --- HELPER FUNCTION: LOAD MODELS ---
MODEL_DIR = "models"  

@st.cache_resource
def load_model(model_name):
    """
    Loads a pickle model from a GitHub repository.
    """
    # Standardize filename: 'K-Means' -> 'k_means_model.pkl'
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    model_url = f"{GITHUB_MODEL_BASE_URL}/{filename}"

    try:
        response = requests.get(model_url)
        response.raise_for_status()  # raises error for 404 / 403

        model = pickle.load(io.BytesIO(response.content))
        return model

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to download model from GitHub: {e}")
    except Exception as e:
        st.error(f"❌ Error loading model {filename}: {e}")
    
    
    return None

# --- PREPARE FEATURE LIST (EXCLUDE CLUSTER LABELS) ---
features = df.select_dtypes(include=[np.number]).columns.tolist()
cluster_cols = ['KMeans_Cluster', 'Hierarchical_Cluster', 'Divisive_Cluster', 
                    'DBSCAN_Cluster', 'HDBSCAN_Cluster', 'GMM_Cluster', 'Spectral_Cluster']
feature_list = [c for c in features if c not in cluster_cols]

# --- Loading Models ---
km = load_model("K-Means")
agg = load_model("Agglomerative") 
bk = load_model("Divisive")
dbscan = load_model("DBSCAN")    
# hdbscan = load_model("HDBSCAN")
gmm = load_model("GMM")
scale = load_model("Scaler")
som = load_model("MiniSom")

# Create models dictionary for easy access
models = {
    'KMeans': km,
    'Agglomerative': agg,
    'Divisive': bk,
    'DBSCAN': dbscan,
    # 'HDBSCAN': hdbscan,
    'GMM': gmm,
    'Scaler': scale,
    'SOM': som
}

# --- APP NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Visualizations", "Model Prediction", "Model Comparison"])


# ==========================================
# 1. HOME PAGE
# ==========================================
if page == "Home":
    

    st.title("🌐 Global Development Intelligence Suite")
    st.markdown("""
        <style>
        .main-header { font-size: 24px; font-weight: bold; color: #1E3A8A; }
        .stat-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #1E3A8A; }
        </style>
    """, unsafe_allow_html=True)

    st.info("💡 **Objective:** Using Unsupervised Machine Learning to segment 2,704 global regions into developmental tiers based on socio-economic indicators.")

    # 2. Key Metrics Dashboard (High-Level Summary)
    st.subheader("📊 Dataset at a Glance")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{len(df):,}")
    k2.metric("Features Used", "25")
    k3.metric("Algorithms Implemented", "8")
    k4.metric("Clustering Tiers", f"{df['KMeans_Cluster'].nunique()}")

    st.markdown("---")

    # 3. Two-Column Layout for Context and Logic
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🚀 Project Scope")
        st.write("""
        This platform analyzes global health, wealth, and infrastructure data. By applying multiple 
        clustering paradigms, we can identify patterns that traditional economic ranking often misses.
        
        **Methodologies Included:**
        - **Centroid-Based:** Standard K-Means (The Baseline)
        - **Density-Based:** DBSCAN & HDBSCAN (Identifying Global Outliers)
        - **Connectivity-Based:** Hierarchical & Divisive (Structural Relationships)
        - **Probabilistic:** Gaussian Mixture Models (Uncertainty & Overlap)
        - **Neural-Based:** Self-Organizing Maps (High-Dimensional Mapping)
        """)
        
        # Display a sample of the data with color coding for clusters
        st.subheader("📍 Data Preview (Development Tiers)")
        st.dataframe(df[['Country', 'GDP', 'Life Expectancy Average', 'KMeans_Cluster']].head(10), use_container_width=True)

    with col2:
        st.subheader("🔍 Strategic Utility")
        with st.expander("💼 Business Intelligence", expanded=True):
            st.write("Analyze market gaps and identify 'look-alike' countries for business expansion.")
        
        with st.expander("🏛️ Policy Design"):
            st.write("Group nations with similar health challenges to share successful policy frameworks.")
        
        with st.expander("📉 Risk Management"):
            st.write("Detect outlier countries that don't fit standard economic models (DBSCAN results).")

    # 4. Visual Navigation Guide
    st.markdown("---")
    st.subheader("🛠️ Quick Actions")
    q1, q2, q3 = st.columns(3)
    if q1.button("View Cluster DNA"):
        st.info("Head to the **Visualizations** page and select 'Cluster Profiles'.")
    if q2.button("Compare Algorithms"):
        st.info("Head to the **Model Comparison** page.")
    if q3.button("Run Business Simulation"):
        st.info("Head to the **Strategic Insights** page.")


# ==========================================
# 2. VISUALIZATIONS PAGE
# ==========================================
elif page == "Visualizations":
    st.title("📊 Data Visualizations")    
    st.header("📊 Clustering Visualizations")
    

    # Dropdown to select specific model visualization
    viz_choice = st.selectbox(
        "Select Model to Visualize:",
        ["Compare: K-Means/Hierarchical/Divisive", 
         "DBSCAN", 
         "HDBSCAN", 
         "Gaussian Mixture (GMM)", 
         "Spectral Clustering",
         "Cluster Profiles & Benchmarks"]
    )
    # Helper for feature selection
    def get_feature_selectors(default_x, default_y):
        c1, c2 = st.columns(2)
        x = c1.selectbox(f"Select X-Axis:", feature_list, index=feature_list.index(default_x) if default_x in feature_list else 0)
        y = c2.selectbox(f"Select Y-Axis:", feature_list, index=feature_list.index(default_y) if default_y in feature_list else 1)
        return x, y, feature_list.index(x), feature_list.index(y)


# --- 1. K-Means / Hierarchical / Divisive ---
    if viz_choice == "Compare: K-Means/Hierarchical/Divisive":
        
        st.subheader("Comparison: Centroid vs. Hierarchical Approaches")
        fig, (ax1, ax2 ,ax3) = plt.subplots(1, 3, figsize=(16, 6))

        sns.scatterplot(data=df, x='Birth Rate', y='Life Expectancy Average', hue='KMeans_Cluster', ax=ax1, palette='Set1')
        ax1.set_title('Standard K-Means (Centroid-Based)')

        sns.scatterplot(data=df, x='Birth Rate', y='Life Expectancy Average', hue='Hierarchical_Cluster', ax=ax2, palette='Set2')
        ax2.set_title('Agglomerative Clustering (Bottom-Up)')

        sns.scatterplot(data=df, x='Birth Rate', y='Life Expectancy Average', hue='Divisive_Cluster', ax=ax3, palette='Set3')
        ax3.set_title('Divisive Clustering (Top-Down)')
        
        st.pyplot(fig)

        # Dendrogram Toggle [cite: 31-42]
        if st.checkbox("Show Dendrograms"):
            st.write("Computing Dendrograms...")
            linkage_agg = sch.linkage(df_scaled_data, method='ward')
            linkage_div = sch.linkage(df_scaled_data, method='complete') # Proxy for divisive

            fig_dendro, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(25, 10))
            
            sch.dendrogram(linkage_agg, truncate_mode='lastp', p=40, ax=ax_d1, leaf_rotation=90)
            ax_d1.axhline(y=25, color='r', linestyle='--', label='Cut-off (d=25)')
            ax_d1.set_title('Agglomerative (Ward Linkage)')
            
            sch.dendrogram(linkage_div, truncate_mode='lastp', p=40, ax=ax_d2, leaf_rotation=90)
            ax_d2.axhline(y=6, color='r', linestyle='--', label='Macro Split')
            ax_d2.set_title('Divisive Proxy (Complete Linkage)')
            
            st.pyplot(fig_dendro)

    # --- 2. DBSCAN ---
    elif viz_choice == "DBSCAN":
        st.subheader("DBSCAN: Density-Based Discovery")
        x_feat, y_feat, _, _ = get_feature_selectors('GDP', 'Energy Usage')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        unique_clusters = sorted(list(set(df['DBSCAN_Cluster'].unique())))
        colors = sns.color_palette("viridis", len(unique_clusters))
        palette = {cluster: colors[i] if cluster != -1 else "#333333" for i, cluster in enumerate(unique_clusters)}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=df, x=x_feat, y=y_feat, hue='DBSCAN_Cluster',
            palette=palette, style=(df['DBSCAN_Cluster'] == -1),
            markers={True: 'X', False: 'o'}, s=100, alpha=0.7, edgecolor='w', ax=ax
        )
        if st.checkbox("Apply Log Scale"):
                ax.set_xscale('log')
                ax.set_yscale('log')

        ax.set_title(f'DBSCAN Clusters ({x_feat} vs {y_feat})'   )
        st.pyplot(fig)        
        n_noise = list(df['DBSCAN_Cluster']).count(-1)
        st.write(f"**Noise Points Detected:** {n_noise}")

    # --- 3. HDBSCAN ---
    elif viz_choice == "HDBSCAN":
        st.subheader("HDBSCAN: Hierarchical Density")
        x_feat, y_feat, _, _ = get_feature_selectors('GDP', 'Birth Rate')       
        fig, ax = plt.subplots(figsize=(12, 8))        
        unique_clusters = sorted(df['HDBSCAN_Cluster'].unique())        
        sns.scatterplot(
            data=df, x=x_feat, y=y_feat, hue='HDBSCAN_Cluster',
            palette="husl", style='HDBSCAN_Cluster', alpha=0.7, s=100, edgecolor='w', ax=ax
        )
        if st.checkbox("Apply Log Scale"):
                ax.set_xscale('log')
                ax.set_yscale('log')
        
        if x_feat == 'GDP' and y_feat == 'Birth Rate':
            ax.set_title('HDBSCAN Clustering: Major Global Development Zones')
        else:
            ax.set_title(f'HDBSCAN Clustering: {x_feat} vs {y_feat}')
        st.pyplot(fig)

    # --- 4. GMM ---
    elif viz_choice == "Gaussian Mixture (GMM)":
        st.subheader("GMM: Probabilistic Clustering")
        x_feat, y_feat, idx_x, idx_y = get_feature_selectors('GDP', 'Infant Mortality Rate')

        
        # Helper function to draw covariance ellipses
        def draw_ellipse(position, covariance, ax=None, **kwargs):
            ax = ax or plt.gca()
            if covariance.shape == (2, 2):
                U, s, Vt = np.linalg.svd(covariance)
                angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
                width, height = 2 * np.sqrt(s)
            else:
                angle = 0
                width, height = 2 * np.sqrt(covariance)
            for nsig in range(1, 4):
                ax.add_patch(Ellipse(xy=position, width=nsig * width, height=nsig * height, angle=angle, **kwargs))

                      
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=df, x=x_feat, y=y_feat, hue='GMM_Cluster', palette='viridis', 
         s=100, alpha=0.4, ax=ax)
        
        if 'GMM' in models:
            gmm = models['GMM']
            try:                
                centers = gmm.means_[:, [idx_x, idx_y]] 
                covars = gmm.covariances_[:, [idx_x, idx_y]][:, :, [idx_x, idx_y]]
                
                for i in range(len(centers)):
                    draw_ellipse(centers[i], covars[i], ax=ax, alpha=0.15, 
                                 color=plt.cm.viridis(i / (gmm.n_components - 1)))                
                st.success(f"Showing ellipses for {x_feat} vs {y_feat}")

            except Exception as e:
                st.error(f"Mapping Error: {e}")
                st.info("Check if the model features match the CSV features.")

            if st.checkbox("Apply Log Scale"):
                ax.set_xscale('log')
                ax.set_yscale('log')

            ax.set_title(f'GMM Clustering: {x_feat} vs {y_feat}')
            st.pyplot(fig)

   # --- 5. SPECTRAL (PCA) ---
    elif viz_choice == "Spectral Clustering":
        st.subheader("Spectral: Graph-Based Clustering (PCA Reduced)")
     
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(df_scaled_data)
        pca_df = pd.DataFrame(data=pca_components, columns=['PC1', 'PC2'])
        pca_df['Spectral_Cluster'] = df['Spectral_Cluster'].values
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='Spectral_Cluster', data=pca_df, 
                        palette='viridis', s=150, alpha=0.9, ax=ax)
        if st.checkbox("Apply Log Scale"):
                ax.set_xscale('log')
                ax.set_yscale('log')
        ax.set_title('Spectral Clustering Visualization (PCA Reduced)')
        st.pyplot(fig)

    # --- 6. PROFILES & MAPS ---
    elif viz_choice == "Cluster Profiles & Benchmarks":
        st.subheader("Global Development Map")
        
        fig_map = px.choropleth(
            df, locations="Country", locationmode='country names',
            color='Life Expectancy Average', 
            hover_data=['KMeans_Cluster', 'GDP'],
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Global Development Map"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        st.subheader("Cluster DNA (Heatmap)")
       
        key_metrics = ['GDP', 'Birth Rate', 'CO2_per_capita', 'Infant Mortality Rate', 'Life Expectancy Average', 'Internet Usage']
        # Check if columns exist before grouping
        available_metrics = [m for m in key_metrics if m in df.columns]
        
        if available_metrics:
            # Allow user to choose which cluster type to benchmark against
            selected_cluster_col = st.selectbox("Select Cluster Type for Benchmarking:", 
                                                ['KMeans_Cluster', 'GMM_Cluster', 'Spectral_Cluster', 
                                                 'HDBSCAN_Cluster', 'Hierarchical_Cluster', 'Divisive_Cluster'])
            
            # Recalculate summary based on selected cluster column
            cluster_summary = df.groupby(selected_cluster_col)[available_metrics].mean()
            scaler = StandardScaler()
            scaled_summary = pd.DataFrame(scaler.fit_transform(cluster_summary), 
                                          index=cluster_summary.index, columns=cluster_summary.columns)
            
            # Heatmap Visualization 
            fig_heat, ax = plt.subplots(figsize=(16, 8))
            sns.heatmap(scaled_summary, annot=cluster_summary, fmt=".2f", cmap="RdYlGn", 
                        linewidths=.5, annot_kws={"size": 12, "weight": "bold"}, ax=ax)
            ax.set_title("Development DNA: Strategic Cluster Profiles (Z-Score Heatmap)", fontsize=20, fontweight='bold')
            ax.set_xlabel('Economic & Social Metrics', fontsize=12)
            ax.set_ylabel(f'{selected_cluster_col} ID', fontsize=12)
            st.pyplot(fig_heat)
            
            # Cluster vs Global Benchmarks Logic 
            st.subheader(f"Performance Analysis: {selected_cluster_col} vs Global Benchmarks")
            global_means = df[available_metrics].mean()
            
            # Calculate grid dimensions (2 rows, 3 columns for 6 metrics)
            num_metrics = len(available_metrics)
            cols = 3
            rows = (num_metrics + cols - 1) // cols
            
            fig_bench, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
            axes = axes.flatten()
            
            for i, metric in enumerate(available_metrics):
                # Create bar plot for each cluster's mean value of the metric
                sns.barplot(
                    x=cluster_summary.index,
                    y=cluster_summary[metric],
                    ax=axes[i],
                    palette="crest",
                    alpha=0.8,
                    edgecolor='black'
                )
                
                # Add the Global Benchmark line 
                avg_val = global_means[metric]
                axes[i].axhline(avg_val, ls='--', color='red', alpha=0.6, label=f'Global Avg: {avg_val:.2f}')
                
                # Formatting each subplot 
                axes[i].set_title(f'{metric}', fontsize=16, fontweight='bold')
                axes[i].set_xlabel('Cluster ID', fontsize=12)
                axes[i].set_ylabel('Mean Value', fontsize=12)
                
                # Annotate bars with actual values
                for p in axes[i].patches:
                    axes[i].annotate(f'{p.get_height():.1f}',
                                     (p.get_x() + p.get_width() / 2., p.get_height()),
                                     ha='center', va='center', xytext=(0, 9),
                                     textcoords='offset points', fontsize=11, fontweight='bold')
            
            # Remove any empty subplots
            for j in range(i + 1, len(axes)):
                fig_bench.delaxes(axes[j])
                
            plt.suptitle(f'{selected_cluster_col} Performance vs. Global Benchmarks', fontsize=24, fontweight='bold', y=1.02)
            plt.tight_layout()
            st.pyplot(fig_bench)

    

# # ==========================================
# # 3. MODEL PREDICTION PAGE
# # ==========================================
elif page == "Model Prediction":
    st.title("🤖 Model Prediction")
    st.write("🎯 Strategic Action Plans by Cluster")

    # calculate feature columns by excluding cluster labels was taking too long, so I hardcoded the list here for simplicity. keeing in mind that we need to upload it on streamlit, we should avoid any heavy computation on the fly.
    
    
    feature_cols = ['Birth Rate', 'Life Expectancy Average', 'GDP', 'Energy Usage', 
                    'Infant Mortality Rate', 'Internet Usage', 'CO2_per_capita']

 
    
    # User selects a country to analyze
    target_country = st.selectbox("Select Country for Strategic Analysis:", df['Country'].unique())
    country_data = df[df['Country'] == target_country].iloc[0]
    cluster_id = country_data['KMeans_Cluster']
    
    st.subheader(f"Strategy Profile for {target_country} (Cluster {cluster_id})")
    
    # Dynamic Business Logic based on your Cluster Profiles [cite: 329-354]
    strategies = {
        0: {"Name": "Premium Market", "Strategy": "High-end product launches and infrastructure investment.", "Priority": "Critical"},
        1: {"Name": "Developing Hub", "Strategy": "Focus on digital literacy and mid-tier consumer goods.", "Priority": "High"},
        2: {"Name": "Emerging Industrial", "Strategy": "Industrial automation and energy efficiency programs.", "Priority": "Medium"},
        3: {"Name": "Foundation Tier", "Strategy": "Basic healthcare and educational aid programs.", "Priority": "Essential"},
        4: {"Name": "Stagnant Growth", "Strategy": "Economic reform incentives and structural support.", "Priority": "Urgent"}
    }
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Economic Standing", strategies[cluster_id]["Name"])
        st.info(f"**Action Plan:** {strategies[cluster_id]['Strategy']}")
    with col2:
        st.warning(f"**Investment Priority:** {strategies[cluster_id]['Priority']}")

 
    # What-If Analysis Simulation

    st.subheader("🛠️ Development Simulation (What-If?)")
    selected_model = st.selectbox("Select Model for Simulation:", ['KMeans_Cluster', 'GMM_Cluster', 'Spectral_Cluster', 'HDBSCAN_Cluster', 'Hierarchical_Cluster', 'Divisive_Cluster'], index=3)

    selected_metric = st.selectbox("Select Metric to Simulate:", ['Internet Usage', 'GDP', 'Energy Usage','life Expectancy Average', 'Birth Rate', 'Infant Mortality Rate', 'CO2_per_capita'])
    current_val = country_data[selected_metric]
    cluster_avg = df[df[selected_model] == cluster_id][selected_metric].mean()

    # Simulation Slider
    simulated_val = st.slider(f"Adjust {selected_metric} for {target_country}:", 
                          float(df[selected_metric].min()), float(df[selected_metric].max()), float(current_val))

    # Visualization
    fig, ax = plt.subplots()
    sns.barplot(x=['Global Avg', f'Cluster {cluster_id} Avg', f'Simulated {target_country}'], 
            y=[df[selected_metric].mean(), cluster_avg, simulated_val], palette='viridis')
    st.pyplot(fig)

    st.subheader("🤝 Peer Group Benchmarking")
    st.write(f"Countries most similar to **{target_country}** in Cluster {cluster_id}:")

    # Simple logic to find peers in the same cluster
    peers = df[df['KMeans_Cluster'] == cluster_id][['Country', 'GDP', 'Life Expectancy Average', 'Internet Usage']]
    # Excluding the selected country itself
    peers = peers[peers['Country'] != target_country].head(5)

    st.table(peers)
    st.caption("Strategy: Analyze the 'Success Stories' in this list to replicate growth patterns.")

    # --- 4. MARKET OPPORTUNITY DISCOVERY ---
    st.subheader("🎯 High-Potential Market Discovery")

    
    selected_metric_gap = st.selectbox("Select Metric for Gap Analysis:", 
                                    ['Internet Usage', 'Energy Usage', 'GDP'])

   
    cluster_avg_gap = df[df['KMeans_Cluster'] == cluster_id][selected_metric_gap].mean()
    individual_val = country_data[selected_metric_gap]

    
    gap = cluster_avg_gap - individual_val

    st.write(f"### Analysis for {target_country}")

    if gap > 0:
        st.success(f"**Opportunity Identified!**")
        st.write(f"{target_country} is currently **{gap:.1f}% below** its peer group average in {selected_metric_gap}.")
        st.write("💡 **Business Insight:** There is significant room for growth in this sector to match cluster competitors.")
    else:
        st.info(f"{target_country} is already performing above its peer group average in {selected_metric_gap}.")

    
    fig_gap, ax_gap = plt.subplots(figsize=(10, 4))
    comparison_df = pd.DataFrame({
        'Metric': [f'{target_country}', 'Cluster Average', 'Global Average'],
        'Value': [individual_val, cluster_avg_gap, df[selected_metric_gap].mean()]
    })
    sns.barplot(data=comparison_df, x='Value', y='Metric', palette='magma', ax=ax_gap)
    ax_gap.axvline(cluster_avg_gap, color='red', linestyle='--', label='Target (Cluster Mean)')
    st.pyplot(fig_gap)


# # ==========================================
# 4. MODEL COMPARISON PAGE
# ==========================================
elif page == "Model Comparison":
    st.title("⚖️ Model Performance Comparison")
    
       
    st.subheader("Silhouette Analysis")
    
    comparison_data = {
        'Model': ['K-Means', 'Hierarchical','Divisive','DBSCAN' ,'HDBSCAN','GMM','Spectral'],
        'Silhouette Score': [0.228, 0.281, 0.253, -0.097, 0.631, 0.198, 0.224] 
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Model', y='Silhouette Score', data=comparison_df.sort_values('Silhouette Score', ascending=False), palette='magma', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    
    # Structure bands
    plt.axhspan(0.7, 1.0, facecolor='green', alpha=0.1, label='Strong')
    plt.axhspan(0.5, 0.7, facecolor='yellow', alpha=0.1, label='Medium')
    plt.axhspan(0.2, 0.5, facecolor='orange', alpha=0.1, label='Weak')
    
    st.pyplot(fig)
    st.dataframe(comparison_df.sort_values('Silhouette Score', ascending=False))
