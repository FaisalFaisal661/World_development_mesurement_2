# pages/1_Exploration.py
import streamlit as st
import plotly.express as px
from data_loader import load_data_from_drive

st.title("Data Exploration & Visualization")

if 'drive_url' in st.session_state:
    df = load_data_from_drive(st.session_state['drive_url'])
    
    st.subheader("Dataset Summary")
    st.write(df.head())
    
    # Visualizations from your notebook
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Feature Distribution")
        feature = st.selectbox("Select Feature", df.columns)
        fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
        st.plotly_chart(fig)

    with col2:
        st.write("#### Relationship Analysis")
        x_axis = st.selectbox("X Axis", df.columns, index=0)
        y_axis = st.selectbox("Y Axis", df.columns, index=1)
        fig_scatter = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_scatter)
else:
    st.warning("Please provide a data link on the Main Page.")