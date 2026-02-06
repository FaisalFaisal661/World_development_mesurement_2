# pages/2_Prediction.py
import streamlit as st
import joblib
# import your trained model here

st.title("ML Model Predictor")

# Sidebar or main widgets for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    # Example for Titanic/Classification
    age = st.sidebar.slider('Age', 0, 100, 25)
    fare = st.sidebar.number_input('Fare', 0.0, 500.0, 32.0)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    # Add other widgets based on your notebook features
    return {"Age": age, "Fare": fare, "Sex": sex}

input_df = user_input_features()
st.subheader("User Input Parameters")
st.write(input_df)

if st.button("Predict"):
    # Here you would load your .pkl model and run .predict()
    # model = joblib.load('model.pkl')
    # prediction = model.predict(input_df)
    st.write("Clicking this would run your XGBoost or Logistic Regression model.")
    st.success("Result: Survival / Price Prediction would appear here.")