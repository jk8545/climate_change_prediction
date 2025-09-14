import streamlit as st
import pandas as pd
import joblib

# --- 1. LOAD SAVED ARTIFACTS ---
# Load the trained model and the preprocessor dictionary
model = joblib.load('climate_model.joblib')
preprocessor = joblib.load('data_preprocessor.joblib')

# Extract the individual components from the preprocessor dictionary
scaler = preprocessor['scaler']
scaler_columns = preprocessor['scaler_columns']
model_features = preprocessor['model_features']


# --- 2. SET UP THE STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Climate Risk Predictor", layout="centered")
st.title("🌍 Climate Risk Score Predictor")
st.write("""
This app predicts a **Climate Risk Score** (from 0 to 1), where higher scores indicate greater environmental risk. 
Provide the following key metrics to get a prediction.
""")

# --- 3. GET USER INPUTS VIA A FORM ---
with st.form("prediction_form"):
    st.header("Enter Climate Data:")
    
    # Create sliders for each feature the model needs
    # The default values are set to be reasonable averages
    sea_level = st.slider("Sea Level Rise (mm)", min_value=1400, max_value=1700, value=1550, step=1)
    weather_events = st.slider("Number of Extreme Weather Events (Annual)", min_value=0, max_value=20, value=7, step=1)
    temp_f = st.slider("Average Temperature (°F)", min_value=20.0, max_value=100.0, value=68.0, step=0.1)
    year = st.slider("Year", min_value=2025, max_value=2050, value=2030, step=1)
    
    # The model was trained on both F and C, so we calculate Celsius
    temp_c = (temp_f - 32) * 5/9
    
    # The submit button for the form
    submit_button = st.form_submit_button(label='Predict Risk Score')


# --- 4. PROCESS INPUTS AND PREDICT WHEN THE FORM IS SUBMITTED ---
if submit_button:
    # 4.1. Create a dictionary from the user's inputs
    user_inputs = {
        'Sea_Level_mm': sea_level,
        'Extreme_Weather_Events': weather_events,
        'Temperature_Fahrenheit': temp_f,
        'Year': year,
        'Temperature_Celsius': temp_c
    }
    
    # 4.2. Build the DataFrame for the scaler
    # This DataFrame MUST have all the columns the scaler was trained on, in the correct order.
    # We fill it with the user's inputs and use 0 for any other columns.
    scaler_input_df = pd.DataFrame(columns=scaler_columns)
    scaler_input_df.loc[0] = 0 # Initialize row with zeros
    for key, value in user_inputs.items():
        if key in scaler_input_df.columns:
            scaler_input_df[key] = value

    # 4.3. Scale the data
    # The scaler returns a NumPy array, so we convert it back to a DataFrame
    scaled_values = scaler.transform(scaler_input_df)
    scaled_df = pd.DataFrame(scaled_values, columns=scaler_columns)

    # 4.4. Select only the features required by the model
    model_input_df = scaled_df[model_features]
    
    # 4.5. Make the prediction
    prediction = model.predict(model_input_df)
    predicted_score = prediction[0]
    
    # 4.6. Display the result
    st.subheader("Prediction Result")
    st.metric(label="Predicted Climate Risk Score", value=f"{predicted_score:.3f}")

    if predicted_score < 0.33:
        st.success("The predicted climate risk is LOW.")
    elif predicted_score < 0.66:
        st.warning("The predicted climate risk is MODERATE.")
    else:
        st.error("The predicted climate risk is HIGH.")