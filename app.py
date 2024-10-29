import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the trained model and scaler
model = joblib.load('models/rul_model.pkl')
scaler = joblib.load('models/scaler.save')

# Define sensors to drop
drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Load and preprocess the test data
test_data_path = 'CMaps/test_FD001.txt'
columns = ['unit_number', 'time_in_cycles'] + \
          [f'op_setting_{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]
test_df = pd.read_csv(test_data_path, sep='\s+', header=None, names=columns)
test_df = test_df.drop(columns=drop_sensors)

# Select features for prediction
features = [col for col in test_df.columns if col.startswith('op_setting_') or col.startswith('sensor_')]
test_df[features] = scaler.transform(test_df[features])

# App title
st.title("AI-Driven Predictive Maintenance for Turbofan Engines")

# Sidebar for user input
st.sidebar.header("Select Engine Unit")
unit_numbers = test_df['unit_number'].unique()
selected_unit = st.sidebar.selectbox("Engine Unit Number", unit_numbers)

# Get data for the selected unit
unit_data = test_df[test_df['unit_number'] == selected_unit].copy()

# Predict RUL for the selected unit (display RUL before sensor trends)
latest_data = unit_data[features].iloc[-1].values.reshape(1, -1)
predicted_rul = model.predict(latest_data)[0]
st.header(f"Predicted Remaining Useful Life (RUL) for Engine Unit {selected_unit}")
st.metric(label="Predicted RUL (in cycles)", value=f"{predicted_rul:.2f}")
st.write("""
This value represents the estimated number of cycles remaining before the engine may require maintenance. A lower RUL indicates the engine is closer to failure, allowing users to schedule maintenance proactively.
""")

# Display sensor trends with context
st.header(f"Sensor Data Trends for Engine Unit {selected_unit}")
st.write("""
The graph below shows the trends in selected sensor data over operational cycles for Engine Unit {}. Monitoring these trends helps identify changes that may impact the engine's health, providing valuable insights into degradation patterns.
""".format(selected_unit))

# Select sensors to plot and add y-axis description
sensor_options = [col for col in unit_data.columns if col.startswith('sensor_')]
selected_sensors = st.multiselect("Select Sensors to Plot", sensor_options, default=sensor_options[:3])

if selected_sensors:
    fig = px.line(unit_data, x='time_in_cycles', y=selected_sensors, 
                  title=f"Sensor Trends Over Time for Engine Unit {selected_unit}")
    fig.update_layout(
        xaxis_title="Operational Cycles",
        yaxis_title="Sensor Readings (Standardized Values)",
        legend_title="Sensors"
    )
    st.plotly_chart(fig)

# Add statistics for selected sensors
st.write("### Sensor Statistics")
st.write("""
Understanding statistical summaries for each sensor can provide insights into how individual readings contribute to overall engine health.
""")
for sensor in selected_sensors:
    sensor_data = unit_data[sensor]
    st.write(f"**{sensor}**")
    st.write(f"- Mean: {sensor_data.mean():.2f}")
    st.write(f"- Min: {sensor_data.min():.2f}")
    st.write(f"- Max: {sensor_data.max():.2f}")
    st.write(f"- Standard Deviation: {sensor_data.std():.2f}")

# Custom sensor input for RUL prediction
st.sidebar.header("Custom Sensor Input")
st.sidebar.write("Adjust sensor readings to predict RUL for hypothetical scenarios.")

user_input = {}
for feature in features:
    min_value = float(test_df[feature].min())
    max_value = float(test_df[feature].max())
    mean_value = float(test_df[feature].mean())
    
    # Handle cases where min_value and max_value are the same
    if min_value == max_value:
        st.sidebar.warning(f"Feature '{feature}' has a constant value across data.")
        user_value = st.sidebar.number_input(feature, value=mean_value)
    else:
        user_value = st.sidebar.slider(feature, min_value, max_value, mean_value)

    user_input[feature] = user_value

if st.sidebar.button("Predict RUL"):
    user_df = pd.DataFrame([user_input])
    user_predicted_rul = model.predict(user_df)[0]
    st.subheader("Predicted RUL for Custom Input")
    st.write(f"Predicted RUL: {user_predicted_rul:.2f} cycles")
    st.write("""
    This prediction reflects the estimated RUL if the engine were operating under the specified custom sensor conditions. It helps in evaluating how different operational scenarios impact engine longevity.
    """)

# About section in the sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses a machine learning model trained on the NASA Turbofan Engine Degradation Simulation dataset to predict the Remaining Useful Life (RUL) of turbofan engines.
Developed by Rishit Reddy.
""")
