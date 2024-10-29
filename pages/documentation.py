# pages/2_Documentation.py

import streamlit as st

def main():
    st.title("Project Documentation")

    st.header("1. Project Overview")
    st.markdown("""
    **"AI-Driven Predictive Maintenance for Industrial Turbofan Engines"** is a project aimed at predicting the Remaining Useful Life (RUL) of turbofan engines. By leveraging historical operational and sensor data, this project enables proactive maintenance scheduling by predicting when an engine component might fail, helping reduce downtime and maintenance costs.
    """)

    st.header("2. Purpose of This Application")
    st.markdown("""
    This application allows users to:
    - **Visualize engine sensor data** for different units and observe patterns that may indicate wear or degradation.
    - **Predict the Remaining Useful Life (RUL)** for each engine based on current operational data, enabling better maintenance planning.
    - **Simulate custom sensor inputs** to explore how different operational conditions might affect the engine’s RUL.

    This tool is especially useful for maintenance teams, engineers, and data scientists in industries that rely on machinery with a high impact of downtime, such as aviation and energy.
    """)

    st.header("3. Dataset Description")
    st.markdown("""
    The model was trained using the **NASA Turbofan Engine Degradation Simulation** dataset. This dataset includes:
    - **Multiple engine units** tracked over time until failure.
    - **Operational settings** and **sensor readings** that reflect different engine performance metrics.
    - **Remaining Useful Life (RUL)** labels, which indicate how many cycles remain before each engine's failure.

    Each engine unit’s data consists of various cycles representing its operational lifespan, providing a historical basis to predict RUL.
    """)

    st.header("4. Elements and Features in the Application")
    st.markdown("""
    ### Home Page Features
    The Home page serves as the main interactive dashboard where users can:
    
    - **Select Engine Unit**:
      - Use the dropdown to choose a specific engine unit to view its data. Each unit represents a distinct engine with a unique history and degradation pattern.

    - **Sensor Data Trends**:
      - This section provides a time-series plot of selected sensors for the chosen engine unit. By visualizing sensor trends over operational cycles, users can observe patterns or anomalies that might indicate degradation.
      - Users can select specific sensors from a list to include in the plot.

    - **Predicted Remaining Useful Life (RUL)**:
      - This is the core feature of the application. Based on the latest sensor readings, the model predicts the Remaining Useful Life of the selected engine unit in cycles. A lower RUL indicates that the engine is closer to failure.
      - This prediction helps users understand how much operational time is left before the engine likely requires maintenance.

    - **Custom Sensor Input for RUL Prediction**:
      - This feature allows users to manually adjust sensor values and operational settings to simulate different conditions and see how these changes affect the RUL.
      - For example, users can increase or decrease sensor values related to temperature or vibration to test how the engine might perform under different conditions.
      - After selecting custom values, clicking "Predict RUL" displays the new RUL prediction for the hypothetical scenario.

    ### Sidebar Options
    - **Engine Unit Selection**:
      - The sidebar provides options to select an engine unit and adjust custom sensor inputs.
    
    - **About**:
      - Provides a brief description of the application and its purpose.

    ### Documentation Page
    - This page (Documentation) explains the application’s purpose, usage instructions, and an overview of the dataset and features.
    """)

    st.header("5. Understanding Each Feature and Sensor")
    st.markdown("""
    ### Operational Settings
    - **op_setting_1**: Represents general environmental settings, such as altitude. Higher or lower values could impact engine performance.
    - **op_setting_2**: May reflect pressure-related parameters influencing engine efficiency.
    - **op_setting_3**: Indicates temperature-related factors or other environmental conditions.

    ### Sensor Readings
    - Each sensor captures specific performance metrics that help monitor engine health. Here are some examples:
      - **sensor_1**: Likely captures engine speed, which affects wear and tear.
      - **sensor_2**: Could measure torque, a critical factor in mechanical stress.
      - **sensor_21**: May capture vibration data, which often indicates mechanical degradation.
    
    By understanding these sensors and settings, users can interpret how changes in these values impact the engine’s RUL.
    """)

    st.header("6. How to Use This Application")
    st.markdown("""
    - **Step 1: Select Engine Unit**:
      - From the sidebar, choose the engine unit you want to examine. This will update the main dashboard to display data and predictions specific to that unit.

    - **Step 2: View Sensor Data Trends**:
      - In the "Sensor Data Trends" section, select the sensors you wish to plot and view time-series data. This helps visualize how certain metrics evolve over time for the selected engine.

    - **Step 3: Check Predicted RUL**:
      - The predicted RUL value for the chosen unit appears in the "Predicted Remaining Useful Life" section, indicating how much operational time is left before maintenance is likely needed.

    - **Step 4: Simulate Custom Scenarios**:
      - In the sidebar, adjust sensor readings and operational settings to simulate various scenarios. After making your adjustments, click "Predict RUL" to see how these changes affect the engine’s remaining life.
    """)

    st.header("7. Future Enhancements")
    st.markdown("""
    In the future, this application could incorporate additional features to enhance its usability and accuracy:
    - **Model Optimization**: Experiment with alternative models and hyperparameters to improve prediction accuracy.
    - **Real-Time Data Integration**: Allow the application to receive live sensor data for real-time RUL predictions.
    - **Additional Visualizations**: Add more interactive plots and dashboards to enhance data exploration and provide a clearer view of engine health.
    - **User Authentication**: Secure the application by adding a login feature to restrict access to authorized personnel.
    """)

if __name__ == "__main__":
    main()
