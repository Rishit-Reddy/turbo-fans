# train_model.py

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Define sensors to drop based on variance analysis
drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Set paths to your data files
train_data_path = 'CMaps/train_FD001.txt'
test_data_path = 'CMaps/test_FD001.txt'
rul_data_path = 'CMaps/RUL_FD001.txt'

# Load the data
def load_data(train_path, test_path, rul_path):
    columns = ['unit_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv(train_path, sep='\s+', header=None, names=columns)
    test_df = pd.read_csv(test_path, sep='\s+', header=None, names=columns)
    rul_df = pd.read_csv(rul_path, sep='\s+', header=None, names=['RUL'])
    
    return train_df, test_df, rul_df

# Calculate Remaining Useful Life (RUL) for training data
def calculate_rul(train_df):
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max()
    train_df['RUL'] = train_df.apply(lambda row: max_cycles[row['unit_number']] - row['time_in_cycles'], axis=1)
    return train_df

# Data preprocessing
def preprocess_data(train_df, test_df):
    # Drop the defined sensors
    train_df = train_df.drop(columns=drop_sensors)
    test_df = test_df.drop(columns=drop_sensors)

    # Features and target
    features = [col for col in train_df.columns if col.startswith('op_setting_') or col.startswith('sensor_')]
    target = 'RUL'
    
    # Normalize the data
    scaler = MinMaxScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    # Save the scaler for later use
    joblib.dump(scaler, 'models/scaler.save')
    
    return train_df, test_df, features, target

# Train the model
def train_model(train_df, features, target):
    X_train = train_df[features]
    y_train = train_df[target]
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'models/rul_model.pkl')
    
    return model

# Evaluate the model
def evaluate_model(model, test_df, rul_df, features):
    X_test = test_df.groupby('unit_number').last().reset_index()
    X_test = X_test[features]
    y_true = rul_df['RUL']
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"Model Evaluation Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    # Plot True vs Predicted RUL
    plt.figure(figsize=(10,6))
    plt.plot(y_true.values, label='True RUL')
    plt.plot(y_pred, label='Predicted RUL')
    plt.xlabel('Engine ID')
    plt.ylabel('RUL')
    plt.title('True vs Predicted RUL')
    plt.legend()
    plt.show()

# Main function
def main():
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load data
    train_df, test_df, rul_df = load_data(train_data_path, test_data_path, rul_data_path)
    print("Data loaded successfully.")
    
    # Calculate RUL for training data
    train_df = calculate_rul(train_df)
    print("Calculated RUL for training data.")
    
    # Preprocess data
    train_df, test_df, features, target = preprocess_data(train_df, test_df)
    print("Data preprocessing completed.")
    
    # Train model
    model = train_model(train_df, features, target)
    print("Model trained and saved successfully.")
    
    # Evaluate model
    evaluate_model(model, test_df, rul_df, features)

if __name__ == "__main__":
    main()
