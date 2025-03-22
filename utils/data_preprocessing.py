import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load and return the purchase history data."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def preprocess_data(df, look_back=5):
    """Preprocess data for LSTM model with multiple features."""
    feature_cols = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month']
    target_col = 'Demand'
    
    # Scale features and target separately
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features = feature_scaler.fit_transform(df[feature_cols].values)
    target = target_scaler.fit_transform(df[[target_col]].values)
    
    if len(features) <= look_back:
        raise ValueError(f"Data has {len(features)} rows, but at least {look_back + 1} rows are required for look_back={look_back}")
    
    # Create sequences
    X, y = [], []
    for i in range(len(features) - look_back):
        X.append(features[i:i + look_back])  # Shape: (look_back, n_features)
        y.append(target[i + look_back])      # Shape: (1,)
    
    X = np.array(X)  # Shape: (samples, look_back, n_features)
    y = np.array(y)  # Shape: (samples, 1)
    return X, y, feature_scaler, target_scaler

def prepare_prediction_data(df, look_back=5):
    """Prepare the most recent data for prediction."""
    feature_cols = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month']
    
    feature_scaler = MinMaxScaler()
    features = feature_scaler.fit_transform(df[feature_cols].values)
    
    if len(features) < look_back:
        raise ValueError(f"Data has {len(features)} rows, but at least {look_back} rows are required for look_back={look_back}")
    
    return features[-look_back:], feature_scaler