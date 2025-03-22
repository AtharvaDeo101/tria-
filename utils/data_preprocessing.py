# utils/data_preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def preprocess_data(df):
    # Features: Historical_Sales, Promotion, Day_of_Week, Month, Product_ID (encoded)
    features = ['Historical_Sales', 'Promotion', 'Day_of_Week', 'Month', 'Product_ID']
    X = df[features]
    y = df['Demand']
    
    # One-hot encode Product_ID
    X = pd.get_dummies(X, columns=['Product_ID'], drop_first=True)
    
    # Scale features but keep as DataFrame
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler