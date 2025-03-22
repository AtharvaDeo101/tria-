# main.py
import pandas as pd
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data
from utils.visualization import plot_predictions
from models.demand_model import DemandForecastModel
from datetime import timedelta

def generate_predictions(file_path, forecast_days=7):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(file_path)
    X, y, feature_scaler = preprocess_data(df)

    # Train the model
    print("Training demand forecast model...")
    model = DemandForecastModel()
    model.train(X, y)

    # Prepare future data for prediction
    print(f"Generating predictions for {forecast_days} days...")
    last_date = pd.to_datetime(df['Date']).max()
    products = df['Product_ID'].unique()
    all_predictions = []

    # Store feature names from training data
    feature_names = X.columns

    for product_id in products:
        product_df = df[df['Product_ID'] == product_id]
        last_row = product_df.iloc[-1]
        
        for i in range(forecast_days):
            next_date = last_date + timedelta(days=i + 1)
            next_day_of_week = next_date.weekday()
            next_month = next_date.month
            
            # Features for prediction (keep as DataFrame)
            future_features = pd.DataFrame({
                'Historical_Sales': [last_row['Historical_Sales']],
                'Promotion': [0],
                'Day_of_Week': [next_day_of_week],
                'Month': [next_month],
                'Product_ID': [product_id]
            })
            
            # Preprocess future data
            future_X = pd.get_dummies(future_features, columns=['Product_ID'], drop_first=True)
            future_X = future_X.reindex(columns=feature_names, fill_value=0)  # Align with training columns
            
            # Scale but keep as DataFrame with column names
            future_X_scaled = pd.DataFrame(feature_scaler.transform(future_X), columns=feature_names)
            
            # Predict demand
            pred_demand = model.predict(future_X_scaled)[0]
            
            # Store prediction
            all_predictions.append({
                'predicted_date': next_date,
                'predicted_product_id': product_id,
                'predicted_product_name': last_row['Product_name'],
                'predicted_demand': pred_demand
            })

    # Create output DataFrame
    output_df = pd.DataFrame(all_predictions)
    
    # Select top 3 predicted demands
    top_3_df = output_df.nlargest(3, 'predicted_demand')
    
    # Save predictions (format date as string for CSV)
    output_df_to_save = top_3_df.copy()
    output_df_to_save['predicted_date'] = output_df_to_save['predicted_date'].dt.strftime('%Y-%m-%d')
    output_file = 'data/predictions.csv'
    output_df_to_save.to_csv(output_file, index=False)
    print(f"Top 3 predictions saved to '{output_file}'")

    # Visualize predictions (only top 3)
    print("Generating visualization for top 3 predictions...")
    plot_predictions(df, top_3_df)

    return top_3_df

if __name__ == "__main__":
    input_file = 'data/multi_shop_data.csv'

    forecast_days = 3  # Kept as 3 per your run
    predictions = generate_predictions(input_file, forecast_days)
    print("\nTop 3 predictions by predicted_demand:")
    print(predictions)