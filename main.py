import pandas as pd
import numpy as np
from utils.data_preprocessing import load_data, preprocess_data, prepare_prediction_data
from models.demand_model import DemandForecastModel
from datetime import timedelta

def generate_predictions(file_path, forecast_days=3):
    # Load data
    df = load_data(file_path)

    try:
        # Preprocess data for training
        X, y, feature_scaler, target_scaler = preprocess_data(df)

        # Train the model
        model = DemandForecastModel(look_back=5, input_size=4)  # 4 features
        model.train(X, y)

        # Prepare data for prediction
        last_sequence, feature_scaler = prepare_prediction_data(df)
        last_sequence = np.reshape(last_sequence, (1, 5, 4))  # Shape: (1, look_back, n_features)

        # Predict future demand
        predictions = []
        current_sequence = last_sequence.copy()
        last_date = df['Date'].max()
        last_row = df.iloc[-1]

        for i in range(forecast_days):
            pred_demand = model.predict(current_sequence)
            pred_demand = target_scaler.inverse_transform(pred_demand)[0][0]
            predictions.append(pred_demand)

            # Update sequence with predicted demand and estimated features
            next_date = last_date + timedelta(days=i + 1)
            next_day_of_week = next_date.weekday()
            next_month = next_date.month
            
            # Assume no promotion and use last Historical_Sales
            next_features = np.array([[last_row['Historical_Sales'], 0, next_day_of_week, next_month]])
            next_features_scaled = feature_scaler.transform(next_features)
            
            # Shift and update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = next_features_scaled[0]

        # Generate output DataFrame
        future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
        most_common_product = df['Product_ID'].mode()[0]  # Most frequent product
        output_df = pd.DataFrame({
            'predicted_date': future_dates,
            'predicted_product_id': [most_common_product] * forecast_days,
            'predicted_demand': predictions
        })

        # Save predictions
        output_df.to_csv('data/predictions.csv', index=False)
        print("Predictions saved to 'data/predictions.csv'")
        return output_df

    except ValueError as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    sample_file = 'data/supply_chain_demand_data.csv'
    generate_predictions(sample_file)