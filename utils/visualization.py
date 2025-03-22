# utils/visualization.py
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions(historical_df, prediction_df, output_path='data/predictions_plot.png'):
    plt.figure(figsize=(12, 8))
    products = prediction_df['predicted_product_id'].unique()
    
    for product_id in products:
        # Work with copies to avoid warnings
        hist_data = historical_df[historical_df['Product_ID'] == product_id].copy()
        pred_data = prediction_df[prediction_df['predicted_product_id'] == product_id].copy()
        
        # Ensure dates are datetime
        hist_data.loc[:, 'Date'] = pd.to_datetime(hist_data['Date'])
        pred_data.loc[:, 'predicted_date'] = pd.to_datetime(pred_data['predicted_date'])
        
        plt.plot(hist_data['Date'], hist_data['Demand'], label=f"{pred_data['predicted_product_name'].iloc[0]} Historical", alpha=0.5)
        plt.plot(pred_data['predicted_date'], pred_data['predicted_demand'], label=f"{pred_data['predicted_product_name'].iloc[0]} Predicted", linestyle='--')
    
    plt.title('Historical Demand vs Predicted Demand (Top 3)')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")