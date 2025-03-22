import pandas as pd

def analyze_shop_activity(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Group by shop_id and Product_ID, summing Historical_Sales
    sales_summary = df.groupby(['shop_id', 'Product_ID'])['Historical_Sales'].sum().reset_index()
    
    # For each Product_ID, find the shop with the maximum Historical_Sales
    most_active_shops = sales_summary.loc[sales_summary.groupby('Product_ID')['Historical_Sales'].idxmax()]
    
    # Sort by Product_ID for readability
    most_active_shops = most_active_shops.sort_values('Product_ID')
    
    # Reset index for clean output
    most_active_shops = most_active_shops.reset_index(drop=True)
    
    # Rename columns for clarity
    most_active_shops.columns = ['Shop_ID', 'Product_ID', 'Total_Historical_Sales']
    
    # Print results
    print("Most Active Shops by Product (based on Total Historical Sales):")
    print(most_active_shops.to_string(index=False))
    print(f"\nTotal rows analyzed: {len(df)}")
    
    # Save to CSV
    output_file = 'data/most_active_shops_by_product.csv'
    most_active_shops.to_csv(output_file, index=False)
    print(f"\nResults saved to '{output_file}'")
    
    return most_active_shops

if __name__ == "__main__":
    # File path to the multi-shop data
    input_file = 'data/multi_shop_data.csv'
    
    # Analyze and display results
    analyze_shop_activity(input_file)