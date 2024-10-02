import pandas as pd
import os

def preprocess_data(input_path, output_path):
    # Print current working directory
    print(f"Current Working Directory: {os.getcwd()}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Handle missing values using ffill (forward fill)
    df = df.ffill()
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with the mean
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
    
    # Make sure we retain 'customerID' in this step
    print(f"Columns in DataFrame: {df.columns}")
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Data preprocessing completed. Processed data saved to: {output_path}")

if __name__ == '__main__':
    preprocess_data('C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/raw/customer_data.csv',
                    'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/processed/processed_data.csv')
