import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def encode_features(input_path, output_path):
    try:
        # Print current working directory
        print(f"Current Working Directory: {os.getcwd()}")

        # Load data
        df = pd.read_csv(input_path)

        # Select categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Initialize LabelEncoder
        le = LabelEncoder()

        # Encode categorical features except for 'customerID'
        for col in categorical_cols:
            if col != 'customerID':
                df[col] = le.fit_transform(df[col])
        
        # Save encoded data
        df.to_csv(output_path, index=False)
        print(f"Feature encoding completed. Processed data saved to: {output_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check if the file exists at the specified path: {input_path}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Use absolute paths to avoid path issues
    input_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/processed/processed_data.csv'
    output_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/processed/encoded_data.csv'
    
    encode_features(input_file, output_file)
