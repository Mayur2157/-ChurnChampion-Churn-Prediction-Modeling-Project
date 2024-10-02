import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

def train_model(data_path, model_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data file at {data_path} does not exist.")
    
    df = pd.read_csv(data_path)
    
    # Ensure 'Churn' column exists in the DataFrame
    if 'Churn' not in df.columns:
        raise ValueError("'Churn' column is missing from the data.")
    
    # Drop 'customerID' for training
    if 'customerID' in df.columns:
        X = df.drop(['Churn', 'customerID'], axis=1)
    else:
        X = df.drop('Churn', axis=1)
    
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to: {model_path}")

if __name__ == '__main__':
    # Use absolute paths or correct relative paths
    input_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/processed/encoded_data.csv'
    output_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/models/churn_model.pkl'
    
    train_model(input_file, output_file)
