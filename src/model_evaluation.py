import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(data_path, model_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The data file at {data_path} does not exist.")
    
    df = pd.read_csv(data_path)
    
    # Ensure 'Churn' column exists in the DataFrame
    if 'Churn' not in df.columns:
        raise ValueError("'Churn' column is missing from the data.")
    
    # Drop 'customerID' if it exists in the DataFrame
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X)

    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))

if __name__ == '__main__':
    input_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/data/processed/encoded_data.csv'
    model_file = 'C:/Users/Mayur/OneDrive/Documents/INTERNSHIP/ChurnChampion/models/churn_model.pkl'
    
    evaluate_model(input_file, model_file)
