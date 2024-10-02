import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load your dataset
df = pd.read_csv('data/processed/processed_data.csv')

# Specify categorical columns
categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']  # Adjust based on your dataset

# Create a dictionary to hold label encoders
label_encoders = {}

# Create and fit LabelEncoders
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

# Save label encoders to a file
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Label encoders saved successfully.")
