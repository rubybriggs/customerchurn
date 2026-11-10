import os
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input/customer_churn.csv")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Reading input dataset from {args.input_data}...")
    df = pd.read_csv(args.input_data)
    
    # Clean numeric features
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(float)
    df['tenure'] = df['tenure'].astype(int)
    
    # Encode target label
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    elif 'churn' in df.columns:
        df['churn'] = df['churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # One-hot encode categorical features
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    existing_cats = [c for c in categorical_cols if c in df.columns]
    df_encoded = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
    # Remove special characters from column headers
    df_encoded.columns = [
        c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        for c in df_encoded.columns
    ]
    
    # Drop primary key and identifier columns
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['customerID'])
    if 'customerid' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['customerid'])
    
    # Train/Test Split (80/20)
    train_df, test_df = np.split(df_encoded.sample(frac=1, random_state=42), [int(0.8 * len(df_encoded))])
    
    os.makedirs(f"{args.output_dir}/train", exist_ok=True)
    os.makedirs(f"{args.output_dir}/test", exist_ok=True)
    
    train_df.to_csv(f"{args.output_dir}/train/train.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test/test.csv", index=False)
    
    print(f"Data preprocessing complete. Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    main()