import os
import time
import pandas as pd
import numpy as np

def run_etl_pipeline(input_local_path: str, output_local_path: str):
    print("="*50)
    print("STARTING ETL PIPELINE")
    print("="*50)
    
    # 1. SAFETY CHECK: Verify the input file exists and has data
    if not os.path.exists(input_local_path):
        raise FileNotFoundError(f"CRITICAL: Input file not found at {input_local_path}")
    
    file_size = os.path.getsize(input_local_path)
    print(f"Input file found. Size: {file_size} bytes")
    
    if file_size == 0:
        raise ValueError("CRITICAL: Input file is 0 bytes (empty). Check your S3 source data.")

    # 2. Read CSV with maximum compatibility
    print(f"Loading raw data from: {input_local_path}")
    try:
        # error_bad_lines=False works on almost all Pandas versions to skip bad rows
        df = pd.read_csv(input_local_path, engine='python', error_bad_lines=False)
    except Exception as e:
        # Fallback for very new Pandas versions where error_bad_lines was removed
        df = pd.read_csv(input_local_path, engine='python', on_bad_lines='skip')
        
    print(f"Successfully loaded dataframe. Shape: {df.shape}")

    # 3. Standardize Primary Identifiers & Types
    if 'customerID' not in df.columns:
        print("Warning: 'customerID' not found. Generating default IDs.")
        df['customerID'] = [f'CUST-{i:06d}' for i in range(len(df))]
    else:
        df['customerID'] = df['customerID'].astype(str).str.strip()

    # 4. Clean & Coerce Numerical Features
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip().replace('', np.nan)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    
    if 'tenure' in df.columns:
        # Safe conversion to integer
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)

    # 5. Binary Label Encoding
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

    # 6. One-Hot Encoding for Categorical Features
    categorical_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    
    existing_cats = [col for col in categorical_cols if col in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    # Clean feature names for AWS Glue/Feature Store compatibility
    df.columns = [
        c.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        for c in df.columns
    ]

    # 7. SageMaker Feature Store Required Metadata
    df['EventTime'] = time.time()
    df['customerID'] = df['customerID'].astype(str)

    print(f"ETL transformations complete. Final shape: {df.shape}")

    # 8. Save to LOCAL output path 
    # SageMaker Processing will automatically upload this folder to your S3 output URI
    print(f"Saving processed data to local path: {output_local_path}")
    df.to_csv(output_local_path, index=False)
    print("Successfully saved processed data locally.")

if __name__ == "__main__":
    # These paths are mounted by the SageMaker ProcessingJob configuration
    LOCAL_INPUT = "/opt/ml/processing/input/CustomerChurn.csv"
    LOCAL_OUTPUT = "/opt/ml/processing/output/customer_churn_processed.csv"
    
    run_etl_pipeline(LOCAL_INPUT, LOCAL_OUTPUT)