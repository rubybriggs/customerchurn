from sagemaker.inputs import TrainingInput
import sagemaker
from sagemaker.tuner import HyperparameterTuner, IntegerParameter, ContinuousParameter
import pandas as pd
import boto3
import os

# 1. Configuration
session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
bucket = "company-mlops-churn-production-106679514998"
region = "us-east-1"

# S3 Paths
raw_data_uri = f"s3://{bucket}/processed/v1/customer_churn_processed.csv"
hpo_data_prefix = "hpo-data"

def prepare_data_for_sagemaker():
    print("--- Preparing Data for SageMaker XGBoost Container ---")
    print("Downloading processed data...")
    df = pd.read_csv(raw_data_uri)
    
    # Drop non-feature columns
    df = df.drop(columns=['customerid', 'customerID', 'eventtime', 'EventTime'], errors='ignore')
    
    # Find the target column
    target_col = 'churn' if 'churn' in df.columns else 'Churn'
    if target_col not in df.columns:
        raise KeyError("Could not find 'churn' or 'Churn' column!")
        
    print(f"Found target column: {target_col}")
    
    # SageMaker XGBoost requires Target Variable as the FIRST column
    cols = [c for c in df.columns if c != target_col]
    df = df[[target_col] + cols]
    
    # Split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Save locally without headers
    train_df.to_csv('train.csv', index=False, header=False)
    val_df.to_csv('validation.csv', index=False, header=False)
    
    # Upload to S3
    print("Uploading train/val data to S3...")
    s3_client = boto3.client('s3')
    # Note: We upload to a folder structure: hpo-data/train/train.csv
    s3_client.upload_file('train.csv', bucket, f"{hpo_data_prefix}/train/train.csv")
    s3_client.upload_file('validation.csv', bucket, f"{hpo_data_prefix}/validation/validation.csv")
    print("Data Ready.")

def run_hpo_job():
    print("--- Starting SageMaker Hyperparameter Tuning ---")
    
    # Define the Pre-built XGBoost Container (DLC)
    container = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")
    
    estimator = sagemaker.estimator.Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        output_path=f"s3://{bucket}/hpo-output/",
        sagemaker_session=session,
        hyperparameters={
            "objective": "binary:logistic",
            "num_round": "100",
            "eval_metric": "auc"
        }
    )

    # Define Hyperparameter Ranges
    hyperparameter_ranges = {
        "max_depth": IntegerParameter(3, 9),
        "eta": ContinuousParameter(0.01, 0.3),
        "min_child_weight": IntegerParameter(1, 10),
        "subsample": ContinuousParameter(0.5, 1.0),
        "colsample_bytree": ContinuousParameter(0.5, 1.0)
    }

    tuner = HyperparameterTuner(
        estimator,
        "validation:auc",
        hyperparameter_ranges,
        max_jobs=6,
        max_parallel_jobs=2,
        objective_type="Maximize"
    )

        # --- THE FIX: Use TrainingInput for newer SDK versions ---
    train_input = TrainingInput(
        s3_data=f"s3://{bucket}/{hpo_data_prefix}/train/",
        content_type='text/csv'
    )
    validation_input = TrainingInput(
        s3_data=f"s3://{bucket}/{hpo_data_prefix}/validation/",
        content_type='text/csv'
    )

    print("Launching Tuning Job...")
    # Pass the s3_input objects instead of raw strings
    tuner.fit({"train": train_input, "validation": validation_input})
    
    print(f"Tuning Job Name: {tuner.latest_tuning_job.job_name}")

if __name__ == "__main__":
    prepare_data_for_sagemaker()
    run_hpo_job()