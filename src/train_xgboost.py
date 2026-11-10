import os
import time
import tarfile
import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# 1. Initialize AWS & SageMaker Sessions
session = sagemaker.Session()
boto_session = boto3.Session(region_name="us-east-1")
sm_client = boto_session.client("sagemaker")
s3_client = boto_session.client("s3")

REGION = "us-east-1"
# FIXED: Corrected bucket name and removed spaces
BUCKET = "company-mlops-churn-production-106679514998"
FEATURE_GROUP_NAME = "customer-churn-feature-group"
MODEL_PACKAGE_GROUP_NAME = "customer-churn-model-group"

def extract_features_from_offline_store():
    print(f"Querying Offline Store from Feature Group: '{FEATURE_GROUP_NAME}'...")
    fg = FeatureGroup(name=FEATURE_GROUP_NAME, sagemaker_session=session)
    query = fg.athena_query()
    table_name = query.table_name
    
    # Query the table
    query_string = f'SELECT * FROM "{table_name}"'
    output_location = f"s3://{BUCKET}/athena-results/"
    
    print(f"Running Athena Query on table: {table_name}")
    query.run(query_string=query_string, output_location=output_location)
    query.wait()
    
    df = query.as_dataframe()
    print(f"Data retrieved successfully. Total rows/cols: {df.shape}")
    return df

def train_xgboost_model(df):
    print("--- Cleaning Schema and Engineering Training Sets ---")
    
    # Print columns to debug (optional, but helpful)
    print(f"Columns found in DataFrame: {list(df.columns)[:5]}...") # Just printing first 5
    
    # FIX: Use lowercase names because Athena/Glue lowercases them
    # "EventTime" -> "eventtime", "customerID" -> "customerid"
    metadata_cols = ["eventtime", "write_time", "api_invocation_time", "is_deleted", "customerid"]
    
    # Drop metadata columns safely
    cols_to_drop = [col for col in metadata_cols if col in df.columns]
    print(f"Dropping metadata columns: {cols_to_drop}")
    df_clean = df.drop(columns=cols_to_drop)
    
    # FIX: Look for lowercase "churn"
    target_col = "churn"
    if target_col not in df_clean.columns:
        # Fallback if it happens to be capitalized
        if "Churn" in df_clean.columns:
            target_col = "Churn"
        else:
            raise KeyError(f"Target column 'churn' or 'Churn' not found in columns: {df_clean.columns}")

    # Separate target variable from features
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].astype(int)
    
    print(f"Training feature dimension count: {X.shape[1]}")
    
    # CRITICAL: Strip feature column string names by fitting on raw NumPy arrays.
    X_matrix = X.values
    y_vector = y.values
    
    print("--- Training XGBoost Candidate Model ---")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss"
    )
    model.fit(X_matrix, y_vector)
    print("XGBoost Training Complete!")
    return model

def package_and_upload_model(model):
    print("--- Packaging Model Artifact for AWS Serving Container ---")
    local_dir = "model_artifact"
    os.makedirs(local_dir, exist_ok=True)
    
    model_file_path = os.path.join(local_dir, "xgboost-model")
    
    # CRITICAL FIX: Force TEXT JSON format (not UBJSON)
    # The SageMaker container requires .json extension to use text format
    booster = model.get_booster()
    
    # Save with .json extension to force text JSON format
    temp_path = model_file_path + ".json"
    booster.save_model(temp_path)
    
    # Rename to 'xgboost-model' (no extension) as required by SageMaker
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
    os.rename(temp_path, model_file_path)
    
    print(f"Model saved as TEXT JSON: {os.path.getsize(model_file_path)} bytes")
    
    # Create tar.gz
    tar_filename = "model.tar.gz"
    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(model_file_path, arcname="xgboost-model")
    
    # Verify contents
    print("Contents of model.tar.gz:")
    with tarfile.open(tar_filename, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"  {member.name} ({member.size} bytes)")
    
    # Upload to S3
    s3_key = "models/xgboost-model.tar.gz"
    s3_client.upload_file(tar_filename, BUCKET, s3_key)
    s3_uri = f"s3://{BUCKET}/{s3_key}"
    print(f"Uploaded to: {s3_uri}")
    
    return s3_uri

def register_model_package(s3_artifact_uri):
    print("--- Registering Model in SageMaker Model Registry ---")
    
    # Ensure Model Package Group exists
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription="Customer Churn XGBoost Production Model Package Group"
        )
        print(f"Created Model Package Group: {MODEL_PACKAGE_GROUP_NAME}")
    except sm_client.exceptions.ClientError:
        print(f"Model Package Group '{MODEL_PACKAGE_GROUP_NAME}' already exists.")
        
    # Retrieve standard AWS XGBoost Container Image URI
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=REGION,
        version="1.5-1"
    )
    
    create_model_package_input = {
        "ModelPackageGroupName": MODEL_PACKAGE_GROUP_NAME,
        "ModelPackageDescription": "Production XGBoost churn classifier fitted on raw NumPy arrays",
        "ModelApprovalStatus": "Approved",
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": s3_artifact_uri
                }
            ],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge", "ml.t2.medium"],
            "SupportedContentTypes": ["text/csv", "application/json"],
            "SupportedResponseMIMETypes": ["text/csv", "application/json"]
        }
    }
    
    response = sm_client.create_model_package(**create_model_package_input)
    model_package_arn = response["ModelPackageArn"]
    print(f"Model Package Successfully Registered!")
    print(f"Model Package ARN: {model_package_arn}")
    return model_package_arn

if __name__ == "__main__":
    # Execute the pipeline
    df = extract_features_from_offline_store()
    model = train_xgboost_model(df)
    s3_uri = package_and_upload_model(model)
    register_model_package(s3_uri)