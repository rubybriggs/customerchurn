import os
import boto3
import sagemaker
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-s3-uri", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"--- Registering Model from: {args.model_s3_uri} ---")
    
    region = "us-east-1"
    sm_client = boto3.client("sagemaker", region_name=region)
    
    # Get the XGBoost Container Image URI
    image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")
    
    print("Creating Model Package in Registry...")
    response = sm_client.create_model_package(
        ModelPackageGroupName="customer-churn-model-group",
        ModelPackageDescription="Orchestration Pipeline Model (Automated)",
        ModelApprovalStatus="Approved",
        InferenceSpecification={
            "Containers": [{"Image": image_uri, "ModelDataUrl": args.model_s3_uri}],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge", "ml.t2.medium"],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"]
        }
    )
    print(f"✅ Success! Model Package ARN: {response['ModelPackageArn']}")

if __name__ == "__main__":
    main()