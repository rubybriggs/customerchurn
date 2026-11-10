import sagemaker
from sagemaker.model import Model
import boto3
import time

# Configuration
session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
bucket = "company-mlops-churn-production-106679514998"
region = "us-east-1"

# 1. Get the XGBoost Container Image
print("--- Getting XGBoost Container ---")
image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")
print(f"Using Image: {image_uri}")

# 2. Define the Model (points to the model.tar.gz we just uploaded)
model_data = f"s3://{bucket}/models/xgboost-model.tar.gz"

model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role,
    sagemaker_session=session
)

# 3. Deploy to Endpoint
print("\n--- Deploying Production Endpoint ---")
print("This will take 5-10 minutes...")

try:
    # Try to delete existing endpoint first
    sm_client = boto3.client('sagemaker', region_name=region)
    try:
        sm_client.delete_endpoint(EndpointName="customer-churn-endpoint-prod")
        print("Deleted old endpoint")
    except:
        pass
    
    try:
        sm_client.delete_endpoint_config(EndpointConfigName="customer-churn-endpoint-config-prod")
        print("Deleted old endpoint config")
    except:
        pass
except:
    pass

# Deploy new endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",  # More stable for testing
    endpoint_name="customer-churn-endpoint-prod"
)

print(f"\n✅ SUCCESS!")
print(f"Endpoint Name: {predictor.endpoint_name}")
print(f"Status: InService")
print(f"Ready for predictions!")