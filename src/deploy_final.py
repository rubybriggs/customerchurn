import sagemaker
from sagemaker.model import Model

session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
bucket = "company-mlops-churn-production-106679514998"
region = "us-east-1"

# Use XGBoost 1.5-1 container
image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

model = Model(
    image_uri=image_uri,
    model_data=f"s3://{bucket}/models/xgboost-model.tar.gz",
    role=role,
    sagemaker_session=session
)

print("Deploying endpoint with ml.m5.xlarge (this takes 5-10 mins)...")

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.xlarge",
    endpoint_name="customer-churn-endpoint-prod"
)

print(f"✅ SUCCESS! Endpoint {predictor.endpoint_name} is InService!")