import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput  # <--- ADD THIS IMPORT
import boto3

# Configuration
session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
bucket = "company-mlops-churn-production-106679514998"
region = "us-east-1"

# Use the data we already prepared for the HPO job!
train_data = f"s3://{bucket}/hpo-data/train/"
val_data = f"s3://{bucket}/hpo-data/validation/"

print("--- Starting Native SageMaker Training ---")

# 1. Get the exact XGBoost 1.5-1 Container Image
image_uri = sagemaker.image_uris.retrieve("xgboost", region, version="1.5-1")

# 2. Create the Estimator
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    output_path=f"s3://{bucket}/native-training-output/",
    sagemaker_session=session,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": "100",
        "max_depth": "5",
        "eta": "0.1",
        "eval_metric": "auc"
    }
)

# 3. CRITICAL FIX: Define inputs as CSV
train_input = TrainingInput(s3_data=train_data, content_type='text/csv')
val_input = TrainingInput(s3_data=val_data, content_type='text/csv')

# 4. Run the Training Job
print("Launching training job... (Wait 5-10 mins)")
estimator.fit({"train": train_input, "validation": val_input})

print(f"Training Complete! Model saved to: {estimator.model_data}")

# 5. Register this perfect model in the Registry
print("--- Registering Model ---")
sm_client = boto3.client("sagemaker", region_name=region)
model_package_group = "customer-churn-model-group"

response = sm_client.create_model_package(
    ModelPackageGroupName=model_package_group,
    ModelPackageDescription="Native XGBoost Model (Format Guaranteed Compatible)",
    ModelApprovalStatus="Approved",
    InferenceSpecification={
        "Containers": [{"Image": image_uri, "ModelDataUrl": estimator.model_data}],
        "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
        "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge", "ml.t2.medium"],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    }
)

print(f"✅ Model Registered: {response['ModelPackageArn']}")