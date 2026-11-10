import sagemaker
from sagemaker.model import ModelPackage
import boto3
import time

# Configuration
session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
region = "us-east-1"
MODEL_PACKAGE_GROUP = "customer-churn-model-group"
ENDPOINT_NAME = "customer-churn-endpoint-prod"

# Generate a unique name for the config to avoid collisions
TIMESTAMP = int(time.time())
UNIQUE_CONFIG_NAME = f"churn-config-{TIMESTAMP}"

sm_client = boto3.client("sagemaker", region_name=region)

# 1. Get the latest approved model package
print("--- Fetching Latest Approved Model Package ---")
response = sm_client.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

if not response["ModelPackageSummaryList"]:
    raise Exception("No approved model packages found!")

latest_package = response["ModelPackageSummaryList"][0]
package_arn = latest_package["ModelPackageArn"]
version = latest_package["ModelPackageVersion"]

print(f"Using Model Package Version {version}")

# 2. Create Model Object
print("\n--- Creating Model Object ---")
model_name = f"churn-model-{TIMESTAMP}"

# Create the model resource first
sm_client.create_model(
    ModelName=model_name,
    Containers=[{"ModelPackageName": package_arn}],
    ExecutionRoleArn=role
)
print(f"Created Model: {model_name}")

# 3. Create Endpoint Configuration (Unique Name)
print(f"\n--- Creating Endpoint Config: {UNIQUE_CONFIG_NAME} ---")
sm_client.create_endpoint_config(
    EndpointConfigName=UNIQUE_CONFIG_NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.t2.medium" # Using t2.medium for stability
    }]
)

# 4. Deploy or Update Endpoint
print(f"\n--- Deploying to Endpoint: {ENDPOINT_NAME} ---")
try:
    # Check if endpoint exists
    sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Endpoint exists. Updating with new config {UNIQUE_CONFIG_NAME}...")
    
    # Update existing endpoint
    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=UNIQUE_CONFIG_NAME
    )
    
except sm_client.exceptions.ClientError as e:
    if "Could not find" in str(e):
        print(f"Endpoint does not exist. Creating new endpoint...")
        # Create new endpoint
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=UNIQUE_CONFIG_NAME
        )
    else:
        raise

# 5. Wait for completion
print("Waiting for endpoint to be InService (this takes 5-10 mins)...")
waiter = sm_client.get_waiter('endpoint_in_service')
waiter.wait(EndpointName=ENDPOINT_NAME)

print(f"\n✅ SUCCESS!")
print(f"Endpoint: {ENDPOINT_NAME} is now InService!")