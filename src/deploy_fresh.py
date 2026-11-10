import sagemaker
from sagemaker.model import ModelPackage
import boto3
import time

session = sagemaker.Session()
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
region = "us-east-1"
MODEL_PACKAGE_GROUP = "customer-churn-model-group"

# Use UNIQUE endpoint name to avoid all conflicts
TIMESTAMP = int(time.time())
ENDPOINT_NAME = f"churn-endpoint-{TIMESTAMP}"

sm_client = boto3.client("sagemaker", region_name=region)

# Get latest approved model
response = sm_client.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)

package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"Using model: {package_arn}")

# Create model
model_name = f"churn-model-{TIMESTAMP}"
sm_client.create_model(
    ModelName=model_name,
    Containers=[{"ModelPackageName": package_arn}],
    ExecutionRoleArn=role
)

# Create endpoint config
config_name = f"churn-config-{TIMESTAMP}"
sm_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.xlarge"
    }]
)

# Create endpoint
print(f"Creating endpoint: {ENDPOINT_NAME}")
sm_client.create_endpoint(
    EndpointName=ENDPOINT_NAME,
    EndpointConfigName=config_name
)

print("Waiting for endpoint (5-10 mins)...")
print(f"Endpoint Name: {ENDPOINT_NAME}")
print("Check SageMaker Console > Endpoints to monitor")

try:
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME, WaiterConfig={'Delay': 30, 'MaxAttempts': 40})
    print(f"\n✅ SUCCESS! {ENDPOINT_NAME} is InService!")
except Exception as e:
    print(f"\n❌ Failed: {e}")
    print("Check CloudWatch Logs for the endpoint")