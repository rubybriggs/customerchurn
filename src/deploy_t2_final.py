import boto3
import time

region = "us-east-1"
sm_client = boto3.client("sagemaker", region_name=region)
MODEL_PACKAGE_GROUP = "customer-churn-model-group"
ROLE = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# 1. Get latest model (Version 9)
print("Fetching latest approved model...")
response = sm_client.list_model_packages(
    ModelPackageGroupName=MODEL_PACKAGE_GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
    MaxResults=1
)
package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"Using Model: {package_arn}")

# 2. Create unique names to avoid collisions
timestamp = int(time.time())
model_name = f"churn-model-t2-{timestamp}"
config_name = f"churn-config-t2-{timestamp}"
endpoint_name = f"churn-endpoint-t2-{timestamp}"

# 3. Create Model
print("Creating Model resource...")
sm_client.create_model(
    ModelName=model_name,
    Containers=[{"ModelPackageName": package_arn}],
    ExecutionRoleArn=ROLE
)

# 4. Create Config with ml.t2.medium (The Fix for Capacity Issues)
print("Creating Endpoint Config with ml.t2.medium...")
sm_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.t2.medium"  # <--- THIS IS THE KEY FIX
    }]
)

# 5. Create Endpoint
print(f"Creating Endpoint: {endpoint_name}")
sm_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=config_name
)

# 6. Wait
print("Waiting for endpoint (this takes 5-10 mins)...")
print("Go to SageMaker Console > Endpoints to watch it turn Green.")
try:
    waiter = sm_client.get_waiter('endpoint_in_service')
    # Wait up to 20 minutes (80 attempts * 15 seconds)
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={'Delay': 15, 'MaxAttempts': 80}
    )
    print(f"\n✅ SUCCESS! {endpoint_name} is InService!")
except Exception as e:
    print(f"\n❌ Failed: {e}")
    print("Click 'CloudWatch Logs' in the console to see why.")