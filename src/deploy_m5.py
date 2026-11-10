import boto3
import time

region = "us-east-1"
sm_client = boto3.client("sagemaker", region_name=region)
MODEL_PACKAGE_GROUP = "customer-churn-model-group"
ROLE = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

# 1. Get latest model (Version 10)
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

# 2. Create unique names
timestamp = int(time.time())
model_name = f"churn-model-m5-{timestamp}"
config_name = f"churn-config-m5-{timestamp}"
endpoint_name = f"churn-endpoint-m5-{timestamp}"

# 3. Create Model
print("Creating Model resource...")
sm_client.create_model(
    ModelName=model_name,
    Containers=[{"ModelPackageName": package_arn}],
    ExecutionRoleArn=ROLE
)

# 4. Create Config with ml.m5.xlarge (Required for Auto-Scaling)
print("Creating Endpoint Config with ml.m5.xlarge...")
sm_client.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.xlarge"  # <--- CHANGED TO M5
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
try:
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={'Delay': 15, 'MaxAttempts': 60}
    )
    print(f"\n✅ SUCCESS! {endpoint_name} is InService!")
    print(f"\nIMPORTANT: Copy this name for the next steps: {endpoint_name}")
except Exception as e:
    print(f"\n❌ Failed: {e}")