import boto3
import time
import sagemaker
from botocore.exceptions import ClientError

sm_client = boto3.client("sagemaker", region_name="us-east-1")
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

MODEL_PACKAGE_GROUP = "customer-churn-model-group"
ENDPOINT_CONFIG_NAME = "customer-churn-endpoint-config-prod"
ENDPOINT_NAME = "customer-churn-endpoint-prod"

def deploy_approved_model():
    print(f"--- Step 1: Fetching Latest Approved Model Package from '{MODEL_PACKAGE_GROUP}' ---")
    
    approved_packages = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending"
    )["ModelPackageSummaryList"]
    
    if not approved_packages:
        raise Exception("No approved model packages found in Model Registry!")
        
    latest_package_arn = approved_packages[0]["ModelPackageArn"]
    print(f"Found Approved Model Package ARN: {latest_package_arn}")
    
    model_name = f"churn-model-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        Containers=[{"ModelPackageName": latest_package_arn}],
        ExecutionRoleArn=role
    )
    print(f"Created SageMaker Model Resource: {model_name}")
    
    print("--- Step 2: Provisioning Endpoint Configuration ---")
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
    except ClientError:
        pass
        
    sm_client.create_endpoint_config(
        EndpointConfigName=ENDPOINT_CONFIG_NAME,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "InitialVariantWeight": 1.0
        }]
    )
    print(f"Endpoint Config '{ENDPOINT_CONFIG_NAME}' created.")
    
    print(f"--- Step 3: Deploying Endpoint '{ENDPOINT_NAME}' ---")
    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint exists. Updating...")
        sm_client.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME)
    except ClientError:
        print(f"Creating new production endpoint...")
        sm_client.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME)
        
    print("Waiting for endpoint provisioning (5-8 mins)...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=ENDPOINT_NAME)
    print(f"SUCCESS! Endpoint '{ENDPOINT_NAME}' is live!")

if __name__ == "__main__":
    deploy_approved_model()