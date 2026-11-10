import boto3

runtime_client = boto3.client("sagemaker-runtime", region_name="us-east-1")
ENDPOINT_NAME="churn-endpoint-m5-1793884521"

# 30 numerical features matching the clean matrix schema
sample_customer_payload = "0,1,0,2,1,1,55.3,110.6,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0"

def invoke_inference():
    print(f"Sending REST prediction request to endpoint '{ENDPOINT_NAME}'...")
    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=sample_customer_payload
    )
    result = response["Body"].read().decode("utf-8")
    churn_probability = float(result.strip())
    
    print("\n================ PREDICTION RESULT ================")
    print(f"Customer Churn Risk Probability: {churn_probability * 100:.2f}%")
    print(f"Raw Output: {result}")
    print("==================================================\n")

if __name__ == "__main__":
    invoke_inference()