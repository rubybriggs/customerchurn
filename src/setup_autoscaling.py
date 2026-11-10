import boto3

autoscaling_client = boto3.client("application-autoscaling", region_name="us-east-1")
ENDPOINT_NAME="churn-endpoint-m5-1793884521"
RESOURCE_ID = f"endpoint/{ENDPOINT_NAME}/variant/AllTraffic"

def configure_autoscaling():
    print(f"Configuring Auto-Scaling for Endpoint Variant: {RESOURCE_ID}")
    
    autoscaling_client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=3
    )
    
    autoscaling_client.put_scaling_policy(
        PolicyName="SageMakerVariantInvocationsPerMinTarget",
        ServiceNamespace="sagemaker",
        ResourceId=RESOURCE_ID,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 1000.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60
        }
    )
    print("Auto-scaling policy successfully attached!")

if __name__ == "__main__":
    configure_autoscaling()