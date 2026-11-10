import json
import boto3

# Use the SageMaker execution role instead of IAM user
events_client = boto3.client("events", region_name="us-east-1")
sagemaker_client = boto3.client("sagemaker", region_name="us-east-1")

RULE_NAME = "CustomerChurn-WeeklyPipelineTrigger"
PIPELINE_NAME = "CustomerChurn-Orchestration-Pipeline"

# Get the actual pipeline ARN
response = sagemaker_client.describe_pipeline(PipelineName=PIPELINE_NAME)
SAGEMAKER_PIPELINE_ARN = response["PipelineArn"]

# Use the correct account ID and role
ROLE_ARN = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

def setup_eventbridge_schedule():
    print(f"Creating EventBridge Rule '{RULE_NAME}'...")
    
    # 1. Put Schedule Rule (Runs every Sunday at 00:00 UTC)
    rule_response = events_client.put_rule(
        Name=RULE_NAME,
        ScheduleExpression="cron(0 0 ? * SUN *)",
        State="ENABLED",
        Description="Triggers weekly customer churn model retraining pipeline."
    )
    
    print(f"Rule ARN: {rule_response['RuleArn']}")
    
    # 2. Add Target: SageMaker Pipeline
    target_response = events_client.put_targets(
        Rule=RULE_NAME,
        Targets=[{
            "Id": "SageMakerPipelineTarget",
            "Arn": SAGEMAKER_PIPELINE_ARN,
            "RoleArn": ROLE_ARN,
            "SageMakerPipelineParameters": {
                "PipelineParameterList": []
            }
        }]
    )
    
    print("EventBridge Target linked to SageMaker Pipeline successfully!")

if __name__ == "__main__":
    setup_eventbridge_schedule()