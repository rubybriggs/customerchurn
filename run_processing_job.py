import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor

sagemaker_session = sagemaker.Session()

# Replace this with your exact Role ARN if needed
role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

bucket = "company-mlops-churn-production-106679514998"

script_processor = ScriptProcessor(
    command=["python3"],
    image_uri=sagemaker.image_uris.retrieve(
        framework="sklearn", region="us-east-1", version="1.2-1"
    ),
    role=role,
    instance_count=1,
    instance_type="ml.m5.xlarge",
    base_job_name="churn-etl-processing",
)

script_processor.run(
    # ✅ FIX 1: Point to S3 location (after you upload it)
    code=f"s3://{bucket}/code/etl_processing.py",
    inputs=[
        ProcessingInput(
            source=f"s3://{bucket}/raw/v1/",  # ✅ Point to folder, not specific file
            destination="/opt/ml/processing/input",  # ✅ FIX 2: Just the folder, no filename
        ),
    ],
    outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{bucket}/processed/v1/",
        ),
    ],
)