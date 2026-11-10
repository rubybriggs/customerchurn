import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.model import Model

# Configuration
REGION = "us-east-1"
BUCKET = "company-mlops-churn-production-106679514998"
ROLE = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
MODEL_PACKAGE_GROUP_NAME = "customer-churn-model-group"
sagemaker_session = sagemaker.Session()


def build_cd_pipeline():
    # 1. Processing Step: Governance & Metric Evaluation
    image_uri = sagemaker.image_uris.retrieve(framework="sklearn", region=REGION, version="1.2-1")
    
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=ROLE,
        sagemaker_session=sagemaker_session
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    step_eval = ProcessingStep(
        name="EvaluateChurnModelQuality",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/models/xgboost-model.tar.gz",
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=f"s3://{BUCKET}/processed/v1/customer_churn_processed.csv",
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation"
            )
        ],
        code="src/evaluate.py",
        property_files=[evaluation_report]
    )
    
    # 2. Condition Step: Governance Quality Gate (PR-AUC >= 0.80)
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.pr_auc.value"
        ),
        right=0.80
    )
    
    # 3. Model Registration Step (Triggered if Condition succeeds)
    xgb_image_uri = sagemaker.image_uris.retrieve(framework="xgboost", region=REGION, version="1.5-1")
    
    model = Model(
        image_uri=xgb_image_uri,
        model_data=f"s3://{BUCKET}/models/xgboost-model.tar.gz",
        sagemaker_session=sagemaker_session,
        role=ROLE
    )
    
    # Use the model's built-in register method which returns the Step object
    step_register = model.register(
    content_types=["text/csv", "application/json"],
    response_types=["text/csv", "application/json"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
    approval_status="Approved"
    )
    
    # 4. Construct Pipeline Graph
    pipeline = Pipeline(
        name="CustomerChurn-CD-Pipeline",
        steps=[
            step_eval,
            ConditionStep(
                name="CheckQualityGate",
                conditions=[cond_gte],
                if_steps=[step_register],
                else_steps=[]
            )
        ],
        sagemaker_session=sagemaker_session
    )
    
    return pipeline


if __name__ == "__main__":
    pipeline = build_cd_pipeline()
