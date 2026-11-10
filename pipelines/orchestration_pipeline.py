import os
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

def get_pipeline(
    region="us-east-1",
    role=None,
    default_bucket="company-mlops-churn-production-106679514998",
    pipeline_name="CustomerChurn-Orchestration-Pipeline"
):
    pipeline_session = PipelineSession()
    
    if role is None:
        role = sagemaker.get_execution_role(sagemaker_session=pipeline_session)
    
    # -------------------------------------------------------------------------
    # 1. Preprocessing Step
    # -------------------------------------------------------------------------
    sklearn_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn", region=region, version="1.2-1"
    )
    
    script_processor = ScriptProcessor(
        image_uri=sklearn_image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session
    )
    
    step_process = ProcessingStep(
        name="PreprocessChurnData",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=f"s3://{default_bucket}/raw/v1/customer_churn.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test")
        ],
        code="src/preprocess.py"
    )
    
    # -------------------------------------------------------------------------
    # 2. Training Step (XGBoost)
    # -------------------------------------------------------------------------
    xgb_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost", region=region, version="1.5-1"
    )
    
    xgb_estimator = Estimator(
        image_uri=xgb_image_uri,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
        output_path=f"s3://{default_bucket}/models/pipeline-output"
    )
    
    xgb_estimator.set_hyperparameters(
        max_depth=5,
        eta=0.1,
        num_round=100,
        objective="binary:logistic",
        eval_metric="logloss"
    )
    
    step_train = TrainingStep(
        name="TrainXGBoostModel",
        estimator=xgb_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )
    
    # -------------------------------------------------------------------------
    # 3. Model Quality Evaluation Step
    # -------------------------------------------------------------------------
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    step_eval = ProcessingStep(
        name="EvaluateModelQuality",
        processor=script_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
        ],
        code="src/evaluate.py",
        property_files=[evaluation_report]
    )
    
    # -------------------------------------------------------------------------
    # 4. Model Registration Step (Using ProcessingStep)
    # -------------------------------------------------------------------------
    step_register = ProcessingStep(
        name="RegisterModelInRegistry",
        processor=script_processor,
        inputs=[],
        code="src/register_model.py",
        job_arguments=[
            "--model-s3-uri", 
            step_train.properties.ModelArtifacts.S3ModelArtifacts
        ]
    )
    
    # -------------------------------------------------------------------------
    # 5. Condition Step (PR-AUC >= 0.80)
    # -------------------------------------------------------------------------
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.pr_auc.value"
        ),
        right=0.80
    )
    
    step_cond = ConditionStep(
        name="CheckPRACMetricGate",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[]
    )
    
    # -------------------------------------------------------------------------
    # 6. Pipeline Construction
    # -------------------------------------------------------------------------
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session
    )
    
    return pipeline

if __name__ == "__main__":
    boto_session = boto3.Session(region_name="us-east-1")
    
    # Explicitly define the role since we are running locally
    sm_role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
    
    pipeline = get_pipeline(role=sm_role)
    
    print("Upserting SageMaker Pipeline definition to AWS...")
    pipeline.upsert(role_arn=sm_role)
    
    print("Starting pipeline execution...")
    execution = pipeline.start()
    print(f"Pipeline Execution Started! Execution ARN: {execution.arn}")