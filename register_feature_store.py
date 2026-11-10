import boto3
import pandas as pd
import sagemaker
import time
from sagemaker.feature_store.feature_group import FeatureGroup


def main():
    # 1. Initialize Sessions
    sagemaker_session = sagemaker.Session()
    boto_session = boto3.Session(region_name='us-east-1')
    sagemaker_client = boto_session.client('sagemaker')
    featurestore_runtime = boto_session.client('sagemaker-featurestore-runtime')

    role = "arn:aws:iam::592444418608:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
    bucket = "company-mlops-churn-production-106679514998"
    feature_group_name = "customer-churn-feature-group"

    # 2. Load Processed Dataset
    print("Loading processed data from S3...")
    df = pd.read_csv(f"s3://{bucket}/processed/v1/customer_churn_processed.csv")
    print(f"Loaded {len(df)} records with {len(df.columns)} features")

    # 3. Instantiate Feature Group
    print(f"Creating Feature Group: {feature_group_name}")
    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sagemaker_session
    )

    # Load feature definitions dynamically from dataframe schema
    feature_group.load_feature_definitions(data_frame=df)

    # Specify Record Identifier and Event Time feature names
    record_identifier_feature_name = "customerID"
    event_time_feature_name = "EventTime"

    # 4. Create Feature Group on AWS
    print("Provisioning Feature Group on AWS SageMaker...")
    try:
        feature_group.create(
            s3_uri=f"s3://{bucket}/feature-store/",
            record_identifier_name=record_identifier_feature_name,
            event_time_feature_name=event_time_feature_name,
            role_arn=role,
            enable_online_store=True
        )
    except Exception as e:
        if "ResourceInUse" in str(e) or "already exists" in str(e):
            print("Feature Group already exists. Skipping creation.")
        else:
            raise e

    # Wait for Feature Group provisioning status to become Active
    print("Checking Feature Group status...")
    status = feature_group.describe().get("FeatureGroupStatus")
    print(f"Current status: {status}")

    while status == "Creating":
        print("Waiting for Feature Group creation...")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
        print(f"Current status: {status}")
        if status == "Failed":
            raise Exception(f"Failed to create Feature Group: {feature_group.describe()}")

    print(f"Feature Group {feature_group_name} is ready (Status: {status}).")

    # 5. Ingest Records into Feature Store (max_workers=1 to avoid Windows multiprocessing issues)
    print("Ingesting records into SageMaker Feature Store...")
    feature_group.ingest(data_frame=df, max_workers=1, wait=True)
    print("Data ingestion to Feature Store complete!")

    # 6. Verify Ingestion
    print(f"\nFeature Store Summary:")
    print(f"- Feature Group: {feature_group_name}")
    print(f"- Total Records Ingested: {len(df)}")
    print(f"- Total Features: {len(df.columns)}")
    print(f"- Online Store: Enabled")
    print(f"- Offline Store: Enabled (S3)")


if __name__ == '__main__':
    main()