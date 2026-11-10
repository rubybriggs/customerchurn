import boto3
import json
from botocore.exceptions import ClientError


def main():
    # 1. Initialize the Feature Store Runtime Client
    runtime_client = boto3.client(
        'sagemaker-featurestore-runtime',
        region_name='us-east-1'
    )

    feature_group_name = 'customer-churn-feature-group'

    # 2. Get a valid Customer ID from your dataset
    # The original Telco churn dataset has IDs like "7590-VHVEG"
    # Let's try a few common ones from that dataset
    test_customer_ids = [
        '7590-VHVEG',
        '5575-GNVDE',
        '3668-QPYBK',
        '7795-CFOCW',
        '9237-HQITU'
    ]

    print("=" * 60)
    print("VERIFYING ONLINE FEATURE STORE RETRIEVAL")
    print("=" * 60)

    # 3. Try to retrieve records
    for customer_id in test_customer_ids:
        print(f"\nTrying to retrieve record for Customer ID: {customer_id}")
        
        try:
            response = runtime_client.get_record(
                FeatureGroupName=feature_group_name,
                RecordIdentifierValueAsString=customer_id
            )

            record = response.get('Record', [])
            
            if record:
                print(f"✅ SUCCESS! Found record for {customer_id}")
                print(f"Number of features retrieved: {len(record)}")
                
                # Display first 5 features as sample
                print("\nSample features (first 5):")
                for feature in record[:5]:
                    feature_name = feature.get('FeatureName', 'Unknown')
                    feature_value = feature.get('ValueAsString', 'N/A')
                    print(f"  - {feature_name}: {feature_value}")
                
                # Save full record to JSON for inspection
                with open(f'record_{customer_id}.json', 'w') as f:
                    json.dump(record, f, indent=2)
                print(f"\nFull record saved to: record_{customer_id}.json")
                break  # Stop after finding one successful record
            else:
                print(f"⚠️  No record found for {customer_id}")
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFound':
                print(f"❌ Record not found for {customer_id}")
            else:
                print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("ONLINE STORE VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()