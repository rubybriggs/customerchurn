import os
import json
import tarfile
import pathlib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def evaluate_model():
    print("--- Governance Evaluation: Testing Model Quality ---")
    
    # 1. Paths set by SageMaker Pipelines environment
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.csv"
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 2. Extract trained model binary
    with tarfile.open(model_path) as tar:
        tar.extractall(path="./extracted_model")
    booster = xgb.Booster()
    booster.load_model("./extracted_model/xgboost-model")
    
    # 3. Load Evaluation Data
    df_test = pd.read_csv(test_path)
    
    # Handle case sensitivity for 'Churn' column
    target_col = 'churn' if 'churn' in df_test.columns else 'Churn'
    
    X_test = df_test.drop(columns=[target_col]).values
    y_test = df_test[target_col].values
    
    # 4. Generate Predictions
    dtest = xgb.DMatrix(X_test)
    probabilities = booster.predict(dtest)
    
    # 5. Compute Metrics
    roc_auc = float(roc_auc_score(y_test, probabilities))
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc = float(auc(recall, precision))
    
    print(f"Evaluation Complete. ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    
    # 6. Format Standard SageMaker Evaluation Report
    report_dict = {
        "classification_metrics": {
            "roc_auc": {"value": roc_auc, "standard_deviation": "NaN"},
            "pr_auc": {"value": pr_auc, "standard_deviation": "NaN"}
        }
    }
    
    evaluation_path = os.path.join(output_dir, "evaluation.json")
    with open(evaluation_path, "w") as f:
        json.dump(report_dict, f, indent=4)
    print(f"Saved evaluation report to: {evaluation_path}")

if __name__ == "__main__":
    evaluate_model()