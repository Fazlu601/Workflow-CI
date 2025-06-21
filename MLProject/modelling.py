import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("file:///mlruns")  # Paksa gunakan path lokal di Linux

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load data
data = pd.read_excel(args.data_path)
X = data.drop('Churn Label', axis=1)
y = data['Churn Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# MLflow tracking
with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Signature and model registry
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnPredictionModel", signature=signature)

    # Log metrics report as artifact
    with open("metrics_report.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")

    mlflow.log_artifact("metrics_report.txt")

print("✅ Model training complete and metrics logged to MLflow.")
print(f"Model accuracy: {accuracy}")

# Tambahan simpan ke file lokal untuk inference.py
import joblib
joblib.dump(model, "model.pkl")

