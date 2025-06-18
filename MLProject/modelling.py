import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.models.signature import infer_signature


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data = pd.read_excel(args.data_path)
X = data.drop('Churn Label', axis=1)
y = data['Churn Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average='macro'))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average='macro'))
    mlflow.sklearn.log_model(model, "model")
print("Model training complete and metrics logged to MLflow.")
print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")

# Setelah model dilatih
mlflow.sklearn.log_model(model, "model", registered_model_name="ChurnPredictionModel")
signature = infer_signature(X_test, y_pred)
mlflow.log_signature(signature)

with open("metrics_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1 Score: {f1}\n")
mlflow.log_artifact("metrics_report.txt")
