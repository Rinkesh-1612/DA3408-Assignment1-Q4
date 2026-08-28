"""Q4 capstone (Partner A): train, log to MLflow, register the model, promote to Staging."""
import subprocess

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
MODEL_NAME = "q4-digits-classifier"

git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()

df = pd.read_csv("data/q4_digits.csv")
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

n_estimators = 200
max_depth = 10

mlflow.set_experiment("q4-capstone-reproducibility")
with mlflow.start_run(run_name="q4-partner-a-run") as run:
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("seed", SEED)
    mlflow.set_tag("git_commit", git_commit)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=SEED
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average="macro")

    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1_macro", f1)

    model_info = mlflow.sklearn.log_model(
        model, name="model", registered_model_name=MODEL_NAME
    )

    print(f"run_id={run.info.run_id}")
    print(f"git_commit={git_commit}")
    print(f"val_accuracy={acc:.4f} val_f1_macro={f1:.4f}")

client = mlflow.MlflowClient()
latest = client.get_latest_versions(MODEL_NAME)[-1]
client.transition_model_version_stage(
    name=MODEL_NAME, version=latest.version, stage="Staging"
)
print(f"Registered {MODEL_NAME} v{latest.version} and transitioned to Staging")
