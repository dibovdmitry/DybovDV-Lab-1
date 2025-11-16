import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.models import infer_signature

# Setting a tracking URI(points to a running server)
mlflow.set_tracking_uri("http://localhost:5000")

# Create or install an active experiment
experiment_name = "Iris_Classification_Baseline"
mlflow.set_experiment(experiment_name)

# Uploading Data
iris = load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the model parameters for logging
params = {
    "solver":"liblinear",
    "max_iter": 350,
    "multi_class": "auto",
    "random_state": 99
}
# Start of MLflow launch
with mlflow.start_run():
    mlflow.log_params(params)
    # Creating and training model
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    # Prediction and calculation of metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # Logging metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    # Input_example data and signature
    input_example = pd.DataFrame(X_test[:1], columns=iris.feature_names)
    signature = infer_signature(X_train, model.predict(X_train))
    # Logging model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )
    # Creating and logging an artifact
    fig, ax = plt.subplots()
    ax.bar(['Accuracy', 'Precision', 'Recall', 'F1'], [accuracy, precision, recall, f1])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Metrics')
    plt.savefig("metrics_plot.png") # Saving chart to a file
    mlflow.log_artifact("metrics_plot.png") # Log file like artifact

    # Output of metrics to the console
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Run completed and logged to MLflow!")
