import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---------------- Load MLflow Model ----------------
# Replace <RUN_ID> with your actual MLflow run ID
model_uri = "runs:/<RUN_ID>/wine_clustering_model"
model = mlflow.sklearn.load_model(model_uri)

# ---------------- Load Test Data ----------------
# Option 1: Load from CSV
# test_data = pd.read_csv("wine_test.csv")

# Option 2: Use in-memory test row(s)
test_data = pd.DataFrame([
    {
        "Alcohol": 13.2,
        "Malic_Acid": 2.77,
        "Ash": 2.51,
        "Alcalinity_of_Ash": 18.5,
        "Magnesium": 98,
        "Total_Phenols": 2.0,
        "Flavanoids": 2.5,
        "Nonflavanoid_Phenols": 0.29,
        "Proanthocyanins": 1.3,
        "Color_Intensity": 2.0,
        "Hue": 1.68,
        "OD280": 4.68,
        "Proline": 1015
    },
    {
        "Alcohol": 12.37,
        "Malic_Acid": 1.17,
        "Ash": 1.92,
        "Alcalinity_of_Ash": 19.6,
        "Magnesium": 78,
        "Total_Phenols": 2.11,
        "Flavanoids": 2.0,
        "Nonflavanoid_Phenols": 0.27,
        "Proanthocyanins": 1.04,
        "Color_Intensity": 4.68,
        "Hue": 1.12,
        "OD280": 3.48,
        "Proline": 510
    }
])

# ---------------- Feature Engineering ----------------
test_data["Phenol_Ratio"] = test_data["Flavanoids"] / (test_data["Total_Phenols"] + 1e-6)
test_data["Phenol_Sum"] = test_data["Flavanoids"] + test_data["OD280"] + test_data["Total_Phenols"]

# ---------------- Predict Clusters ----------------
predicted_clusters = model.predict(test_data)
print("🔮 Predicted Clusters:", predicted_clusters)

# ---------------- Internal Evaluation Metrics ----------------
if len(test_data) > 1:
    silhouette = silhouette_score(test_data, predicted_clusters)
    calinski = calinski_harabasz_score(test_data, predicted_clusters)
    davies = davies_bouldin_score(test_data, predicted_clusters)

    print("\n✅ Internal Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz Score: {calinski}")
    print(f"Davies-Bouldin Index: {davies}")
else:
    print("⚠️ Only one sample provided. Internal metrics require multiple samples.")

# ---------------- Log Metrics and Parameters to MLflow ----------------
with mlflow.start_run(run_name="Wine_Clustering_Test_Run"):

    # Log internal metrics
    if len(test_data) > 1:
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.log_metric("calinski_harabasz", calinski)
        mlflow.log_metric("davies_bouldin", davies)

    # Log model parameters
    if hasattr(model.named_steps['kmeans'], 'n_clusters'):
        mlflow.log_param("n_clusters", model.named_steps['kmeans'].n_clusters)
    if hasattr(model.named_steps['pca'], 'n_components'):
        mlflow.log_param("pca_components", model.named_steps['pca'].n_components)

    # Optionally log the model itself again
    mlflow.sklearn.log_model(model, "wine_clustering_model_test")

    print("\n✅ Metrics and parameters logged to MLflow successfully!")
