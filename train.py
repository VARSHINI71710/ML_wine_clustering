import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# ---------------- Load dataset ----------------
df = pd.read_csv("C:/Users/varsh/Downloads/wine-clustering-extended.csv")

# ---------------- Feature Engineering ----------------
# Make sure column names match exactly (case-sensitive)
df["Phenol_Ratio"] = df["Flavanoids"] / df["Total_Phenols"]
df["Phenol_Sum"] = df["Flavanoids"] + df["OD280"] + df["Total_Phenols"]

# Drop target if exists
X = df.drop(columns=["Cluster"], errors="ignore")

# ---------------- Define pipeline ----------------
n_components = 2
n_clusters = 2

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components)),
    ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
])

# ---------------- MLflow Logging ----------------
mlflow.set_experiment("Wine_Clustering")

with mlflow.start_run(run_name="wine_clustering_run"):

    # Fit pipeline
    pipeline.fit(X)
    labels = pipeline["kmeans"].labels_

    # ---------------- Internal Metrics ----------------
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)

    # Log metrics
    mlflow.log_metric("silhouette_score", silhouette)
    mlflow.log_metric("calinski_harabasz", calinski)
    mlflow.log_metric("davies_bouldin", davies)

    # ---------------- Log Parameters ----------------
    mlflow.log_param("n_pca_components", n_components)
    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_param("random_state", 42)

    # ---------------- Save Model ----------------
    mlflow.sklearn.log_model(
        pipeline,
        "wine_clustering_model",
        registered_model_name="WineClusteringModel"
    )

    # ---------------- Print ----------------
    print("✅ Model trained and logged with metrics & parameters")
    print(f"Silhouette Score: {silhouette}")
    print(f"Calinski-Harabasz: {calinski}")
    print(f"Davies-Bouldin: {davies}")
    print(f"Parameters: n_components={n_components}, n_clusters={n_clusters}, random_state=42")
