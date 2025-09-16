import gradio as gr
import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


scaler = load("scaler.joblib")     
pca = load("pca.joblib")          
kmeans = load("kmeans.joblib")     

df = pd.read_csv("wine-clustering-extended.csv")

feature_names = [

    "Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity", "Magnesium",
    "Total_Phenols", "Flavanoids", "Nonflavanoid_Phenols", "Proanthocyanins",
    "Color_Intensity", "Hue", "OD280", "Proline"
]

X = df[feature_names]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_pca)

def predict_cluster(*features):
    try:
        sample = np.array(features).reshape(1, -1)

        sample_scaled = scaler.transform(sample)
        sample_pca = pca.transform(sample_scaled)
        cluster = int(kmeans.predict(sample_pca)[0])

        cluster_map = {0: "🍷 Medium Quality", 1: "🍷 Good Quality"}
        return cluster_map.get(cluster, f"Cluster {cluster}")
    except Exception as e:
        return f"❌ Error: {str(e)}"
VALID_USERS = {"admin": "1234", "user": "pass"}  

def login(username, password):
    if username in VALID_USERS and VALID_USERS[username] == password:
        return gr.update(visible=False), gr.update(visible=True), f"✅ Welcome, {username}!"
    else:
        return gr.update(visible=True), gr.update(visible=False), "❌ Invalid credentials."

with gr.Blocks(theme="soft") as app:
    gr.Markdown("## 🍷 Wine Quality Clustering App with Login")

    with gr.Group(visible=True) as login_page:
        gr.Markdown("### 🔐 Login to Continue")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login")
        login_msg = gr.Label()

    with gr.Group(visible=False) as prediction_page:
        gr.Markdown("### 🍇 Enter Wine Features (13 inputs)")
        inputs = [gr.Number(label=col) for col in feature_names]
        output = gr.Textbox(label="Cluster Result")
        predict_btn = gr.Button("Predict Cluster")
        logout_btn = gr.Button("Logout 🔓")

    login_btn.click(
        login,
        inputs=[username, password],
        outputs=[login_page, prediction_page, login_msg]
    )

    predict_btn.click(
        predict_cluster,
        inputs=inputs,
        outputs=output
    )

    logout_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), "✅ Logged out"),
        outputs=[login_page, prediction_page, login_msg]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app1:app", host="127.0.0.1", port=7860, reload=True)