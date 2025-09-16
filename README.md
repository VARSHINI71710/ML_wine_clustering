🍷 Wine Clustering App

An interactive Machine Learning app that clusters wine data into different groups using KMeans, PCA, and StandardScaler.
Built with Python, Scikit-learn, Joblib, and Streamlit/Gradio, and deployed on Hugging Face Spaces.

🚀 Features

Upload wine dataset or use the preloaded one.

Data preprocessing with StandardScaler.

Dimensionality reduction using PCA.

Clustering with KMeans.

Visualize results in 2D and explore cluster assignments.

Fully deployed on Hugging Face.

📂 Project Structure
📦 wine-clustering-app
 ┣ 📜 app.py                 # Main application file
 ┣ 📜 requirements.txt       # Python dependencies
 ┣ 📜 README.md              # Project documentation
 ┣ 📜 scaler.joblib          # Saved StandardScaler
 ┣ 📜 pca.joblib             # Saved PCA model
 ┣ 📜 kmeans.joblib          # Saved KMeans model
 ┗ 📜 wine-clustering-extended.csv  # Dataset (optional)

⚙️ Installation (Run Locally)

Clone the repo and install dependencies:

git clone https://huggingface.co/spaces/<your-username>/wine-clustering-app
cd wine-clustering-app
pip install -r requirements.txt


Run the app:

streamlit run app.py

📦 Requirements

requirements.txt contains:

streamlit
scikit-learn
pandas
numpy
joblib
matplotlib

🔑 Key Files

scaler.joblib → StandardScaler for feature normalization.

pca.joblib → PCA transformation (2D visualization).

kmeans.joblib → Pretrained clustering model.

app.py → Loads models and runs the interactive UI.

☁️ Deployment on Hugging Face

Create a new Space on Hugging Face.

Choose Streamlit SDK.

Upload all files:

app.py

requirements.txt

scaler.joblib

pca.joblib

kmeans.joblib

(optional) wine-clustering-extended.csv

Wait for the build → Your app will go live 🚀

🧠 MLflow Logging (Optional Advanced)

During training, we logged:

Clustering Metrics: Silhouette Score, Davies-Bouldin Score, Calinski-Harabasz Index.

Hyperparameters: Number of clusters, PCA components, scaling method.

Artifacts: Saved models (.joblib).

This helps track experiments and compare results.

🎯 Future Improvements

Add more clustering algorithms (DBSCAN, Hierarchical).

Enable custom dataset uploads.

Improve visualization with interactive plots.

Integrate full MLflow tracking in Hugging Face.

🙌 Credits

Developed by Varshini S ✨
Built with ❤️ using Python, Scikit-learn, and Hugging Face Spaces.
