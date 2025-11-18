import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib

# Use SAME dataset as RandomForest
from train_model import generate_dataset

df = generate_dataset()

X = df[["sleep", "study", "exercise", "anxiety", "depression", "focus", "diet"]]

kmeans_scaler = StandardScaler()
X_scaled = kmeans_scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# PCA graph
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans.labels_)
plt.title("KMeans Cluster Visualization")
plt.savefig("kmeans_clusters.png")
plt.close()

joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(kmeans_scaler, "kmeans_scaler.pkl")

print("✅ KMeans model & scaler saved!")
