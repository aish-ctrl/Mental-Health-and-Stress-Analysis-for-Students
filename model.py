import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------------------------------------------------
# 1. GENERATE SYNTHETIC STUDENT DATASET
# ---------------------------------------------------
def generate_dataset(n=500, random_state=42):
    rng = np.random.RandomState(random_state)

    sleep = rng.normal(7, 1.5, n).clip(2, 11)
    study = rng.normal(3, 1.8, n).clip(0, 10)
    exercise = rng.normal(2, 1.2, n).clip(0, 7)
    anxiety = rng.normal(5, 2.2, n).clip(0, 10)
    depression = rng.normal(4, 2.0, n).clip(0, 10)
    focus = rng.normal(6, 1.8, n).clip(0, 10)
    diet = rng.normal(5, 2.0, n).clip(0, 10)

    # Stress score formula
    score = (
        (10 - sleep) * 0.7 +
        (study * 0.3) +
        (anxiety * 1.2) +
        (depression * 0.9) -
        (exercise * 0.4) -
        (focus * 0.3) -
        (diet * 0.2)
    )

    # Assign labels
    stress = np.where(
        score < 6, "Low",
        np.where(score < 15, "Medium", "High")
    )

    df = pd.DataFrame({
        "sleep": sleep,
        "study": study,
        "exercise": exercise,
        "anxiety": anxiety,
        "depression": depression,
        "focus": focus,
        "diet": diet,
        "stress": stress
    })

    return df


# Generate dataset
df = generate_dataset()

# ---------------------------------------------------
# 2. ENCODE LABELS & SCALE FEATURES
# ---------------------------------------------------
label_encoder = LabelEncoder()
df["stress_encoded"] = label_encoder.fit_transform(df["stress"])

features = ["sleep", "study", "exercise", "anxiety", "depression", "focus", "diet"]
X = df[features]
y = df["stress_encoded"]

# RF scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# 3. TRAIN RANDOM FOREST
# ---------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Accuracy
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("🎯 RandomForest Accuracy:", accuracy)

# Save accuracy
with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Feature Importance
plt.figure(figsize=(7, 5))
plt.barh(features, rf_model.feature_importances_)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------
# 4. TRAIN KMEANS (UNSUPERVISED)
# ---------------------------------------------------
kmeans_scaler = StandardScaler()
X_kmeans_scaled = kmeans_scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_kmeans_scaled)

# PCA Visual
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_kmeans_scaled)

plt.figure(figsize=(7, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
plt.title("KMeans Cluster Visualization (PCA)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.savefig("kmeans_clusters.png")
plt.close()

# ---------------------------------------------------
# 5. SAVE ALL MODELS AND SCALERS
# ---------------------------------------------------
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans_scaler, "kmeans_scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ ALL MODELS SAVED SUCCESSFULLY!")
