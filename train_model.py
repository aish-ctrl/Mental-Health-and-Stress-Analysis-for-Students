import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --------------------------------------------------------
# 1. GENERATE SYNTHETIC DATASET (500 STUDENTS)
# --------------------------------------------------------
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

    labels = np.where(score < 6, "Low",
              np.where(score < 15, "Medium", "High"))

    df = pd.DataFrame({
        "sleep": sleep,
        "study": study,
        "exercise": exercise,
        "anxiety": anxiety,
        "depression": depression,
        "focus": focus,
        "diet": diet,
        "stress": labels
    })

    return df

df = generate_dataset()

# --------------------------------------------------------
# 2. ENCODING, SCALING
# --------------------------------------------------------
label_encoder = LabelEncoder()
df["stress_encoded"] = label_encoder.fit_transform(df["stress"])

X = df[["sleep", "study", "exercise", "anxiety", "depression", "focus", "diet"]]
y = df["stress_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------------
# 3. TRAIN MODEL
# --------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------------------------------------
# 4. ACCURACY
# --------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("🎯 Accuracy:", accuracy)

with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

# --------------------------------------------------------
# 5. CONFUSION MATRIX PLOT
# --------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# --------------------------------------------------------
# 6. FEATURE IMPORTANCE PLOT
# --------------------------------------------------------
importances = model.feature_importances_
plt.figure(figsize=(7,5))
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

# --------------------------------------------------------
# 7. SAVE FILES
# --------------------------------------------------------
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ RF model, scaler, encoder saved successfully!")
