from flask import Flask, render_template, request, send_from_directory, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# -------------------------------------------------------
# LOAD ALL MODELS & SCALERS
# -------------------------------------------------------
rf_model = joblib.load("rf_model.pkl")             # Random Forest
kmeans_model = joblib.load("kmeans_model.pkl")     # KMeans

rf_scaler = joblib.load("scaler.pkl")              # Random Forest scaler
kmeans_scaler = joblib.load("kmeans_scaler.pkl")   # KMeans scaler

label_encoder = joblib.load("label_encoder.pkl")   # Label Encoder

# Load saved model accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()


# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------
# ML PREDICTION (AJAX REQUEST)
# -------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect user inputs
        input_values = {
            "sleep": float(request.form["sleep"]),
            "study": float(request.form["study"]),
            "exercise": float(request.form["exercise"]),
            "anxiety": float(request.form["anxiety"]),
            "depression": float(request.form["depression"]),
            "focus": float(request.form["focus"]),
            "diet": float(request.form["diet"])
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_values])

        # -------------------------
        # RANDOM FOREST PREDICTION
        # -------------------------
        scaled_rf = rf_scaler.transform(df)
        pred_class_num = rf_model.predict(scaled_rf)[0]
        pred_class_label = label_encoder.inverse_transform([pred_class_num])[0]

        # -------------------------
        # KMEANS CLUSTER RESULT
        # -------------------------
        scaled_km = kmeans_scaler.transform(df)
        cluster_group = int(kmeans_model.predict(scaled_km)[0])

        # Return JSON response
        return jsonify({
            "stress": pred_class_label,
            "cluster": cluster_group,
            "accuracy": accuracy
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -------------------------------------------------------
# SERVE GRAPH FILES
# -------------------------------------------------------
@app.route("/graph/<filename>")
def graph(filename):
    return send_from_directory(".", filename)


# -------------------------------------------------------
# START FLASK APP
# -------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
