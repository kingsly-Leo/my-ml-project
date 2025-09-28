import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load LabelEncoders for categorical variables
label_encoders = joblib.load("label_encoders.pkl")

# Define only the required feature names (must match training order)
feature_names = [
    "airline",
    "traveller_type",
    "cabin",
    "seat_comfort",
    "cabin_service",
    "food_bev",
    "entertainment",
    "value_for_money",
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract only necessary inputs from form
        input_data = {
            "airline": request.form.get("airline", "").strip(),
            "traveller_type": request.form.get("traveller_type", "").strip(),
            "cabin": request.form.get("cabin", "").strip(),
            "seat_comfort": request.form.get("seat_comfort", "0"),
            "cabin_service": request.form.get("cabin_service", "0"),
            "food_bev": request.form.get("food_bev", "0"),
            "entertainment": request.form.get("entertainment", "0"),
            "value_for_money": request.form.get("value_for_money", "0"),
        }

        # Convert numerical inputs safely
        for key in [
            "seat_comfort",
            "cabin_service",
            "food_bev",
            "entertainment",
            "value_for_money",
        ]:
            try:
                input_data[key] = float(input_data[key])
            except (ValueError, TypeError):
                input_data[key] = 0.0  # Default to 0 if conversion fails

        # Convert categorical features using saved LabelEncoders
        # If category not seen, assign -1 (numeric) — ensure your scaler/model can handle it
        for col in ["airline", "traveller_type", "cabin"]:
            if col in label_encoders:
                le = label_encoders[col]
                # If unseen category, set to -1 (or choose another default)
                try:
                    if input_data[col] in le.classes_:
                        input_data[col] = int(le.transform([input_data[col]])[0])
                    else:
                        # Option: map unseen to -1 (numeric). Ensure model was trained to handle this.
                        input_data[col] = -1
                except Exception:
                    input_data[col] = -1
            else:
                input_data[col] = -1

        # Create DataFrame with only required features (order matters)
        df = pd.DataFrame([input_data])
        df = df[feature_names]

        # Apply scaling (scaler must accept these columns in this order)
        df_scaled = scaler.transform(df)

        # Predict using the model
        prediction = model.predict(df_scaled)[0]
        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df_scaled)[0]

        # Debug prints (they show up in container logs)
        app.logger.info(f"Prediction raw: {prediction}, probabilities: {probabilities}")

        # Normalize prediction to int 0/1
        try:
            # If numpy scalar, cast to Python int
            prediction_value = int(prediction)
        except Exception:
            # If string label, map "yes"/"no" or similar
            if isinstance(prediction, str):
                prediction_value = 1 if prediction.lower() in ("yes", "true", "1") else 0
            else:
                prediction_value = 0

        result = "Recommended" if prediction_value == 1 else "Not Recommended"

        prob_display = None
        if probabilities is not None:
            # Safe index: if model has two classes, show probabilities[1]
            if len(probabilities) >= 2:
                prob_display = round(float(probabilities[1]) * 100, 2)
            else:
                prob_display = round(float(probabilities[0]) * 100, 2)

        return render_template("index.html", prediction=result, probability=prob_display)

    except Exception as e:
        app.logger.exception("Error in /predict")
        # Return user-friendly JSON on AJAX calls or JSON requests
        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": str(e)}), 500
        # Otherwise render error on page (simple)
        return render_template("index.html", error=str(e))


if __name__ == "__main__":
    # Production-ready: use PORT env var (Render/Railway) and bind to 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    # Don't enable debug=True in production
    app.run(host="0.0.0.0", port=port)

