import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load LabelEncoders for categorical variables
label_encoders = joblib.load("label_encoders.pkl")

# Define only the required feature names
feature_names = ["airline", "traveller_type", "cabin", "seat_comfort", 
                 "cabin_service", "food_bev", "entertainment", "value_for_money"]

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
        for key in ["seat_comfort", "cabin_service", "food_bev", "entertainment", "value_for_money"]:
            try:
                input_data[key] = float(input_data[key])
            except ValueError:
                input_data[key] = 0.0  # Default to 0 if conversion fails

        # Convert categorical features using saved LabelEncoders
        for col in ["airline", "traveller_type", "cabin"]:
            if col in label_encoders:
                if input_data[col] in label_encoders[col].classes_:
                    input_data[col] = label_encoders[col].transform([input_data[col]])[0]
                else:
                    input_data[col] = -1  # Assign -1 to unseen categories
            else:
                input_data[col] = -1  # Assign -1 if encoder is missing

        # Create DataFrame with only required features
        df = pd.DataFrame([input_data])

        # Ensure feature order matches training
        df = df[feature_names]

        # Apply scaling
        df_scaled = scaler.transform(df)

        # Predict using the model
        prediction = model.predict(df_scaled)[0]
        probabilities = model.predict_proba(df_scaled)[0]

        # Debugging prints (Check predictions)
        print(f"🔍 Prediction: {prediction}, Probabilities: {probabilities}")

        # Convert prediction to numeric (if needed)
        if isinstance(prediction, np.generic):  # Fixes NumPy int/str issue
            prediction = np.asscalar(prediction)
        elif isinstance(prediction, str):  # If model returns "yes"/"no" as string
            prediction = 1 if prediction.lower() == "yes" else 0

        # Fix condition (Ensure '1' is correctly mapped to 'Recommended')
        result = "Recommended" if prediction == 1 else "Not Recommended"

        return render_template("index.html", prediction=result, probability=round(probabilities[1] * 100, 2))

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
