import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 📌 Step 1: Load dataset
df = pd.read_csv("cleaned_airline_reviews.csv")

# 📌 Step 2: Define important columns (ensure only useful features are included)
important_features = ["airline", "traveller_type", "cabin", "seat_comfort", 
                      "cabin_service", "food_bev", "entertainment", "value_for_money"]

# 📌 Step 3: Ensure target variable exists
if "recommended" not in df.columns:
    raise ValueError("❌ 'recommended' column is missing in the dataset!")

# 📌 Step 4: Encode categorical variables
label_encoders = {}
for col in ["airline", "traveller_type", "cabin"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Save LabelEncoders for Flask app
joblib.dump(label_encoders, "label_encoders.pkl")

# 📌 Step 5: Extract features (X) and target (y)
X = df[important_features]
y = df["recommended"]

# 📌 Step 6: Check class distribution before SMOTE
print("\n🔥 Before SMOTE:\n", y.value_counts())

# 📌 Step 7: Apply SMOTE for class balancing
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 📌 Step 8: Check class distribution after SMOTE
print("\n✅ After SMOTE:\n", pd.Series(y_resampled).value_counts())

# 📌 Step 9: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 📌 Step 10: Apply feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for deployment
joblib.dump(scaler, "scaler.pkl")

# 📌 Step 11: Train the Random Forest Model (Balanced Training)
model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 📌 Step 12: Check predictions on training data to see if the model is still biased
y_train_pred = model.predict(X_train_scaled)
print("\n🔍 Training Set Predictions:\n", pd.Series(y_train_pred).value_counts())

# Save the trained model
joblib.dump(model, "rf_model.pkl")

print("\n✅ Model trained and saved successfully!")
