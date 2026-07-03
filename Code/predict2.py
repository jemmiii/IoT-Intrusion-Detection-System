import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# -------------------------------
# 1. LOAD MODELS (Dataset 2 wale)
# -------------------------------
lgb_model = lgb.Booster(model_file="intrusion_model_dataset2.txt")
rf_model = joblib.load("rf_model_dataset2.pkl")
dt_model = joblib.load("dt_model_dataset2.pkl")

# -------------------------------
# 2. LOAD LABEL CLASSES
# -------------------------------
classes = np.load("label_classes.npy", allow_pickle=True)

# -------------------------------
# 3. LOAD TEST DATA
# -------------------------------
df = pd.read_csv("Code/test_data_dataset2.csv")

# -------------------------------
# 4. CLEANING
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# -------------------------------
# 5. HANDLE OBJECT COLUMNS
# -------------------------------
# Save label if exists
if 'type' in df.columns:
    labels = df['type']
else:
    labels = None

# Remove all string columns
df = df.select_dtypes(exclude=['object'])

# Add label back if existed
if labels is not None:
    df['type'] = labels

# -------------------------------
# 6. DROP LABEL COLUMNS
# -------------------------------
X = df.drop(columns=['type', 'label', 'label_encoded'], errors='ignore')

# -------------------------------
# 7. LIGHTGBM PREDICTION
# -------------------------------
lgb_pred = np.argmax(lgb_model.predict(X), axis=1)
lgb_labels = classes[lgb_pred]

# -------------------------------
# 8. RANDOM FOREST PREDICTION
# -------------------------------
rf_pred = rf_model.predict(X)
rf_labels = classes[rf_pred]

# -------------------------------
# 9. DECISION TREE PREDICTION
# -------------------------------
dt_pred = dt_model.predict(X)
dt_labels = classes[dt_pred]

# -------------------------------
# 10. ADD RESULTS TO DATAFRAME
# -------------------------------
df['LGB_Prediction'] = lgb_labels
df['RF_Prediction'] = rf_labels
df['DT_Prediction'] = dt_labels

# -------------------------------
# 11. PRINT OUTPUT
# -------------------------------
print("\nPredictions:\n")
print(df[['LGB_Prediction', 'RF_Prediction', 'DT_Prediction']].head())


df.to_csv("output_dataset2_predictions.csv", index=False)

print("\n✅ Predictions saved to output_dataset2_predictions.csv")