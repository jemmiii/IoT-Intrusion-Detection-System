import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

# -------------------------------
# 1. LOAD MODELS
# -------------------------------
lgb_model = lgb.Booster(model_file="intrusion_model.txt")
rf_model = joblib.load("rf_model.pkl")
dt_model = joblib.load("dt_model.pkl")

# -------------------------------
# 2. LOAD LABELS
# -------------------------------
classes = np.load("label_classes.npy", allow_pickle=True)

# -------------------------------
# 3. LOAD TEST DATA (IoT)
# -------------------------------
df = pd.read_csv("Code/test_data.csv")

# -------------------------------
# 4. CLEANING
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# -------------------------------
# 5. FEATURE ENGINEERING (same as training)
# -------------------------------
df['syn_ack_ratio'] = df['syn_count'] / (df['ack_count'] + 1)
df['bytes_per_packet'] = df['Tot size'] / (df['Number'] + 1)
df['packet_rate'] = df['Number'] / (df['flow_duration'] + 1)
df['byte_rate'] = df['Tot size'] / (df['flow_duration'] + 1)

df['flag_total'] = (
    df['syn_flag_number'] +
    df['ack_flag_number'] +
    df['rst_flag_number'] +
    df['fin_flag_number']
)

# -------------------------------
# 6. DROP LABEL
# -------------------------------
X = df.drop(columns=['label', 'label_encoded'], errors='ignore')

# -------------------------------
# 7. LIGHTGBM
# -------------------------------
lgb_pred = np.argmax(lgb_model.predict(X), axis=1)
lgb_labels = classes[lgb_pred]

# -------------------------------
# 8. RANDOM FOREST
# -------------------------------
rf_pred = rf_model.predict(X)
rf_labels = classes[rf_pred]

# -------------------------------
# 9. DECISION TREE
# -------------------------------
dt_pred = dt_model.predict(X)
dt_labels = classes[dt_pred]

# -------------------------------
# 10. ADD RESULTS
# -------------------------------
df['LGB_Prediction'] = lgb_labels
df['RF_Prediction'] = rf_labels
df['DT_Prediction'] = dt_labels

# -------------------------------
# 11. SHOW OUTPUT
# -------------------------------
print("\nPredictions:\n")
print(df[['LGB_Prediction', 'RF_Prediction', 'DT_Prediction']].head())

# -------------------------------
# 12. SAVE OUTPUT
# -------------------------------
df.to_csv("output_iot_predictions.csv", index=False)

print("\n✅ Predictions saved to output_iot_predictions.csv")