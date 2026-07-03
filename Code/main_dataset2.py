import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("Code/train_test_network.csv")

# -------------------------------
# 2. CLEANING
# -------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# -------------------------------
# 3. HANDLE OBJECT COLUMNS (IMPORTANT)
# -------------------------------
labels = df['type']   # save label

df = df.select_dtypes(exclude=['object'])  # remove string columns

df['type'] = labels   # add label back

# -------------------------------
# 4. LABEL ENCODING
# -------------------------------
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['type'])

X = df.drop(columns=['type', 'label', 'label_encoded'], errors='ignore')
y = df['label_encoded']

# -------------------------------
# 5. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 6. CLASS WEIGHTS
# -------------------------------
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

# -------------------------------
# 7. LIGHTGBM
# -------------------------------
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'verbosity': -1
}

model = lgb.train(
    params,
    train_data,
    valid_sets=[valid_data],
    num_boost_round=200
)

# -------------------------------
# 8. LIGHTGBM RESULT
# -------------------------------
y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n🔹 LightGBM Results:\n")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# 9. RANDOM FOREST
# -------------------------------
print("\n🔹 Random Forest Results:\n")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# -------------------------------
# 10. DECISION TREE
# -------------------------------
print("\n🔹 Decision Tree Results:\n")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

# -------------------------------
# 11. SAVE TEST SAMPLE
# -------------------------------
test_df = df.sample(100)
test_df.to_csv("Code/test_data_dataset2.csv", index=False)

print("✅ test_data_dataset2.csv created")
import joblib

# -------------------------------
# SAVE MODELS (Dataset 2 ke liye alag naam)
# -------------------------------
model.save_model("intrusion_model_dataset2.txt")

joblib.dump(rf, "rf_model_dataset2.pkl")
joblib.dump(dt, "dt_model_dataset2.pkl")

print("✅ Dataset 2 models saved")