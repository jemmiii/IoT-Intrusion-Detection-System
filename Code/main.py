import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
df = pd.read_csv("Code/IoT_Intrusion.csv")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

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

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df.drop(columns=['label', 'label_encoded'])
y = df['label_encoded']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))


train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',

    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,

    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,

    'lambda_l1': 0.1,
    'lambda_l2': 0.2,

    'verbosity': -1
}


model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    num_boost_round=1000,
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(100)
    ]
)


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)


print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


print("\n🔹 Random Forest Results:\n")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\n🔹 Decision Tree Results:\n")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

import matplotlib.pyplot as plt

lgb.plot_importance(model, max_num_features=20)
plt.title("Top Feature Importance")
plt.show()

model.save_model("intrusion_model.txt")


np.save("label_classes.npy", le.classes_)


test_df = df.sample(100)
test_df.to_csv("Code/test_data.csv", index=False)

print("✅ test_data.csv ")

print(X.columns)

import joblib

# Save models
joblib.dump(rf, "rf_model.pkl")
joblib.dump(dt, "dt_model.pkl")

print("✅ RF & DT models saved")