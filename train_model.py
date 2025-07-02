import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')

# --- Handle missing values ---
feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
for col in feature_cols:
    if train_data[col].dtype == 'O':
        mode_value = train_data[col].mode()[0]
        train_data[col] = train_data[col].fillna(mode_value)
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(mode_value)
    else:
        mean_value = train_data[col].mean()
        train_data[col] = train_data[col].fillna(mean_value)
        if col in test_data.columns:
            test_data[col] = test_data[col].fillna(mean_value)

# --- Encode only 'Sex' and 'Embarked' ---
label_encoders = {}
for col in ['Sex', 'Embarked']:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    if col in test_data.columns:
        # Handle unseen labels in test set
        test_data[col] = test_data[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
    label_encoders[col] = le

# --- Prepare features and target ---
X_train = train_data[feature_cols]
y_train = train_data['Survived']
X_test = test_data[feature_cols]
y_test = test_data['Survived'] if 'Survived' in test_data.columns else None

# --- Scale features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# --- Save model, scaler, and encoders ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Training complete. Model, scaler, and encoders saved.")
