import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle, json, os

train = pd.read_csv('data/processed/train.csv')
X_train = train.drop('Churn', axis=1)
y_train = train['Churn']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

y_pred = model.predict(X_train)
acc = accuracy_score(y_train, y_pred)

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f, indent=2)