import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv("dataset_motion.csv", header=None)
df = data.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X.columns = X.columns.astype(str)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save
with open("model_motion.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("Motion model saved")