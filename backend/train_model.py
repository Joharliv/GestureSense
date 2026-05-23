import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

def add_features(X):
    X = X.copy()

    x4 = X.iloc[:, 4*3]
    y4 = X.iloc[:, 4*3 + 1]

    
    x8 = X.iloc[:, 8*3]
    y8 = X.iloc[:, 8*3 + 1]
    
   
    x12 = X.iloc[:, 12*3]
    y12 = X.iloc[:, 12*3 + 1]
    
    X["thumb_index_distance"] = np.sqrt((x4 - x8)**2 + (y4 - y8)**2)
    X["index_middle_distance"] = np.sqrt((x8 - x12)**2 + (y8 - y12)**2)
    X["thumb_middle_distance"] = np.sqrt((x4 - x12)**2 + (y4 - y12)**2)

    # horizontal spread
    X["x_diff"] = abs(x8 - x12)

    return X


data = pd.read_csv("dataset.csv", header=None)
df = data.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1]   
y = df.iloc[:, -1]    

X = add_features(X)
X.columns = X.columns.astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("Model saved as model.pkl")

