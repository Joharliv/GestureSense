import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle

# Load dataset
df = pd.read_csv("dataset_lstm.csv", header=None)

# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Reshape for LSTM
timesteps = 30
features = int(X.shape[1] / timesteps)

X = X.reshape(-1, timesteps, features)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps, features)))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Save model
model.save("gesture_lstm.h5")

print("Model trained and saved!")