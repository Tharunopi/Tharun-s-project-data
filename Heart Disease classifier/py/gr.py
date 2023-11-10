import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

data_path = 'C:\\Users\\tharu\\PycharmProjects\\pythondfe\\heart_disease_data.csv'
data = pd.read_csv(data_path)

X = data.drop(columns=['target']).values
y = data['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
model.save("heart_model.h5")

model = tf.keras.models.load_model("heart_model.h5")

import numpy as np
from sklearn.preprocessing import StandardScaler

custom_input = np.array([[37, 1, 2, 130, 250, 0, 1, 187, 0, 3.5, 0, 0, 2]])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

custom_input_scaled = scaler.transform(custom_input)

predictions = model.predict(custom_input_scaled)

threshold = 0.5
predicted_labels = (predictions > threshold).astype(int)

print("Predicted Labels:", predicted_labels)
