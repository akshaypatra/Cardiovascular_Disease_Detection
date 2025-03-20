import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import Sequential, layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import wfdb
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loading & Preprocessing ECG Dataset

data_path = os.path.join(os.path.dirname(__file__), "mit-bih-arrhythmia-database", "100")
record = wfdb.rdrecord(data_path)
annotation = wfdb.rdann(data_path, 'atr')

# Extracting ECG signal & labels

ecg_signal = record.p_signal[:, 0]  
labels = annotation.sample  

# Normalizing ECG signal

scaler = StandardScaler()
ecg_signal = scaler.fit_transform(ecg_signal.reshape(-1, 1)).flatten()

# Encoding class labels

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Preparing Data for Training

seq_length = 200  

X, y = [], []
for i in range(len(ecg_signal) - seq_length):
    X.append(ecg_signal[i: i + seq_length])
    closest_index = np.argmin(np.abs(annotation.sample - i))
    y.append(labels[closest_index])

X = np.array(X)
y = np.array(y)



assert len(X) == len(y), f"Mismatch: X has {len(X)} samples, but y has {len(y)} labels."

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(-1, seq_length, 1)
X_test = X_test.reshape(-1, seq_length, 1)

#  Building the CNN-LSTM Model
model = Sequential([
    layers.Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(seq_length, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2),

    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(np.unique(y)), activation='softmax')  
])

# Compiling Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#  Defining Callbacks for Better Training
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

#  Training the Model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

#  Evaluating the Model

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

#  Visualizing Model Performance

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("CNN-LSTM ECG Classification Performance")
plt.show()

#  Saving the Model

model.save("ecg_cardiovascular_model_optimized.h5")
