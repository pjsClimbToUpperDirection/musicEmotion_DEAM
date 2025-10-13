from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

import kagglehub

from processing import total_data_extractor

# Download latest version
dataset_path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")

print("Path to dataset files:", dataset_path)
# List the files in the dataset directory
print(os.listdir(dataset_path))

# Define the paths for "DEAM_audio/MEMD_audio" and the static annotations CSV
audio_dir = os.path.join(dataset_path, 'DEAM_audio', 'MEMD_audio')
static_csv = os.path.join(dataset_path, 'DEAM_Annotations', 'annotations',
                          'annotations averaged per song', 'song_level',
                          'static_annotations_averaged_songs_1_2000.csv')

# Print to verify the paths
print("Audio Directory Path:", audio_dir)
print("Static CSV Path:", static_csv)

# Check if the paths exist
if os.path.exists(audio_dir):
    print("Audio directory exists.")
else:
    print("Audio directory does not exist!")
if os.path.exists(static_csv):
    print("Static CSV file exists.")
else:
    print("Static CSV file does not exist!")

X_by_song, y_by_song = total_data_extractor(static_csv, audio_dir)

# Get list of song IDs
song_ids = list(X_by_song.keys())

# x_song = []
# x_song.extend(X_by_song[song_ids[0]])
# print("x_song.extend(X_by_song[song_id]).shape: ", np.array(x_song).shape) #  (9, 128, 431)

# Split songs into train, validation, and test sets
train_ids, temp_ids = train_test_split(song_ids, test_size=0.3, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

# Flatten into segment-level lists
X_train, y_train = [], []
for sid in train_ids:
    X_train.extend(X_by_song[sid])
    y_train.extend(y_by_song[sid])

X_val, y_val = [], []
for sid in val_ids:
    X_val.extend(X_by_song[sid])
    y_val.extend(y_by_song[sid])

X_test, y_test = [], []
for sid in test_ids:
    X_test.extend(X_by_song[sid])
    y_test.extend(y_by_song[sid])

# Convert to numpy arrays
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Expand dimensions for CNN or LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Train segments: (10973, 128, 431, 1), Val segments: (2357, 128, 431, 1), Test segments: (2357, 128, 431, 1)
# Train labels: (10973, 2), Val labels: (2357, 2), Test labels: (2357, 2)
print(f"Train segments: {X_train.shape}, Val segments: {X_val.shape}, Test segments: {X_test.shape}")
print(f"Train labels: {y_train.shape}, Val labels: {y_val.shape}, Test labels: {y_test.shape}")


# normalize the label values
# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scaler on training labels only (flatten to 2D for scaler)
y_train_2d = y_train.reshape(-1, 2)  # Shape: (n_train_segments, 2) -> 정규화를 위하여 라벨로 주어지는 2개의 데이터를 각각 array로 변환(대괄호로 각각을 감쌈)
scaler.fit(y_train_2d)  # Compute min and max from training set only(정규화 이전 주어진 값에 따라 fitting)
joblib.dump(scaler, 'minmax_scaler.pkl') # Save the fitted scaler to a file (저장)

# Transform all sets
# linear 형식으로 반환되는 두개의 값을 정규화
print("y_train: ", y_train)
y_train_normalized = scaler.transform(y_train_2d).reshape(y_train.shape) # reshape(-1, 2) 이전 모양으로 복구
print("y_train_normalized: ", y_train_normalized)
print("y_train_normalized_reverted: ", scaler.inverse_transform(y_train_normalized))

y_val_normalized = scaler.transform(y_val.reshape(-1, 2)).reshape(y_val.shape)
y_test_normalized = scaler.transform(y_test.reshape(-1, 2)).reshape(y_test.shape)

print(f"Training label min: {scaler.data_min_}, max: {scaler.data_max_}")
print(f"Normalized training label example: {y_train_normalized[0]}")
print(f"Normalized validation label example: {y_val_normalized[0]}")
print(f"Normalized test label example: {y_test_normalized[0]}")

# Define CNN model with Input layer
model = models.Sequential([
    layers.Input(shape=(128, 431, 1)),  # Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='linear')  # Predicting valence and arousal
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Model training
history = model.fit(X_train, y_train_normalized,
                    validation_data=(X_val, y_val_normalized),
                    epochs=50,
                    batch_size=32,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10),
                        tf.keras.callbacks.ModelCheckpoint("best_cnn_model.keras", save_best_only=True)
                    ])

# Evaluate model
y_pred = model.predict(X_test)
mae_valence = np.mean(np.abs(y_pred[:, 0] - y_test_normalized[:, 0]))
mae_arousal = np.mean(np.abs(y_pred[:, 1] - y_test_normalized[:, 1]))
print(f"y_pred(native): {y_pred[:, 0]}") # example: [-321.65237 -318.12027 -294.13358 ... -314.55313 -309.49283 -301.92532]
print(f"Test MAE - Valence: {mae_valence:.4f}, Arousal: {mae_arousal:.4f}")

# Additional metrics
mse_valence = np.mean((y_pred[:, 0] - y_test_normalized[:, 0]) ** 2)
mse_arousal = np.mean((y_pred[:, 1] - y_test_normalized[:, 1]) ** 2)
rmse_valence = np.sqrt(mse_valence)
rmse_arousal = np.sqrt(mse_arousal)
print(f"Test MSE - Valence: {mse_valence:.4f}, Arousal: {mse_arousal:.4f}")
print(f"Test RMSE - Valence: {rmse_valence:.4f}, Arousal: {rmse_arousal:.4f}")


# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()