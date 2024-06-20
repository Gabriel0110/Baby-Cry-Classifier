import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import os
from joblib import dump, load
import matplotlib.pyplot as plt

# Constants
RATE = 44100  # Sample rate
RECORD_SECONDS = 4  # Duration for each audio segment
CHUNKSIZE = RATE * RECORD_SECONDS  # Number of frames per audio segment
EXPECTED_FEATURE_LENGTH = 25  # Set this to the expected length of your feature vector
VALID_CLASSES = [0, 1, 2, 3, 4]  # Replace with the actual classes of your model
DATASET_PATH = './data/donateacry-corpus_features_final.csv' # Path to the dataset

# Augmentation function
def augment_features(features, method):
    if method == "noise":
        noise = np.random.normal(0, 0.01, features.shape)
        return features + noise
    elif method == "scaling":
        scaling_factor = np.random.uniform(0.9, 1.1)
        return features * scaling_factor
    elif method == "mixup":
        alpha = 0.2  # mixup interpolation coefficient
        sample = df.sample(n=1).drop(columns=['Cry_Audio_File', 'Cry_Reason']).values
        return alpha * features + (1 - alpha) * sample
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

# Load the data
df = pd.read_csv(DATASET_PATH)

AUGMENTATIONS = {
    "noise": 3,   # apply noise augmentation once
    "scaling": 1,  # apply scaling augmentation once
    "mixup": 0    # don't apply mixup
}

# Original data
X_original = df.drop(columns=['Cry_Audio_File', 'Cry_Reason']).values
y_original = df['Cry_Reason'].values

# Create augmented data
X_augmented = []
y_augmented = []

for i in range(X_original.shape[0]):
    for method, num_times in AUGMENTATIONS.items():
        for _ in range(num_times):
            augmented = augment_features(X_original[i], method)
            X_augmented.append(augmented)
            y_augmented.append(y_original[i])

X_augmented = np.vstack(X_augmented)
y_augmented = np.array(y_augmented)

# Combine original and augmented data
X_combined = np.vstack([X_original, X_augmented])
y_combined = np.hstack([y_original, y_augmented])

print(X_combined.shape)
print(y_combined.shape)

# Shuffle the data
np.random.seed(42)
shuffle_indices = np.random.permutation(X_combined.shape[0])
X_combined = X_combined[shuffle_indices]
y_combined = y_combined[shuffle_indices]

# Split the data into training, validation, and test sets
# X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.33, stratify=y_combined)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp)

# Create a scaler object
scaler = StandardScaler()

# Check if we have a trained scaler
if os.path.exists('models/scaler.joblib'):
    # Load the trained scaler
    scaler = load('models/scaler.joblib')
    
    # Transform the data
    X_scaled = scaler.transform(X_combined)
else:
    # Fit the scaler to the data and transform
    X_scaled = scaler.fit_transform(X_combined)
    
    # Save the trained scaler
    dump(scaler, 'models/scaler.joblib')

# Only transform the validation and test data
# X_val_scaled = scaler.transform(X_val)
# X_test_scaled = scaler.transform(X_test)


# def build_lstm_model(input_dim):
#     model = Sequential()
#     model.add(LSTM(256, return_sequences=True, input_shape=(input_dim, 1)))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(256, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(256, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(256, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(BatchNormalization())
#     model.add(LSTM(32))
#     model.add(Dropout(0.2))
#     model.add(Dense(5, activation='softmax'))  # 5 classes
#     optimizer = Adam(learning_rate=0.002)
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model

# model = build_lstm_model(X_scaled.shape[1])

# #Add a learning rate reducer callback
# lr_reducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, min_lr=0.00001)

# early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# #LSTMs expect input shape in the form of (samples, time steps, features), so we need to reshape our data
# X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))
# #X_val_lstm = np.reshape(X_val_scaled, (X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
# #X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# #Train the enhanced LSTM model
# history = model.fit(X_lstm, y_combined, epochs=50, batch_size=16, callbacks=[lr_reducer, early_stopping])

# #Save the model
# model.save('models/baby_cry_reason_large_lstm_model_1')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_combined, test_size=0.2, stratify=y_combined, random_state=42)

rocket = Rocket()
rocket.fit(X_train)
X_train_transformed = rocket.transform(X_train)
X_test_transformed = rocket.transform(X_test)

print(X_train.shape)
print(X_train_transformed.shape)
print(y_train.shape)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced')
classifier.fit(X_train_transformed, y_train)

accuracy = classifier.score(X_test_transformed, y_test)
print(f"Accuracy: {accuracy:.4f}")

exit()



# Test the model
# loss, accuracy = model.evaluate(X_test_scaled, y_test)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plotting the training and validation loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
#plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plotting the training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.show()