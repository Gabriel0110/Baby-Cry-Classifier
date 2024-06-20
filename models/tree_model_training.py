import xgboost as xgb
import pickle
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.33, stratify=y_combined)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp)

# Create a scaler object
scaler = StandardScaler()

# Fit the scaler to the training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Only transform the validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=5,  # number of classes
    n_estimators=200
)
eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
xgb_model.fit(X_train_scaled, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)

# Save the model
pickle.dump(xgb_model, open('baby_cry_reason_xgb_model', 'wb'))

# Plotting the training and validation errors
results = xgb_model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# SHAP
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_scaled)
shap.summary_plot(shap_values, X_test_scaled, plot_type="bar")