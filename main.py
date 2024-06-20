import pyaudio
import numpy as np
import librosa
from tensorflow import lite as tflite
import pandas as pd
import time
from collections import deque
from joblib import load
import logging
from send_email import send_email, async_send_email
import time

# Constants
RATE = 44100  # Sample rate
RECORD_SECONDS = 4  # Duration for each audio segment
CHUNKSIZE = RATE * RECORD_SECONDS  # Number of frames per audio segment
EXPECTED_FEATURE_LENGTH = 25  # Set this to the expected length of your feature vector
VALID_CLASSES = [0, 1, 2, 3, 4]  # Replace with the actual classes of your model
CLASS_NAMES = ["belly pain", "burping", "discomfort", "hungry", "tired"]  # Translated classes
TOP_TWO_DIFF_THRESHOLD = 0.03  # Threshold for difference between top two probabilities
MODEL_PATHS = [
    'models/lite_baby_cry_model_1.tflite',
    'models/lite_baby_cry_model_2.tflite',
    'models/lite_baby_cry_model_3.tflite',
    'models/lite_baby_cry_model_4.tflite',
    'models/lite_baby_cry_model_5.tflite',
    'models/lite_baby_cry_model_6.tflite',
    'models/lite_baby_cry_model_7.tflite',
    'models/lite_baby_cry_model_8.tflite',
    'models/lite_baby_cry_model_9.tflite',
    'models/lite_baby_cry_model_10.tflite'
]

# Load the scaler
SCALER = load('models/scaler.joblib')

# Set up logging
logger = logging.getLogger('baby_cry_classifier')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('baby_cry_classifier.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CryPredictor:
    def __init__(self, N=3, alpha=0.3):
        self.N = N
        self.alpha = alpha
        self.predictions = deque(maxlen=N)
        self.ema = [0] * len(VALID_CLASSES)

    def update(self, new_probs):
        self.predictions.append(new_probs)
        for i in range(len(VALID_CLASSES)):
            self.ema[i] = (1 - self.alpha) * self.ema[i] + self.alpha * new_probs[i]
        return self.get_combined_prediction()

    def get_combined_prediction(self):
        avg_prediction = np.mean(self.predictions, axis=0)
        combined_prediction = [a + b for a, b in zip(avg_prediction, self.ema)]
        return np.argmax(combined_prediction)

    def reset_predictions(self):
        self.predictions.clear()
        self.ema = [0] * len(VALID_CLASSES)

def is_silent(audio_data, threshold=0.01):
    rms = np.sqrt(np.mean(audio_data**2))
    return rms < threshold

def extract_features(signal):
    amplitude_envelope = np.abs(signal)
    amplitude_envelope_mean = np.mean(amplitude_envelope)
    rms = np.mean(librosa.feature.rms(y=signal))
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    stft = np.abs(librosa.stft(signal))
    stft_mean = np.mean(stft)
    sc = np.mean(librosa.feature.spectral_centroid(y=signal, sr=RATE))
    sban = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=RATE))
    scon = np.mean(librosa.feature.spectral_contrast(y=signal, sr=RATE))
    mfccs = librosa.feature.mfcc(y=signal, sr=RATE, n_mfcc=14)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    mfcc_13_mean = np.mean(mfccs[12, :])
    delta_mfcc_13 = np.mean(librosa.feature.delta(mfccs[12, :]))
    delta2_mfcc_13 = np.mean(librosa.feature.delta(mfccs[12, :], order=2))
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=RATE)
    mel_spec_mean = np.mean(mel_spectrogram)
    features = np.hstack([amplitude_envelope_mean, rms, zcr, stft_mean, sc, sban, scon, mfcc_13_mean, delta_mfcc_13, delta2_mfcc_13, mel_spec_mean, mfccs_processed])
    return features

def set_app_status(status):
    with open('app_current_status.txt', 'w') as file:
        file.write(status)
        logger.info(f"Set app status to {status}")

def initialize_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
    return p, stream

def load_models():
    interpreters, input_details, output_details = [], [], []
    try:
        for model_path in MODEL_PATHS:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            interpreters.append(interpreter)
            input_details.append(interpreter.get_input_details())
            output_details.append(interpreter.get_output_details())
        print("Models loaded successfully.")
        return interpreters, input_details, output_details
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        send_email("Baby Crying Classifier Error", f"Error loading model: {e}")
        set_app_status("STOPPED - ERROR")
        exit()

def read_audio_data(stream):
    try:
        data = stream.read(CHUNKSIZE, exception_on_overflow=False)
        numpy_data = np.frombuffer(data, dtype=np.float32)
        return numpy_data, None
    except Exception as e:
        return None, e

def validate_features(features):
    if len(features) != EXPECTED_FEATURE_LENGTH:
        error_msg = f"Expected feature length of {EXPECTED_FEATURE_LENGTH}, but got {len(features)}"
        return False, error_msg
    return True, None

def reshape_features(features):
    features_scaled = SCALER.transform([features])[0]
    features_reshaped = np.expand_dims(np.expand_dims(features_scaled, axis=0), axis=-1).astype(np.float32)
    return features_reshaped

def perform_inference(interpreters, input_details, output_details, features_reshaped):
    try:
        class_probabilities = []
        for i, interpreter in enumerate(interpreters):
            interpreter.set_tensor(input_details[i][0]['index'], features_reshaped)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[i][0]['index'])
            class_probabilities.append(output_data[0])

        combined_probs = np.stack(class_probabilities)
        avg_probs = combined_probs.mean(axis=0)
        
        #log_and_print_probabilities(class_probabilities, avg_probs)
        
        return avg_probs, None
    except Exception as e:
        return None, e

def log_and_print_probabilities(class_probabilities, avg_probs):
    for j in range(10):
        print(f"\nLSTM {j+1} Probabilities:")
        logger.info(f"\nLSTM {j+1} Probabilities:")
        for i, prob in enumerate(class_probabilities[j]):
            print(f"{CLASS_NAMES[i]} : \t\tProbability: {prob:.2f}")
            logger.info(f"{CLASS_NAMES[i]} : \t\tProbability: {prob:.2f}")
        print(f"\nHighest probability: {CLASS_NAMES[np.argmax(class_probabilities[j])].upper()} with probability {class_probabilities[j][np.argmax(class_probabilities[j])]:.2f}\n")
        logger.info(f"\nHighest probability: {CLASS_NAMES[np.argmax(class_probabilities[j])].upper()} with probability {class_probabilities[j][np.argmax(class_probabilities[j])]:.2f}\n")

    print("\n-----------------\n")
    logger.info("\n-----------------\n")

    print("\nCombined Average Probabilities:")
    logger.info("\nCombined Average Probabilities:")
    for i, prob in enumerate(avg_probs):
        print(f"{CLASS_NAMES[i]} : \t\tProbability: {prob:.2f}")
        logger.info(f"{CLASS_NAMES[i]} : \t\tProbability: {prob:.2f}")
    print(f"\nHighest probability: {CLASS_NAMES[np.argmax(avg_probs)].upper()} with probability {avg_probs[np.argmax(avg_probs)]:.2f}\n")
    logger.info(f"\nHighest probability: {CLASS_NAMES[np.argmax(avg_probs)].upper()} with probability {avg_probs[np.argmax(avg_probs)]:.2f}\n")

def save_detection(temp_detections, combined_pred):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    temp_detections.append((timestamp, CLASS_NAMES[combined_pred].upper()))
    df = pd.DataFrame(temp_detections, columns=["Timestamp", "Classification"])
    df.to_csv('data/detections.csv', mode='a', header=False, index=False)
    temp_detections.clear()
    logger.info(f"Added detection to file: {CLASS_NAMES[combined_pred].upper()}")

def main():
    predictor = CryPredictor()
    temp_detections = []

    interpreters, input_details, output_details = load_models()
    
    retry_count = 0
    max_retries = 10
    while retry_count < max_retries:
        try:
            audio, stream = initialize_stream()
            
            logger.info("Baby Crying Classifier started successfully.")
            print("\nBaby Crying Classifier started successfully.")
            
            logger.info("\nListening... Press Ctrl+C to stop.")
            print("\nListening... Press Ctrl+C to stop.")
            
            set_app_status("RUNNING")
            async_send_email("Baby Crying Classifier Started", "The Baby Crying Classifier has started successfully.")
            
            break
        except Exception as e:
            retry_count += 1
            logger.error(f"Error initializing microphone (attempt {retry_count}): {e}")
            print(f"Error initializing microphone (attempt {retry_count}): {e}")
            if retry_count >= max_retries:
                send_email("Baby Crying Classifier Error", f"Error initializing microphone after {max_retries} attempts: {e}")
                set_app_status("STOPPED - ERROR")
                exit()
        time.sleep(2)  # Wait for 2 seconds before retrying
    
    print("\n\n")

    audio_error_count = 0
    feature_extraction_error_count = 0

    try:
        while True:
            numpy_data, audio_error = read_audio_data(stream)
            if audio_error:
                logger.error(f"Error reading audio data: {audio_error}")
                print(f"Error reading audio data: {audio_error}")
                audio_error_count += 1
                if audio_error_count >= 10:
                    logger.error("Too many errors reading audio data. Exiting...")
                    print("Too many errors reading audio data. Exiting...")
                    send_email("Baby Crying Classifier Error", "Too many errors reading audio data.")
                    set_app_status("STOPPED - ERROR")
                    exit()
                continue

            audio_error_count = 0

            if is_silent(numpy_data):
                print("Silence detected.")
                logger.info("Silence detected.")
                predictor.reset_predictions()
                continue

            try:
                features = extract_features(numpy_data)
                feature_extraction_error_count = 0
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                feature_extraction_error_count += 1
                if feature_extraction_error_count >= 10:
                    logger.error("Too many errors extracting features. Exiting...")
                    print("Too many errors extracting features. Exiting...")
                    send_email("Baby Crying Classifier Error", "Too many errors extracting features.")
                    set_app_status("STOPPED - ERROR")
                    exit()
                continue

            valid_features, error_msg = validate_features(features)
            if not valid_features:
                logger.error(error_msg)
                print(error_msg)
                send_email("Baby Crying Classifier Error", error_msg)
                set_app_status("STOPPED - ERROR")
                exit()

            features_reshaped = reshape_features(features)

            if not np.array_equal(features_reshaped.shape, input_details[0][0]['shape']):
                error_msg = f"Expected input shape of {input_details[0][0]['shape']}, but got {features_reshaped.shape}"
                logger.error(error_msg)
                print(error_msg)
                send_email("Baby Crying Classifier Error", error_msg)
                set_app_status("STOPPED - ERROR")
                exit()

            avg_probs, inference_error = perform_inference(interpreters, input_details, output_details, features_reshaped)
            if inference_error:
                logger.error(f"Error during model inference: {inference_error}")
                print(f"Error during model inference: {inference_error}")
                send_email("Baby Crying Classifier Error", f"Error during model inference: {inference_error}")
                set_app_status("STOPPED - ERROR")
                exit()

            sorted_probs = np.sort(avg_probs)[::-1]
            if sorted_probs[0] - sorted_probs[1] <= TOP_TWO_DIFF_THRESHOLD:
                logger.info("Uncertain of prediction - top two probabilities are too close. Attempting inference again...")
                print("Uncertain of prediction - top two probabilities are too close. Attempting inference again...")
                continue

            combined_pred = predictor.update(avg_probs)
            save_detection(temp_detections, combined_pred)

            logger.info(f"Combined final prediction: {CLASS_NAMES[combined_pred].upper()}")
            print(f"Combined final prediction: {CLASS_NAMES[combined_pred].upper()}")

    except KeyboardInterrupt:
        pass
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if audio:
            audio.terminate()
        logger.info("Baby Crying Classifier stopped successfully.")
        print("Stopped listening.")
        print("Baby Crying Classifier stopped successfully.")
        send_email("Baby Crying Classifier Stopped", "The Baby Crying Classifier has stopped.")
        set_app_status("STOPPED")

if __name__ == "__main__":
    main()
