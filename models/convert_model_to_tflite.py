import tensorflow as tf
from tensorflow.keras.models import load_model


# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model("models/baby_cry_reason_lstm_model_2")
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

# Save the converted model
with open("models/lite_baby_cry_model_2.tflite", "wb") as f:
    f.write(tflite_model)
