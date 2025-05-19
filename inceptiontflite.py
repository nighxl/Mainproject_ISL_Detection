import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

# Load the InceptionV3 model
base_model_inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Convert the model to TFLite with explicit configuration
converter = tf.lite.TFLiteConverter.from_keras_model(base_model_inception)

# Enable experimental new converter (critical for some models)
converter.experimental_new_converter = True

# Allow custom operations (if needed)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS      # Enable TensorFlow select ops (for unsupported operations)
]

# Optional: Apply quantization to reduce model size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save the model
tflite_model = converter.convert()

TFLITE_MODEL_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/inception_model.tflite"
with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved successfully!")