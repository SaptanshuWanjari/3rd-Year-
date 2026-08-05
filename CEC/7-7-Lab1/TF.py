import sys

import tensorflow as tf

print("Welcome to Cloud & Edge Computing for AI Lab")

# Check Python version

print("Python Version:", sys.version)

# Check TensorFlow installation
try:
    import tensorflow as tf

    print("TensorFlow Version:", tf.__version__)
except ImportError:
    print("TensorFlow is not installed.")

# Check GPU

print("GPU Available:", len(tf.config.list_physical_devices("GPU")) > 0)
