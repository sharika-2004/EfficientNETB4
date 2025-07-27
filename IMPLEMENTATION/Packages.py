!pip uninstall -y tf-keras tensorflow-text tensorflow-decision-forests tensorflow
!pip install tensorflow==2.18.0
!pip install keras_applications
!pip install kaggle
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
