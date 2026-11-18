"""Verify GPU computation and one training batch using the actual model."""
from gpu_runtime import configure_gpu

configure_gpu()

import keras
import numpy as np
import tensorflow as tf
from model import build_model
from data import input_names

tf.config.set_soft_device_placement(False)
with tf.device('/GPU:0'):
    product = tf.linalg.matmul(tf.ones((256, 256)), tf.ones((256, 256)))
print(f'Matrix multiplication device: {product.device}; result: {product[0, 0].numpy()}')

cnn = build_model((2, len(input_names), 4))
inputs = np.zeros((32, 2, len(input_names), 4), dtype=np.float16)
labels = keras.utils.to_categorical(np.arange(32) % 2, num_classes=2)
print(f'Model weights device: {cnn.trainable_variables[0].device}')
print(f'Training batch (loss, accuracy): {cnn.train_on_batch(inputs, labels)}')
print('GPU computation and Keras training succeeded.')
