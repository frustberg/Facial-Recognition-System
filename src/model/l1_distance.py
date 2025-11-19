
from tensorflow.keras.layers import Layer
import tensorflow as tf

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def call(self, inputs):
        x, y = inputs
        return tf.math.abs(x - y)
