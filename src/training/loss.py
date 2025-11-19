
import tensorflow as tf
def bce_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)
