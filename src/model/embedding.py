
from tensorflow.keras import layers, Model, Input

def make_embedding(input_shape=(100,100,3), embedding_dim=256):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inp)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(embedding_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    model = Model(inputs=inp, outputs=x, name='embedding_network')
    return model
