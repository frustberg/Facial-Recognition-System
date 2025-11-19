
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense
from src.model.embedding import make_embedding
from src.model.l1_distance import L1Dist

def make_siamese(input_shape=(100,100,3), embedding_dim=256):
    embedding = make_embedding(input_shape, embedding_dim)
    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    embed_1 = embedding(input_1)
    embed_2 = embedding(input_2)

    l1 = L1Dist()([embed_1, embed_2])
    out = Dense(1, activation='sigmoid')(l1)
    model = Model(inputs=[input_1, input_2], outputs=out, name='siamese_network')
    return model
