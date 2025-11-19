
import tensorflow as tf
from src.model.siamese import make_siamese
from src.data_pipeline.dataset_builder import build_pair_dataset
from src.training.callbacks import make_callbacks
import os

def train(anchor_dir, positive_dir, negative_dir, epochs=5, batch_size=16, model_dir='models'):
    ds = build_pair_dataset(anchor_dir, positive_dir, negative_dir, batch_size=batch_size)
    model = make_siamese()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    callbacks = make_callbacks(checkpoint_dir=os.path.join(model_dir, 'checkpoints'))
    model.fit(ds, epochs=epochs, callbacks=callbacks)
    model.save(os.path.join(model_dir, 'siamese_final.h5'))
    return model
