
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

def make_callbacks(checkpoint_dir='models/checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'siamese_{epoch:02d}.h5'),
                                 save_weights_only=True, save_best_only=False)
    early = EarlyStopping(patience=10, restore_best_weights=True)
    return [checkpoint, early]
