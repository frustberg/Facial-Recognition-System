
import tensorflow as tf
from pathlib import Path
from src.data_pipeline.preprocess import load_and_preprocess

def build_pair_dataset(anchor_dir, positive_dir, negative_dir, batch_size=16, shuffle_buffer=1024):
    anchor_paths = list(Path(anchor_dir).glob('*.jpg')) + list(Path(anchor_dir).glob('*.png'))
    positive_paths = list(Path(positive_dir).glob('*.jpg')) + list(Path(positive_dir).glob('*.png'))
    negative_paths = list(Path(negative_dir).glob('*.jpg')) + list(Path(negative_dir).glob('*.png'))

    pos_pairs = list(zip(anchor_paths, positive_paths))
    neg_pairs = list(zip(anchor_paths, negative_paths))
    pairs = [(str(a), str(p), 1) for a,p in pos_pairs] + [(str(a), str(n), 0) for a,n in neg_pairs]

    def gen():
        for a,b,label in pairs:
            yield a, b, label

    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.string, tf.string, tf.int32))
    def _map(a,b,label):
        return (load_and_preprocess(a), load_and_preprocess(b)), tf.cast(label, tf.float32)
    ds = ds.map(_map)
    ds = ds.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
