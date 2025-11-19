
import numpy as np
import os
from src.data_pipeline.preprocess import load_and_preprocess

def verify(model, input_image_path, verification_dir, detection_threshold=0.6, verification_threshold=0.6):
    positives = 0
    trials = 0
    input_img = np.expand_dims(load_and_preprocess(input_image_path).numpy(), axis=0)
    for img_name in os.listdir(verification_dir):
        img_path = os.path.join(verification_dir, img_name)
        val_img = np.expand_dims(load_and_preprocess(img_path).numpy(), axis=0)
        result = model.predict([input_img, val_img])
        if result[0][0] > detection_threshold:
            positives += 1
        trials += 1
    if trials == 0:
        return False, 0, 0.0
    ratio = positives / float(trials)
    verified = ratio > verification_threshold
    return verified, positives, ratio
