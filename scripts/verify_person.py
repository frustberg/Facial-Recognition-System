
import os
from tensorflow.keras.models import load_model
from src.verification.verify import verify
if __name__ == '__main__':
    model_path = os.path.join('models', 'siamese_final.h5')
    if not os.path.exists(model_path):
        print('Model not found. Train first.')
        exit(1)
    model = load_model(model_path, compile=False)
    input_image = os.path.join('data', 'input_image.jpg')  # user should save input image here
    verification_dir = os.path.join('data', 'verification')
    if not os.path.exists(input_image):
        print('Input image not found. Save an image to data/input_image.jpg or modify the script.')
        exit(1)
    verified, positives, ratio = verify(model, input_image, verification_dir)
    print('Verified:', verified, 'positives:', positives, 'ratio:', ratio)
