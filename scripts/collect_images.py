
from src.verification.webcam_capture import capture_and_save
import os
if __name__ == '__main__':
    # Saves into data folder (anchors/positives/negatives will be prefixed)
    save_dir = os.path.join('data')
    capture_and_save(save_dir)
