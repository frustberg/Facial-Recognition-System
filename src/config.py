
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 16
AUTOTUNE = None
DETECTION_THRESHOLD = 0.6
VERIFICATION_THRESHOLD = 0.6
