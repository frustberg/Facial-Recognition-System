
import cv2, os
from datetime import datetime

def capture_and_save(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print('Press a/p/n to save anchor/positive/negative images. q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('capture', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            fname = os.path.join(save_dir, 'anchor_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg')
            cv2.imwrite(fname, frame)
            print('Saved', fname)
        elif key == ord('p'):
            fname = os.path.join(save_dir, 'positive_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg')
            cv2.imwrite(fname, frame)
            print('Saved', fname)
        elif key == ord('n'):
            fname = os.path.join(save_dir, 'negative_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.jpg')
            cv2.imwrite(fname, frame)
            print('Saved', fname)
    cap.release()
    cv2.destroyAllWindows()
