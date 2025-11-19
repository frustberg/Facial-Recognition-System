
from src.training.train import train
from pathlib import Path
import os

if __name__ == '__main__':
    anchor = os.path.join('data', 'anchor')
    positive = os.path.join('data', 'positive')
    negative = os.path.join('data', 'negative')
    Path('models').mkdir(parents=True, exist_ok=True)
    model = train(anchor, positive, negative, epochs=3, batch_size=8, model_dir='models')
    print('Training complete. Model saved to models/siamese_final.h5')
