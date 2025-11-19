
# Facial Verification using a Siamese Neural Network

A complete deep-learning based **facial verification system** that compares two face images and predicts whether they belong to the same person. Built using a **Siamese Network**, custom CNN embeddings, and TensorFlow.

## Project Contents
- `src/` - modular python package (data pipeline, model, training, verification)
- `scripts/` - runnable scripts for training, capturing data and running verification
- `data/` - sample folders for anchors / positives / negatives and verification images
- `notebooks/` - original Jupyter notebook (analysis & experiments)
- `models/` - saved model weights / checkpoints (gitignored by default)

## Quickstart
1. Create a virtual environment and activate it.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Collect images (webcam):
   ```bash
   python scripts/collect_images.py
   ```
4. Train model:
   ```bash
   python scripts/train_model.py
   ```
5. Run verification:
   ```bash
   python scripts/verify_person.py
   ```

