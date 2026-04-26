"""Run this file when you want to train the review model again."""

from ml_logic import METADATA_PATH, MODEL_PATH, VECTORIZER_PATH, train_and_save_model


def train_model():
    print("Loading dataset and training model...")
    _model, _vectorizer, metadata = train_and_save_model()

    print("Training finished.")
    print(f"Rows used: {metadata['rows']}")
    print(f"Accuracy: {metadata['accuracy']}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Vectorizer saved: {VECTORIZER_PATH}")
    print(f"Details saved: {METADATA_PATH}")


if __name__ == "__main__":
    train_model()
