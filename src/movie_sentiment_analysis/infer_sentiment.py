import tensorflow as tf
from tensorflow.keras.models import load_model

def analyze(model, texts: list[str]) -> list[str]:
    """
    Predict sentiments for a batch of movie reviews using the given model.

    Args:
        model: A trained Keras/TensorFlow model.
        texts (list[str]): List of raw movie review strings.

    Returns:
        list[str]: List of predicted sentiment labels (e.g., 'positive', 'negative').
    """
    model = load_model(model)
    probs = model(tf.constant(texts))
    labels = ["positive" if p > 0.5 else "negative" for p in probs.numpy().flatten()]
    return labels
