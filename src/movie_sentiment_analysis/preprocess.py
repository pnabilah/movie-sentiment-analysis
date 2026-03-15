from pathlib import Path
import numpy as np
import tensorflow as tf

def list_file_paths(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]

def train_val_test_path(dataset_path):
    dataset_path = Path(dataset_path)
    train_pos = list_file_paths(dataset_path / "train" / "pos")
    train_neg = list_file_paths(dataset_path / "train" / "neg")
    test_val_pos = list_file_paths(dataset_path / "test" / "pos")
    test_val_neg = list_file_paths(dataset_path / "test" / "neg")
    np.random.shuffle(test_val_pos)
    np.random.shuffle(test_val_neg)
    test_pos = test_val_pos[:5000]      # (5000,)
    test_neg = test_val_neg[:5000]      # (5000,)
    val_pos = test_val_pos[5000:]       # (15000,)
    val_neg = test_val_neg[5000:]       # (15000,)
    return train_pos, train_neg, test_pos, test_neg, val_pos, val_neg

def create_dataset(filepaths_positive, filepaths_negative, n_read_threads=5,
                   shuffle=False, seed=None, batch_size=32):
    # Creates 'Dataset` comprising lines from some text files
    dataset_neg = tf.data.TextLineDataset(filepaths_negative,
                                          num_parallel_reads=n_read_threads)
    dataset_pos = tf.data.TextLineDataset(filepaths_positive,
                                          num_parallel_reads=n_read_threads)
    # Map dataset with positive/negative class
    dataset_neg = dataset_neg.map(lambda review: (review, 0))
    dataset_pos = dataset_pos.map(lambda review: (review, 1))
    dataset = tf.data.Dataset.concatenate(dataset_pos, dataset_neg)    # pos1, pos2,..,neg1, neg2,...
    # Shuffle (optional)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=25_000, seed=seed)
    # Make it into batches
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)

def preprocess(dataset_path="data/raw/aclImdb"):
    # Make train, val, test dataset
    train_pos, train_neg, test_pos, test_neg, val_pos, val_neg = train_val_test_path(dataset_path)
    train_set = create_dataset(train_pos, train_neg, shuffle=True, seed=42)
    val_set = create_dataset(val_pos, val_neg)
    test_set = create_dataset(test_pos, test_neg)
    # Encoding
    max_tokens = 1000
    sample = train_set.map(lambda review, label: review)        # only extract review
    text_vectorization = tf.keras.layers.TextVectorization(max_tokens=max_tokens)
    text_vectorization.adapt(sample)
    return train_set, val_set, test_set, max_tokens, text_vectorization

if __name__ == "__main__":
    print("Hello World")