import json
import os
import urllib.request

import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds

KERAS_DATASETS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"


def get_mnist():
    mnist = tfds.builder("mnist")
    mnist.download_and_prepare()

    ds = {}

    for split in ["train", "test"]:
        ds[split] = tfds.as_numpy(mnist.as_dataset(split=split, batch_size=-1))

        # cast to jnp and rescale pixel values
        ds[split]["image"] = jnp.float32(ds[split]["image"]) / 255
        ds[split]["label"] = jnp.int16(ds[split]["label"])

    # shuffle our training data
    train_images = ds["train"]["image"]
    train_labels = ds["train"]["label"]

    indices = np.arange(train_images.shape[0])
    np.random.shuffle(indices)

    ds["train"]["image"] = train_images[indices]
    ds["train"]["label"] = train_labels[indices]

    return ds["train"], ds["test"]


def zca_whitening(images):
    flat_x = images.reshape(images.shape[0], -1)

    mean = jnp.mean(flat_x, axis=0)
    flat_x -= mean

    cov = jnp.cov(flat_x, rowvar=False)

    # Singular Value Decomposition of Covariance Matrix
    U, S, _ = jnp.linalg.svd(cov)

    epsilon = 1e-5
    zca_matrix = U @ jnp.diag(1.0 / jnp.sqrt(S + epsilon)) @ U.T

    whitened = flat_x @ zca_matrix
    return whitened.reshape(images.shape), mean, zca_matrix


def get_cifar():
    cifar = tfds.builder("cifar10")
    cifar.download_and_prepare()

    ds = {}

    for split in ["train", "test"]:
        ds[split] = tfds.as_numpy(cifar.as_dataset(split=split, batch_size=-1))
        ds[split]["image"] = jnp.float32(ds[split]["image"])
        ds[split]["label"] = jnp.int16(ds[split]["label"])

    # Calculate whitening parameters from training data only
    whitened_train, mean, zca_matrix = zca_whitening(ds["train"]["image"])
    ds["train"]["image"] = whitened_train

    # Apply same whitening parameters to test data
    flat_test = ds["test"]["image"].reshape(ds["test"]["image"].shape[0], -1)

    # Use mean & ZCA matrix from training data
    flat_test -= mean
    whitened_test = flat_test @ zca_matrix
    ds["test"]["image"] = whitened_test.reshape(ds["test"]["image"].shape)

    return ds["train"], ds["test"]


def _get_imdb_keras_mirror():
    """
    Rebuilds the IMDB reviews from the Keras copy hosted on Google Cloud Storage,
    for machines that can't reach ai.stanford.edu. Reviews come pre-tokenized as
    word indices, so they are decoded back to (lowercased, punctuation free)
    text, which CountVectorizer would produce from the raw text anyway.
    """
    cache_dir = os.path.expanduser("~/tensorflow_datasets/downloads/keras_imdb")
    os.makedirs(cache_dir, exist_ok=True)
    for name in ["imdb.npz", "imdb_word_index.json"]:
        path = os.path.join(cache_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{KERAS_DATASETS_URL}/{name}", path)

    with open(os.path.join(cache_dir, "imdb_word_index.json")) as f:
        index_to_word = {i: w for w, i in json.load(f).items()}

    data = np.load(os.path.join(cache_dir, "imdb.npz"), allow_pickle=True)
    rng = np.random.default_rng(113)
    splits = []
    for split in ["train", "test"]:
        text = np.array(
            [
                " ".join(index_to_word[i] for i in review if i in index_to_word)
                for review in data[f"x_{split}"]
            ]
        )
        # The archive is sorted by label, so shuffle it like tfds does
        indices = rng.permutation(len(text))
        labels = data[f"y_{split}"].astype(np.int64)
        splits.append({"text": text[indices], "label": labels[indices]})
    return splits[0], splits[1]


def get_imdb():
    try:
        imdb = tfds.builder("imdb_reviews")
        imdb.download_and_prepare()
    except Exception as e:
        print(f"Could not download imdb_reviews via tfds ({e}), using Keras mirror")
        return _get_imdb_keras_mirror()

    imdb_train, imdb_test = tfds.as_numpy(
        imdb.as_dataset(split=["train", "test"], batch_size=-1)
    )
    return imdb_train, imdb_test


def one_hot_encode(vectorizer, batch, train=False) -> jnp.ndarray:
    encoded = None
    if train:
        encoded = vectorizer.fit_transform(batch)
    else:
        encoded = vectorizer.transform(batch)

    return encoded.toarray().astype(jnp.float32)
