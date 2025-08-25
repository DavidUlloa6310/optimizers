import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


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


def get_imdb():
    imdb = tfds.builder("imdb_reviews")
    imdb.download_and_prepare()

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
