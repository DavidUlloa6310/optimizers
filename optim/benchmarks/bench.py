import os
import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax.training import train_state
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import optax

from optim.benchmarks.datasets import (get_mnist, get_cifar,
                                       zca_whitening, get_imdb, one_hot_encode)
from optim.benchmarks.models import (MNISTRegression, mnist_eval_fn, mnist_update_fn, CNN,
                                     cifar_update_fn, cifar_eval_fn, IMDBClassifier, imdb_update_fn, imdb_eval_fn)
from optim.optimizers.adam import adam
from optim.optimizers.adagrad import adagrad
from optim.optimizers.nesterov_sgd import sgd_nesterov
from optim.optimizers.sgd import sgd, sgd_momentum
from optim.optimizers.rmsprop import rmsprop


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int

    # Dataset (x, y)
    data: jnp.ndarray
    labels: jnp.ndarray


@dataclass
class TestConfig:
    batch_size: int
    data: jnp.ndarray
    labels: jnp.ndarray


@dataclass
class Metrics:
    loss: list = field(default_factory=list)
    accuracy: list = field(default_factory=list)
    grad_norm: list = field(default_factory=list)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process a directory.')
    parser.add_argument('--graph-directory',
                        type=str,
                        required=True,
                        help='Path to the directory to process')

    parser.add_argument('--mnist',
                        action='store_true',
                        help='Should benchmark MNIST dataset')

    parser.add_argument('--cifar',
                        action='store_true',
                        help='Should benchmark Cifar dataset')

    parser.add_argument('--imdb',
                        action='store_true',
                        help='Should benchmark IMDB dataset')

    args = parser.parse_args()
    if not os.path.isdir(args.graph_directory):
        os.makedirs(args.graph_directory)

    return args


def batch_iterator(data, labels, batch_size):
    num_samples = data.shape[0]
    for start_i in range(0, num_samples, batch_size):
        end_i = min(start_i + batch_size, num_samples)
        yield data[start_i:end_i], labels[start_i:end_i]


def graph_results(metrics: Metrics, dataset: str, optimizer: str, path: str):
    # Assuming that the number of batches is the length of the loss list.
    batches = len(metrics.loss)
    x_axis = np.linspace(1, batches, batches)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    ax1.set_xlabel("Batch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.plot(x_axis, metrics.loss, color="blue")
    ax1.set_title(f"Loss ({dataset}, {optimizer})")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.set_xlabel("Batch")
    ax2.set_ylabel("Accuracy", color="red")
    ax2.plot(x_axis, metrics.accuracy, color="red")
    ax2.set_title(f"Accuracy ({dataset}, {optimizer})")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax3.set_xlabel("Batch")
    ax3.set_ylabel("Grad Norm", color="green")
    ax3.plot(x_axis, metrics.grad_norm, color="green")
    ax3.set_title(f"Grad Norm ({dataset}, {optimizer})")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3.grid(True, linestyle='--', alpha=0.7)

    fig.suptitle(f"Training Metrics ({dataset}, {optimizer})", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle

    plt.savefig(os.path.join(path, f"{dataset}_{optimizer}.png"))


def train_model(state, update_fn, config, rng):
    metrics = Metrics()
    batches = math.ceil(len(config.data) / config.batch_size)
    for epoch in tqdm(range(config.epochs)):

        loss, accuracy = 0, 0
        get_batch = batch_iterator(
            config.data, config.labels, config.batch_size)
        for x, y in get_batch:
            rng, dropout_rng = jax.random.split(rng)
            state, batch_loss, batch_accuracy, grad_norm = update_fn(
                state, (x, y), rng)

            metrics.loss.append(batch_loss)
            metrics.accuracy.append(batch_accuracy)
            metrics.grad_norm.append(grad_norm)
            loss += batch_loss
            accuracy += batch_accuracy

        loss /= batches
        accuracy /= batches
        print(f"Epoch {epoch+1}: Loss: {loss:.4f}, "
              f"Train Accuracy: {accuracy:.4f}")
    return state, metrics


def eval_model(state, eval_fn, config):
    get_batch = batch_iterator(config.data, config.labels, config.batch_size)
    batches = math.ceil(len(config.data) / config.batch_size)

    test_loss, test_accuracy = 0, 0
    for x, y in get_batch:
        batch_loss, batch_accuracy = eval_fn(state, x, y)
        test_loss += batch_loss
        test_accuracy += batch_accuracy

    test_loss /= batches
    test_accuracy /= batches
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


def bench_mnist(*optimizers: tuple[optax.GradientTransformation, ...]):
    mnist_train, mnist_test = get_mnist()
    mnist_train['image'] = mnist_train['image'].reshape(-1, 784)
    mnist_test['image'] = mnist_test['image'].reshape(-1, 784)

    print("MNIST Training Data Shapes:\n", "Images: ",
          mnist_train['image'].shape, "\nLabels: ", mnist_train['label'].shape, "\n")
    print("MNIST Test Data Shapes: \n", "Images: ",
          mnist_test['image'].shape, "\nLabels: ", mnist_test['label'].shape)

    mnist_config = TrainingConfig(
        epochs=40,
        batch_size=128,
        data=mnist_train['image'],
        labels=mnist_train['label'],
    )

    for name, optimizer in optimizers:
        model = MNISTRegression()
        rng, model_rng = jax.random.split(rng)

        params = model.init(model_rng, jnp.ones((1, 784)))
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer)

        state, metrics = train_model(
            state, mnist_update_fn, mnist_config, rng)
        test_loss, test_accuracy = eval_model(state, mnist_eval_fn, TestConfig(
            batch_size=mnist_config.batch_size,
            data=mnist_test['image'],
            labels=mnist_test['label']
        ))

        graph_results(metrics, "MNIST", name, args.graph_directory)


def bench_cifar(*optimizers: tuple[optax.GradientTransformation, ...]):
    cifar_train, cifar_test = get_cifar()
    print("CIFAR-10 Training Data Shapes:\n", "Images: ",
          cifar_train['image'].shape, "\nLabels: ", cifar_train['label'].shape, "\n")
    print("CIFAR-10 Test Data Shapes: \n", "Images: ",
          cifar_test['image'].shape, "\nLabels: ", cifar_test['label'].shape)

    cifar_config = TrainingConfig(
        epochs=40,
        batch_size=128,
        data=cifar_train['image'],
        labels=cifar_train['label']
    )
    for name, optimizer in optimizers:
        model = CNN()
        rng, model_rng = jax.random.split(rng)

        params = model.init(model_rng, jnp.ones((1, 32, 32, 3)), True)
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer)
        state, metrics = train_model(
            state, cifar_update_fn, cifar_config, rng=rng)
        test_loss, test_accuracy = eval_model(state, cifar_eval_fn, TestConfig(
            batch_size=cifar_config.batch_size,
            data=cifar_test['image'],
            labels=cifar_test['label']
        ))

        graph_results(metrics, "Cifar", name, args.graph_directory)


def bench_imdb(*optimizers: tuple[optax.GradientTransformation, ...]):
    imdb_train, imdb_test = get_imdb()
    print("IMDB Training Data:\n", "Images: ",
          imdb_train['text'].shape, "\nLabels: ", imdb_train['label'].shape, "\n")
    print("IMDB Test Data: \n", "Images: ",
          imdb_train['text'].shape, "\nLabels: ", imdb_test['label'].shape)

    vectorizer = CountVectorizer(max_features=10_000)
    imdb_train['text'] = one_hot_encode(
        vectorizer, imdb_train['text'], train=True)
    imdb_test['text'] = one_hot_encode(vectorizer, imdb_test['text'])

    model = IMDBClassifier()
    rng, model_rng = jax.random.split(rng)

    params = model.init(model_rng, jnp.ones((1, 10_000)), True)
    imdb_config = TrainingConfig(
        epochs=180,
        batch_size=128,
        data=imdb_train['text'],
        labels=imdb_train['label']
    )

    for name, optimizer in optimizers:
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer)

        state, metrics = train_model(
            state, imdb_update_fn, imdb_config, rng=rng)
        test_loss, test_accuracy = eval_model(state, imdb_eval_fn, TestConfig(
            batch_size=imdb_config.batch_size,
            data=imdb_test['text'],
            labels=imdb_test['label']
        ))

        graph_results(metrics, "IMDB", name, args.graph_directory)


if __name__ == "__main__":
    args = parse_arguments()
    rng = jax.random.key(42)

    optimizers = [
        ("SGD", sgd(0.001)),
        ("SGD_Momentum", sgd_momentum(0.01, 0.9)),
        ("Nesterov_SGD", sgd_nesterov(0.01, 0.9)),
        ("Adagrad", adagrad(0.001)),
        ("RMSProp", rmsprop(0.001)),
        ("Adam", adam(0.001))
    ]

    if args.mnist:
        bench_mnist(*optimizers)

    if args.cifar:
        bench_cifar(*optimizers)

    if args.imdb:
        bench_imdb(*optimizers)
