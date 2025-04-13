import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


def l2_norm(grads):
    leaves, _ = jax.tree_util.tree_flatten(grads)
    return jnp.sqrt(sum([jnp.sum(leaf**2) for leaf in leaves]))

# ===== MNIST Utils ====


class MNISTRegression(nn.Module):
    """
    Logistic Regression Model for MNIST, as used in the Adam paper.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=784)(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def mnist_update_fn(state, batch, rng):

    def loss_fn(params, state, x, y):
        logits = state.apply_fn(params, x)
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, y))

        accuracy = jnp.mean(jnp.argmax(logits, axis=- 1) == y)
        return loss, accuracy

    x, y = batch
    (loss, accuracy), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params, state, x, y)
    state = state.apply_gradients(grads=grads)
    grad_norm = l2_norm(grads)

    return state, loss, accuracy, grad_norm


@jax.jit
def mnist_eval_fn(state, x, y):
    logits = state.apply_fn(state.params, x)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    accuracy = jnp.mean(jnp.argmax(logits, axis=- 1) == y)
    return loss, accuracy

# ===== Cifar Utils ====


class CNN(nn.Module):
    """
    CNN Model for CIFAR-10, as used in the Adam paper.
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool):
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)

        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.relu(x)

        x = nn.Conv(features=64, kernel_size=(5, 5))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.relu(x)

        x = nn.Conv(features=128, kernel_size=(5, 5))(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=1000)(x)
        x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def cifar_update_fn(state, batch, rng):

    def loss_fn(params, state, x, y):
        logits = state.apply_fn(params, x, training=True, rngs={
            "dropout": rng
        })
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, y))

        accuracy = jnp.mean(jnp.argmax(logits, axis=- 1) == y)
        return loss, accuracy

    x, y = batch
    (loss, accuracy), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params, state, x, y)
    state = state.apply_gradients(grads=grads)
    grad_norm = l2_norm(grads)

    return state, loss, accuracy, grad_norm


@jax.jit
def cifar_eval_fn(state, x, y):
    logits = state.apply_fn(state.params, x, training=False)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
    accuracy = jnp.mean(jnp.argmax(logits, axis=- 1) == y)
    return loss, accuracy

# ===== IMDB Utils ====


class IMDBClassifier(nn.Module):
    """
    Dense layer with heavy dropout, as used in the Adam paper.
    """
    @nn.compact
    def __call__(self, x, training):
        x = nn.Dropout(rate=0.5, deterministic=not training)(x)
        x = nn.Dense(features=1)(x)
        return x


@jax.jit
def imdb_update_fn(state, batch, rng):

    def loss_fn(params, state, x, y, rng):
        logits = state.apply_fn(params, x, training=True, rngs={
            "dropout": rng
        })
        logits = jnp.squeeze(logits, axis=-1)
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

        probs = jax.nn.sigmoid(logits)
        preds = (probs > 0.5).astype(jnp.int32)
        accuracy = jnp.mean(preds == y)
        return loss, accuracy

    x, y = batch
    (loss, accuracy), grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params, state, x, y, rng)
    state = state.apply_gradients(grads=grads)
    grad_norm = l2_norm(grads)

    return state, loss, accuracy, grad_norm


@jax.jit
def imdb_eval_fn(state, x, y):
    logits = state.apply_fn(state.params, x, training=False)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))
    probs = jax.nn.sigmoid(logits)

    preds = (probs > 0.5).astype(jnp.int32)
    accuracy = jnp.mean(preds[:, 0] == y)
    return loss, accuracy
