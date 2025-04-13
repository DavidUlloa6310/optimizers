import jax
import jax.numpy as jnp
import optax


def sgd(learning_rate):

    def init_fn(params):
        return {}

    def update_fn(grads, state, params=None):
        updates = jax.tree_util.tree_map(lambda g: -learning_rate * g, grads)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def sgd_momentum(learning_rate, rho):
    def init_fn(params):
        return {
            "velocity": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        }

    def update_fn(grads, state, params=None):
        velocity = state["velocity"]
        updates = jax.tree_map(lambda v, g: rho * v -
                               learning_rate * g, velocity, grads)
        return updates, {"velocity": updates}

    return optax.GradientTransformation(init_fn, update_fn)
