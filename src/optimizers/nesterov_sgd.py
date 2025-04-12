import jax
import jax.numpy as jnp
import optax


def sgd_nesterov(learning_rate, rho):

    def init_fn(params):
        return {
            "velocity": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)
        }

    def update_fn(grads, state, params=None):
        velocity = state["velocity"]

        # calculate v_{t+1}
        new_velocity = jax.tree_util.tree_map(
            lambda v, g: rho * v + g, velocity, grads)
        # actual step, considering v_{t+1} on our gradients
        updates = jax.tree_util.tree_map(
            lambda v, g: -learning_rate * (rho * v + g), velocity, grads)

        return updates, {"velocity": new_velocity}

    return optax.GradientTransformation(init_fn, update_fn)
