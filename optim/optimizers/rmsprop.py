import jax
import jax.numpy as jnp
import optax


def rmsprop(rho, beta=0.9):
    EPSILON = 1e-9

    def init_fn(params):
        return {"velocity": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)}

    def update_fn(grads, state, params=None):
        velocity = state["velocity"]
        new_velocity = jax.tree_util.tree_map(
            lambda v, g: beta * v + (1 - beta) * g**2, velocity, grads
        )
        updates = jax.tree_util.tree_map(
            lambda v, g: -rho * g / (EPSILON + jnp.sqrt(v)), new_velocity, grads
        )
        return updates, {"velocity": new_velocity}

    return optax.GradientTransformation(init_fn, update_fn)
