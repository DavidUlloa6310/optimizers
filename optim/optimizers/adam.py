import jax
import jax.numpy as jnp
import optax


def adam(rho, beta_1=0.9, beta_2=0.999):
    EPSILON = 1e-9

    def init_fn(params):
        return {
            "velocity": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
            "momentum": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
            "time_step": jnp.array(0),
        }

    def update_fn(grads, state, params=None):
        t = state["time_step"] + 1

        new_momentum = jax.tree_util.tree_map(
            lambda m, g: beta_1 * m + (1 - beta_1) * g, state["momentum"], grads
        )
        m_hat = jax.tree_util.tree_map(lambda m: m / (1 - beta_1**t), new_momentum)

        new_velocity = jax.tree_util.tree_map(
            lambda v, g: beta_2 * v + (1 - beta_2) * g**2, state["velocity"], grads
        )
        v_hat = jax.tree_util.tree_map(lambda v: v / (1 - beta_2**t), new_velocity)

        updates = jax.tree_util.tree_map(
            lambda v, m: -rho / (EPSILON + jnp.sqrt(v)) * m, v_hat, m_hat
        )
        return updates, {
            "velocity": new_velocity,
            "momentum": new_momentum,
            "time_step": t,
        }

    return optax.GradientTransformation(init_fn, update_fn)
