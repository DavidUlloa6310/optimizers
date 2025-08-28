import jax.numpy as jnp
import jax
import optax


def label_2d(params: jnp.ndarray) -> jnp.ndarray:
    return jax.tree.map(lambda p: "2d_params" if p.ndim == 2 else "non_2d_params", params)


def _newtonshulz5(G: jnp.ndarray, steps: int = 5, eps: float = 1e-7) -> jnp.ndarray:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.astype(jnp.bfloat16)
    X /= (jnp.linalg.norm(X) + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


def tree_filter(tree: jnp.ndarray, pred: callable[[jnp.ndarray], bool]) -> jnp.ndarray:
    return jax.tree.map(lambda x: x if pred(x) else None, tree)


def muon(rho: float, momentum: float = 0.95):

    def init_fn(params: optax.Params):
        return {
            "beta": jax.tree.map(lambda p: jnp.zeros_like(p), params)
        }

    def update_fn(grads: jnp.ndarray, state: dict[str, jnp.ndarray], params: optax.Params | None = None):
        del params
        beta = jax.tree.map(lambda b, g: b * momentum +
                            g, state["beta"], grads)
        O = jax.tree.map(lambda b: _newtonshulz5(b)
                         if b.ndim == 2 else b, beta)
        return jax.tree.map(lambda o: - rho * o, O), {"beta": beta}

    return optax.GradientTransformation(init_fn, update_fn)
