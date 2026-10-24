import jax
import jax.numpy as jnp
import optax


def _matricize(g: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Unfold a tensor into a (d_axis, prod(other dims)) matrix."""
    return jnp.moveaxis(g, axis, 0).reshape(g.shape[axis], -1)


def _inverse_root(H: jnp.ndarray, power: float, epsilon: float) -> jnp.ndarray:
    """Computes H^{-power} for a symmetric PSD matrix via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(H)
    # H >= epsilon * I by construction, but float32 eigh can return tiny or
    # negative eigenvalues for the (low rank) null space, which explode under
    # the negative power. Clamp them back to epsilon.
    eigvals = jnp.maximum(eigvals, epsilon)
    return (eigvecs * eigvals**-power) @ eigvecs.T


def _is_full(dim: int, max_precond_dim: int) -> bool:
    return dim <= max_precond_dim


def shampoo(
    rho: float,
    epsilon: float = 1e-4,
    update_every: int = 10,
    max_precond_dim: int = 2048,
):
    """
    Shampoo (Gupta, Koren & Singer, 2018) for tensors of any order.

    For an order-k tensor, each mode i keeps a statistic H_i += G_(i) G_(i)^T
    (G_(i) being the mode-i matricization of the gradient), and the update is
    the gradient contracted with H_i^{-1/(2k)} along every mode. For a matrix
    this is the familiar L^{-1/4} G R^{-1/4}.

    Inverse roots are recomputed every `update_every` steps since the
    eigendecomposition is the expensive part. Modes larger than
    `max_precond_dim` fall back to a diagonal (AdaGrad-style) statistic, which
    keeps e.g. the 10,000 x 1 IMDB layer from needing a 10k x 10k matrix.
    """

    def init_leaf(p):
        p = p.reshape(1) if p.ndim == 0 else p
        stats, roots = [], []
        for d in p.shape:
            if _is_full(d, max_precond_dim):
                stats.append(epsilon * jnp.eye(d, dtype=jnp.float32))
                roots.append(jnp.eye(d, dtype=jnp.float32))
            else:
                stats.append(jnp.full((d,), epsilon, dtype=jnp.float32))
                roots.append(jnp.ones((d,), dtype=jnp.float32))
        return stats, roots

    def init_fn(params):
        leaves, _ = jax.tree.flatten(params)
        stats, roots = zip(*[init_leaf(p) for p in leaves]) if leaves else ((), ())
        return {
            "stats": list(stats),
            "roots": list(roots),
            "time_step": jnp.array(0),
        }

    def update_leaf(g, stats, roots, t):
        shape = g.shape
        g = g.reshape(1) if g.ndim == 0 else g
        power = 1.0 / (2 * g.ndim)

        new_stats = []
        for axis, H in enumerate(stats):
            if H.ndim == 2:
                G_i = _matricize(g, axis)
                new_stats.append(H + G_i @ G_i.T)
            else:
                other_axes = tuple(a for a in range(g.ndim) if a != axis)
                new_stats.append(H + jnp.sum(g**2, axis=other_axes))

        def compute_roots(_):
            return [
                _inverse_root(H, power, epsilon) if H.ndim == 2 else H**-power
                for H in new_stats
            ]

        new_roots = jax.lax.cond(
            t % update_every == 0, compute_roots, lambda _: roots, None
        )

        preconditioned = g
        for axis, P in enumerate(new_roots):
            if P.ndim == 2:
                preconditioned = jnp.moveaxis(
                    jnp.tensordot(P, preconditioned, axes=([1], [axis])), 0, axis
                )
            else:
                bshape = [1] * g.ndim
                bshape[axis] = -1
                preconditioned = preconditioned * P.reshape(bshape)

        return -rho * preconditioned.reshape(shape), new_stats, new_roots

    def update_fn(grads, state, params=None):
        del params
        t = state["time_step"]
        leaves, treedef = jax.tree.flatten(grads)

        updates, new_stats, new_roots = [], [], []
        for g, stats, roots in zip(leaves, state["stats"], state["roots"]):
            u, s, r = update_leaf(g, stats, roots, t)
            updates.append(u)
            new_stats.append(s)
            new_roots.append(r)

        return jax.tree.unflatten(treedef, updates), {
            "stats": new_stats,
            "roots": new_roots,
            "time_step": t + 1,
        }

    return optax.GradientTransformation(init_fn, update_fn)
