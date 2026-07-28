"""hwrite testbed: the shared write-characterization contract.

Everything model-agnostic lives here. A "plant" is any differentiable function
    plant_fn(stim) -> rates          # rates: (n_cells,) firing rate per cell
where `stim` is a (n_stim,) vector of per-channel drive, held constant over the
analysis window (coarsest target resolution = one time bin, per-cell rate only).

Rungs 0 / 1 / 3 differ ONLY in `plant_fn`. The target, distance, optimizer and
metrics below are identical across rungs. That sameness is what makes the
cross-rung comparison FAIR (and it must be able to return "no difference").
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


def make_target(key, n_cells, hi=30.0, lo=2.0, frac_hi=0.5):
    """A structured 'memory code' target: a random subset of cells at `hi`, the
    rest at `lo` (the un-stimulated baseline). Coarsest currency = per-cell rate."""
    perm = jax.random.permutation(key, n_cells)
    n_hi = int(round(frac_hi * n_cells))
    return jnp.full((n_cells,), lo).at[perm[:n_hi]].set(hi)


def distance(rates, target):
    """L2 distance on the common per-cell rate grid. Same for every rung."""
    return jnp.sqrt(jnp.sum((rates - target) ** 2))


def solve_write(plant_fn, target, n_stim, *, key, steps=2000, lr=0.05,
                amp_bound=5.0, lambda_cost=1e-3, n_restarts=6, init=None):
    """Optimize a constant per-channel stim to drive plant_fn's rates to `target`.

        loss = distance(plant(stim), target) + lambda_cost * ||stim||^2
        s.t.  |stim_i| <= amp_bound   (box constraint, native to L-BFGS-B)

    Reachability is a MAX over stimuli, ESTIMATED by optimization. A first-order
    method (Adam) under-converged here -- on the stiff LIF f-I AND, on some instances,
    even the smooth encoder -- reporting false plateaus below the true max ("the code
    ran != it found the max"). So we use L-BFGS-B (quasi-Newton, curvature-aware,
    box-constrained) with EXACT jax gradients, plus multi-start (zeros + random) and an
    optional warm-start `init` (a smaller electrode set's solution, padded) to guarantee
    monotonicity where the nested construction demands it.

    `lr` is ignored (kept for call-site stability); `steps` caps L-BFGS-B iterations.
    """
    def loss(stim):
        return distance(plant_fn(stim), target) + lambda_cost * jnp.sum(stim ** 2)

    vg = jax.jit(jax.value_and_grad(loss))

    def fun(x):  # scipy wants (float loss, float64 grad)
        v, g = vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    d_ref = float(distance(plant_fn(jnp.zeros(n_stim)), target))
    bounds = [(-amp_bound, amp_bound)] * n_stim

    starts = [np.zeros(n_stim)]
    if init is not None:
        starts.append(np.clip(np.asarray(init, dtype=np.float64), -amp_bound, amp_bound))
    ks = jax.random.split(key, n_restarts)
    starts += [np.clip(np.asarray(0.3 * amp_bound * jax.random.normal(ks[i], (n_stim,)),
                                  dtype=np.float64), -amp_bound, amp_bound)
               for i in range(n_restarts)]

    best_stim, best_d, best_losses = np.zeros(n_stim), np.inf, []
    for x0 in starts:
        trace = []
        r = minimize(fun, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                     callback=lambda xk: trace.append(fun(xk)[0]),
                     options={"maxiter": steps, "maxfun": 4 * steps})
        d = float(distance(plant_fn(jnp.asarray(r.x)), target))
        if d < best_d:
            best_d, best_stim, best_losses = d, r.x, trace

    stim = jnp.asarray(best_stim)
    return {
        "stim": stim,
        "rates": plant_fn(stim),
        "target": target,
        "dist_baseline": d_ref,
        "dist_final": best_d,
        "reachability": 1.0 - best_d / (d_ref + 1e-9),
        "stim_energy": float(jnp.sum(stim ** 2)),
        "losses": best_losses,
    }


def min_energy_to_reach(plant_fn, target, n_stim, *, key, tol_frac=0.10,
                        amp_bound=15.0, n_restarts=6):
    """Minimum stim ENERGY (||stim||^2) to drive plant_fn to within tol_frac of the
    baseline->target gap (i.e. reachability >= 1 - tol_frac). Two stages:
      1) feasibility: solve_write (min distance, multi-start). If the closest the
         plant can get still exceeds the tolerance, the target is UNREACHABLE at this
         electrode count within the amplitude budget -> {"feasible": False}.
      2) among stims that reach it, minimize ||stim||^2 s.t. distance <= tol (SLSQP
         with exact jax gradients, warm-started from the feasible point).
    Gain-matched plants share stim units, so these energies are comparable across
    rungs -- this is the metric where gain-matching actually matters (unlike reach).
    """
    d_fn = jax.jit(lambda s: distance(plant_fn(s), target))
    d_grad = jax.jit(jax.grad(lambda s: distance(plant_fn(s), target)))
    d_ref = float(d_fn(jnp.zeros(n_stim)))
    tol = tol_frac * d_ref

    feas = solve_write(plant_fn, target, n_stim, key=key,
                       amp_bound=amp_bound, lambda_cost=1e-3, n_restarts=n_restarts)
    if feas["dist_final"] > tol:
        return {"feasible": False, "best_reach": feas["reachability"]}

    x0 = np.asarray(feas["stim"], dtype=np.float64)
    r = minimize(lambda x: float(np.sum(x ** 2)), x0, method="SLSQP",
                 jac=lambda x: 2.0 * np.asarray(x, dtype=np.float64),
                 bounds=[(-amp_bound, amp_bound)] * n_stim,
                 constraints=[{"type": "ineq",
                               "fun": lambda x: float(tol - d_fn(jnp.asarray(x))),
                               "jac": lambda x: -np.asarray(d_grad(jnp.asarray(x)), dtype=np.float64)}],
                 options={"maxiter": 300})
    reach = 1.0 - float(d_fn(jnp.asarray(r.x))) / (d_ref + 1e-9)
    return {"feasible": True, "energy": float(np.sum(r.x ** 2)), "reach": reach}
