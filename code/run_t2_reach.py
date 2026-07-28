"""T2: reachability distribution over random in-band targets, encoder / LIF / HH vs electrode count.

Targets: A {silent 0, active 85} and B {60, 100} -- both inside the common writable band [55,115].
For each (plant, target-type, n_stim) we sample N_TARGETS random cell assignments, solve the write
(L-BFGS-B, warm-started nested electrodes), and collect reachability = 1 - dist_final/dist_baseline.
Reports mean/median reach + success rate (reach >= SUCCESS) vs electrode count, and saves a figure.

Speed: value_and_grad is jitted ONCE per (plant, n_stim) and reused across all targets, so this is
~30 compiles, not one per solve. Amplitude budget AMP is deliberately generous here so reach is
limited by electrode count / coupling rank, NOT amplitude (T4 tightens AMP to probe that instead).

Run from code/ :  python run_t2_reach.py
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)   # LIF f-I is stiff near threshold; float32 froze the optimizer

from testbed import make_target, distance
from plants import make_coupling, encoder_plant, lif_plant, hh_plant
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK, RHEOBASE_X

# --- target-informed init: place each cell near its target drive via HH's f-I inverse, so an
#     active cell starts ABOVE the Type-II jump. Below the jump HH's gradient is dead, which
#     traps L-BFGS in a FALSE plateau (verified: pure-init reach hits 1.0 at n=20). We seed with
#     the least-squares stim that realizes x_target; the gradient step then refines every plant. ---
_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
_xg = np.linspace(RHEOBASE_X, 1.05, 800)                       # rising band only -> _rg increasing
_rg = _sig((I_BLOCK * _xg - HH_A) / HH_B) * (HH_C + HH_D * I_BLOCK * _xg)


def x_target_from_rates(rates):
    """Per-cell desired drive x: HH f-I inverse for rates in the band, silent (-0.3) below it."""
    return np.array([-0.3 if r < 54.0 else float(np.interp(r, _rg, _xg)) for r in np.asarray(rates)])

N_CELLS = 20
COUNTS = [4, 8, 12, 16, 20]
N_TARGETS = 12
AMP = 15.0            # generous: T2 measures STRUCTURAL reach, so amplitude must not bind (T4 tightens it)
LAM = 1e-4
SUCCESS = 0.90
RNG = np.random.default_rng(0)

PLANTS = {"encoder": encoder_plant, "LIF": lif_plant, "HH": hh_plant}
TARGETS = {"A(0/85)": dict(hi=85.0, lo=0.0), "B(60/100)": dict(hi=100.0, lo=60.0)}

key = jax.random.PRNGKey(0)
kW, kT = jax.random.split(key, 2)
W_full = make_coupling(kW, N_CELLS, N_CELLS)          # w_scale default 2.0
tkeys = jax.random.split(kT, N_TARGETS)


def make_solver(plant_fn, n_stim):
    """Jitted value_and_grad over (stim, target); compiles once per plant+n_stim, reused across
    targets. Returns solve(target, starts) -> (reachability, best_stim)."""
    def loss(stim, target):
        return distance(plant_fn(stim), target) + LAM * jnp.sum(stim ** 2)
    vg = jax.jit(jax.value_and_grad(loss))
    bounds = [(-AMP, AMP)] * n_stim

    def solve(target, starts):
        def fun(x):
            v, g = vg(jnp.asarray(x), target)
            return float(v), np.asarray(g, dtype=np.float64)
        d_ref = float(distance(plant_fn(jnp.zeros(n_stim)), target))
        best_d, best_stim = np.inf, np.zeros(n_stim)
        for x0 in starts:
            res = minimize(fun, x0, method="L-BFGS-B", jac=True, bounds=bounds,
                           options={"maxiter": 2000, "maxfun": 8000})
            d = float(distance(plant_fn(jnp.asarray(res.x)), target))
            if d < best_d:
                best_d, best_stim = d, res.x
        return 1.0 - best_d / (d_ref + 1e-9), best_stim, best_d
    return solve


fig_mean = {}
for tname, tp in TARGETS.items():
    targets = [make_target(tk, N_CELLS, hi=tp["hi"], lo=tp["lo"]) for tk in tkeys]
    print(f"\n=== target {tname} ===   (mean reach over {N_TARGETS} targets; sr = frac reach >= {SUCCESS})")
    print(f"{'plant':>8}   " + "   ".join(f"n={c:<2d}" for c in COUNTS))
    for pname, pfn in PLANTS.items():
        reach = np.zeros((N_TARGETS, len(COUNTS)))
        resid = np.zeros((N_TARGETS, len(COUNTS)))    # absolute dist_final (Hz): cross-plant fair
        warm = [None] * N_TARGETS
        for j, ns in enumerate(COUNTS):
            solver = make_solver(pfn(W_full[:, :ns]), ns)
            for i in range(N_TARGETS):
                starts = [np.zeros(ns)]
                if warm[i] is not None:
                    starts.append(np.clip(np.concatenate([warm[i], np.zeros(ns - len(warm[i]))]), -AMP, AMP))
                starts += [np.clip(0.3 * AMP * RNG.standard_normal(ns), -AMP, AMP) for _ in range(2)]
                xt = x_target_from_rates(np.asarray(targets[i]))            # target-informed init
                stim0 = np.linalg.lstsq(np.asarray(W_full[:, :ns]), xt, rcond=None)[0]
                starts.append(np.clip(stim0, -AMP, AMP))
                rc, st, dd = solver(targets[i], starts)
                reach[i, j] = rc
                resid[i, j] = dd
                warm[i] = np.asarray(st)
        mean = reach.mean(0); sr = (reach >= SUCCESS).mean(0)
        fig_mean[(tname, pname)] = mean
        print(f"{pname:>8}   " + "   ".join(f"{m:4.2f}" for m in mean))
        print(f"{'sr':>8}   " + "   ".join(f"{s:4.2f}" for s in sr))
        print(f"{'resid Hz':>8}   " + "   ".join(f"{r:4.0f}" for r in resid.mean(0)))

# figure: reach vs electrodes, one panel per target
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
axes = np.atleast_1d(plt.subplots(1, len(TARGETS), figsize=(4.2 * len(TARGETS), 3.6), sharey=True)[1])
for ax, tname in zip(axes, TARGETS):
    for pname in PLANTS:
        ax.plot(COUNTS, fig_mean[(tname, pname)], "o-", label=pname)
    ax.set_title(f"target {tname}"); ax.set_xlabel("electrode count"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
axes[0].set_ylabel("mean reachability"); axes[0].legend(fontsize=8)
plt.suptitle("T2: reachability vs electrodes (realism ladder)")
plt.tight_layout(); plt.savefig("../figures/t2_reach.png", dpi=130)
print("\nsaved ../figures/t2_reach.png")
