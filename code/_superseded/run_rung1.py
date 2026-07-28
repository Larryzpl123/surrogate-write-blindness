"""Rung 1 run: LIF vs the rung-0 encoder, same testbed / target / coupling.

Produces the first realism-ladder comparison: reachability vs electrode count for
the abstract encoder (rung 0) and the LIF f-I plant (rung 1), overlaid. Both sweeps
are WARM-STARTED (nested electrodes) so each curve is monotonic by construction.

FAIRNESS CAVEAT (read before drawing conclusions): the two plants are matched at the
BASELINE rate (~2 Hz at stim=0) but NOT in gain (f-I slope). So a reach difference
here mixes "model shape (threshold + saturation)" with "overall gain". A clean
realism-only claim needs a stated matching convention -- the natural next step is to
also match the f-I slope at the baseline operating point. Treat this run as the
machinery + first signal, not a publishable "realism changes the answer" result yet.

Run from this folder (in your venv):  python run_rung1.py
"""
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testbed import make_target, solve_write
from plants import make_coupling, encoder_plant, lif_plant

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)  # LIF f-I has a stiff 1/g near threshold; float32 froze the optimizer (dead gradients), float64 fixes it

N_CELLS, AMP, SEED = 20, 15.0, 0
keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
target = make_target(keys[0], N_CELLS)
W_full = make_coupling(keys[1], N_CELLS, N_CELLS)
COUNTS = [4, 8, 12, 16, 20]


def sweep(plant_of_W):
    """Warm-started nested sweep: reach vs electrode count for one plant family."""
    reach, prev = [], None
    for ns in COUNTS:
        init = None if prev is None else jnp.concatenate([prev, jnp.zeros(ns - prev.shape[0])])
        r = solve_write(plant_of_W(W_full[:, :ns]), target, ns, key=keys[2],
                        steps=3000, lr=0.05, amp_bound=AMP, lambda_cost=1e-3, init=init)
        reach.append(r["reachability"]); prev = r["stim"]
    return reach


enc_reach = sweep(lambda W: encoder_plant(W))
lif_reach = sweep(lambda W: lif_plant(W))

print(f"{'n_stim':>7} {'encoder(r0)':>12} {'LIF(r1)':>10}")
for ns, e, l in zip(COUNTS, enc_reach, lif_reach):
    print(f"{ns:7d} {e:12.3f} {l:10.3f}")
print("\nNOTE: baseline AND gain matched (lif_plant gain_match=True); divergence here is f-I shape (realism).")
print("      reachability is a weak probe of realism -- see run_energy.py for the sharp one (write energy).")

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.plot(COUNTS, enc_reach, "o-", label="rung 0: encoder (toy)")
ax.plot(COUNTS, lif_reach, "s-", label="rung 1: LIF f-I")
ax.set_xlabel("electrode count (n_stim)"); ax.set_ylabel("reachability")
ax.set_ylim(0, 1); ax.set_title("realism ladder: reach vs electrodes")
ax.legend()
fig.tight_layout(); fig.savefig("rung1_vs_rung0.png", dpi=130)
print("saved rung1_vs_rung0.png")
