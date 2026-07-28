"""Rung 0 run: abstract encoder + write optimization.

  (1) one write at limited electrodes (n_stim < n_cells): prints reachability /
      energy and plots target-vs-achieved + the loss curve (proves convergence);
  (2) a first CHARACTERIZATION curve: reachability vs electrode count, using a
      fixed NESTED coupling. The sweep WARM-STARTS each larger electrode set from
      the previous solution (padded with zeros), so the curve is guaranteed
      monotonic -- any dip would be an optimizer artifact, which this removes.

Run from this folder (in your venv):  python run_rung0.py
"""
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testbed import make_target, solve_write
from plants import make_coupling, encoder_plant

jax.config.update("jax_platform_name", "cpu")  # M2: CPU (jax-metal GPU is experimental)
jax.config.update("jax_enable_x64", True)  # float64: rung 1's LIF f-I needs it (stiff 1/g); harmless for the encoder

N_CELLS, AMP, SEED = 20, 15.0, 0
keys = jax.random.split(jax.random.PRNGKey(SEED), 4)
target = make_target(keys[0], N_CELLS)
W_full = make_coupling(keys[1], N_CELLS, N_CELLS)  # 20 electrodes max; nested subsets below

# (1) one write with limited electrodes (first 12 of the 20)
N_STIM = 12
res = solve_write(encoder_plant(W_full[:, :N_STIM]), target, N_STIM, key=keys[2],
                  steps=3000, lr=0.05, amp_bound=AMP, lambda_cost=1e-3)
print(f"[single write]  cells={N_CELLS}  stim_channels={N_STIM}  amp_bound={AMP}")
print(f"  dist baseline->target : {res['dist_baseline']:.3f}")
print(f"  dist final            : {res['dist_final']:.3f}")
print(f"  reachability (0..1)   : {res['reachability']:.3f}")
print(f"  stim energy (||s||^2) : {res['stim_energy']:.3f}")

# (2) characterization: reachability vs electrode count (nested + warm-started)
counts, reach, prev = [4, 8, 12, 16, 20], [], None
for ns in counts:
    init = None if prev is None else jnp.concatenate([prev, jnp.zeros(ns - prev.shape[0])])
    r = solve_write(encoder_plant(W_full[:, :ns]), target, ns, key=keys[2],
                    steps=3000, lr=0.05, amp_bound=AMP, lambda_cost=1e-3, init=init)
    reach.append(r["reachability"]); prev = r["stim"]
    print(f"[sweep] n_stim={ns:2d}  reachability={r['reachability']:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
idx = jnp.arange(N_CELLS)
ax[0].bar(idx - 0.2, target, width=0.4, label="target")
ax[0].bar(idx + 0.2, res["rates"], width=0.4, label="achieved")
ax[0].set_xlabel("cell"); ax[0].set_ylabel("rate")
ax[0].set_title(f"target vs achieved (n_stim={N_STIM})"); ax[0].legend()
ax[1].plot(res["losses"]); ax[1].set_xlabel("step"); ax[1].set_ylabel("loss")
ax[1].set_title("optimization (convergence)")
ax[2].plot(counts, reach, "o-"); ax[2].set_xlabel("electrode count (n_stim)")
ax[2].set_ylabel("reachability"); ax[2].set_ylim(0, 1)
ax[2].set_title("characterization: reach vs electrodes")
fig.tight_layout(); fig.savefig("rung0_encoder.png", dpi=130)
print("saved rung0_encoder.png")
