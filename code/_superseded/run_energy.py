"""Energy metric: minimum stim energy to WRITE a fixed memory-code target, encoder
(rung 0) vs LIF (rung 1), gain-matched, at full electrodes, across a few instances.

Reachability barely depends on the plant; write ENERGY does. Reports the toy-vs-LIF
energy ratio per random instance, and flags instances where the toy says the target
is UNREACHABLE while LIF can still write it.

Caveat: the ratio's MAGNITUDE depends on the target rate scale and the LIF params;
the robust claims are the DIRECTION (toy over-estimates the write cost) and that it
is large / sometimes flips feasibility. Run:  python run_energy.py
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from testbed import make_target, min_energy_to_reach
from plants import make_coupling, encoder_plant, lif_plant

N_CELLS, N_STIM, AMP, TOL = 20, 20, 15.0, 0.10

print(f"min stim energy to write the target to reach >= {1 - TOL:.2f}, n_stim={N_STIM}")
print("seed |    enc E |    LIF E | ratio LIF/enc")
for seed in range(4):
    ks = jax.random.split(jax.random.PRNGKey(seed), 4)
    target = make_target(ks[0], N_CELLS)
    W = make_coupling(ks[1], N_CELLS, N_CELLS)[:, :N_STIM]
    e = min_energy_to_reach(encoder_plant(W), target, N_STIM, key=ks[2], tol_frac=TOL, amp_bound=AMP)
    l = min_energy_to_reach(lif_plant(W), target, N_STIM, key=ks[2], tol_frac=TOL, amp_bound=AMP)
    es = f"{e['energy']:.0f}" if e["feasible"] else "unreach"
    ls = f"{l['energy']:.0f}" if l["feasible"] else "unreach"
    ratio = f"{l['energy'] / e['energy']:.3f}" if (e["feasible"] and l["feasible"]) else "--"
    print(f"  {seed}  | {es:>8} | {ls:>8} | {ratio}")
