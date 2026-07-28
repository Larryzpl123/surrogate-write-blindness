"""Foundation check for the common-drive plants (run on your Mac, needs jax). No optimization.

(1) single-cell f-I on the common drive x: encoder/LIF should track HH inside the band
    x in [0.138, 1] (55-115 Hz) and DIVERGE at baseline x=0 (HH silent, toys still firing)
    and below -- i.e. reproduce match_overlay.png on jax, confirming the plants are wired right.
(2) coupling coverage: does a real W @ stim push the drive x across the silent<->active band
    for a reasonable stim amplitude? If not, bump w_scale in plants.make_coupling.

Run:  python foundation_check.py
"""
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from plants import encoder_plant, lif_plant, hh_plant, make_coupling, RHEOBASE_X

# (1) single-cell f-I: 1 cell, 1 electrode, W = [[1]], so stim = [x] drives x directly
W1 = jnp.array([[1.0]])
enc, lif, hh = encoder_plant(W1), lif_plant(W1), hh_plant(W1)
print(f"rheobase at x = {float(RHEOBASE_X):.3f} (HH jumps to ~55 Hz there)")
print("single-cell f-I on common drive x   (enc/LIF ~ HH inside band x in [0.14,1]):")
print(f"{'x':>7}{'encoder':>10}{'LIF':>10}{'HH':>10}")
for x in [-0.10, 0.00, 0.10, 0.138, 0.20, 0.445, 0.60, 0.80, 1.00]:
    s = jnp.array([x])
    print(f"{x:7.3f}{float(enc(s)[0]):10.1f}{float(lif(s)[0]):10.1f}{float(hh(s)[0]):10.1f}")
print("  expect: x=0 -> HH 0 (silent), enc ~59, LIF ~49 ; x=0.445 -> all ~85 ; x=1 -> all ~115")

# (2) coupling coverage: drive-x range for random stim at a few amplitudes
key = jax.random.PRNGKey(0)
N = 20
W = make_coupling(key, N, N)
print("\ncoupling drive coverage (need x to span ~[-0.3, 1] for silent<->active):")
for amp in [3.0, 6.0, 10.0]:
    ks = jax.random.split(key, 50)
    xr = jnp.concatenate([W @ (amp * jax.random.normal(k, (N,))) for k in ks])
    print(f"  amp={amp:4.1f}: x in [{float(xr.min()):+.2f}, {float(xr.max()):+.2f}]"
          f"   HH rate in [{float(hh(jnp.array([float(xr.min())]))[0]):.0f},"
          f" {float(hh(jnp.array([float(min(1.0,xr.max()))]))[0]):.0f}] Hz")
