"""Measure the single-compartment HH f-I curve in JAXLEY (real biophysics), to be
fit as a smooth differentiable surrogate for rung 3 (see topic_scouting.md).

Run on your Mac (needs jaxley):   python measure_hh_fi.py
It prints (I_amp, rate_Hz) points and saves hh_fi.npy + hh_fi.png. Send me the printed
points (or the png) and I'll fit the differentiable surrogate + build hh_plant, then
run reach + energy on the full encoder / LIF / HH ladder.

NOTE: built from the JAXLEY docs; I can't run jaxley in my sandbox, so the exact API
calls and the current RANGE (AMPS) may need a small tweak on first run. If it errors,
paste the traceback; if all rates are 0 or all saturate, we just shift AMPS.
"""
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# --- compat shim: jaxley 0.13.0 calls jnp.clip(x, a_max=...); this jax removed the
#     a_min/a_max kwargs (now min/max). Translate them so jaxley runs unchanged. ---
_orig_clip = jnp.clip
def _clip_compat(x, *args, **kw):
    if "a_min" in kw:
        kw["min"] = kw.pop("a_min")
    if "a_max" in kw:
        kw["max"] = kw.pop("a_max")
    return _orig_clip(x, *args, **kw)
jnp.clip = _clip_compat

import jaxley as jx
from jaxley.channels import HH

DT = 0.025        # ms integration step
T = 300.0         # ms total window
I_DELAY = 50.0    # ms before stim onset (let the onset transient settle)
SS_MS = 200.0     # measure STEADY-STATE rate over the last SS_MS ms: excludes the onset
                  # transient AND reads depolarization block as rate 0 (a plateau has no late spikes)
THRESH = 0.0      # mV, upward-crossing = spike
# fine + low where the action is (rheobase -> repetitive firing), a few higher to confirm block:
AMPS = np.round(np.concatenate([np.arange(0.0, 0.030, 0.001), [0.04, 0.06, 0.10, 0.20]]), 4)


def measure(i_amp):
    cell = jx.Cell()
    cell.insert(HH())
    cell.record("v")
    stim = jx.step_current(i_delay=I_DELAY, i_dur=T - I_DELAY, i_amp=float(i_amp),
                           delta_t=DT, t_max=T)
    cell.stimulate(stim)
    v = np.asarray(jx.integrate(cell, delta_t=DT)).ravel()
    seg = v[int((T - SS_MS) / DT):]                       # steady-state window
    above = seg > THRESH
    spikes = int(np.sum((~above[:-1]) & (above[1:])))
    rate = spikes / (SS_MS / 1000.0)                      # Hz (steady state)
    v_late = float(np.mean(v[int((T - 30.0) / DT):]))     # mean V over last 30 ms
    block = rate < 1.0 and v_late > -40.0                 # plateaued high, not firing = depol block
    return rate, v_late, block


fi = []
for a in AMPS:
    rate, v_late, block = measure(a)
    fi.append((float(a), float(rate)))
    tag = "   [depol block]" if block else ""
    print(f"  I={a:.4f}  rate={rate:6.1f} Hz   Vlate={v_late:6.1f} mV{tag}")

arr = np.array(fi)
np.save("../results/hh_fi.npy", arr)
print("saved ../results/hh_fi.npy")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.plot(arr[:, 0], arr[:, 1], "o-")
    plt.xlabel("I_amp"); plt.ylabel("rate (Hz)")
    plt.title("single-compartment HH f-I (JAXLEY)")
    plt.tight_layout(); plt.savefig("../figures/hh_fi.png", dpi=130)
    print("saved ../figures/hh_fi.png")
except Exception as e:
    print("plot skipped:", e)
