"""THE RESULT: a smooth differentiable surrogate -- the compute-light standard for model-based
write optimization -- erases the mechanistic writable-set boundary (HH's Type-II onset floor at
~55 Hz and its depolarization block above ~115 Hz). A write the surrogate CERTIFIES as perfect is
silently WRONG on the real neuron, and nothing in the pipeline flags it.

Two demonstrations, both against the SAME faithful HH f-I measured in JAXLEY (results/hh_fi.npy):
  Part 1  single cell: request rate rho, compare what the surrogate claims vs what real HH delivers.
  Part 2  network write (full electrodes): optimize stim on the surrogate for a mixed rate code,
          then evaluate that stim on real HH. Surrogate residual ~0 ("success"); true residual large.

Pure numpy/scipy (the surrogate f-I is matched_params, identical to plants.hh_plant; the write is a
plain L-BFGS solve -- the optimizer is not the point, the surrogate!=reality gap is). Run from code/:
    python run_surrogate_blindness.py
Writes figures/surrogate_blindness.png. The boundary is real (it is in the raw JAXLEY f-I); only the
smooth surrogate loses it, and smoothness is exactly what gradient-based write optimization requires.
"""
import numpy as np
from scipy.optimize import minimize
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK

# --- the smooth surrogate (identical to plants.hh_plant's f-I) ---
_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
surrogate = lambda x: _sig((I_BLOCK * x - HH_A) / HH_B) * (HH_C + HH_D * I_BLOCK * x)   # drive x -> claimed rate

# --- the faithful HH f-I, straight from the raw JAXLEY measurement (has the jump AND the block) ---
_raw = np.load("../results/hh_fi.npy"); _I, _r = _raw[:, 0], _raw[:, 1]
_RHEO_I, _BLOCK_I = 0.004, 0.030            # rheobase (Type-II jump to 55 Hz) and depol-block onset
_rise = (_I >= _RHEO_I) & (_I <= _BLOCK_I)


def true_hh(x):
    """Real HH rate for common drive x. 0 below rheobase (Type-II: no rates in (0,55)) and above
    block; the measured rising curve in between. This is ground truth (it came out of JAXLEY)."""
    I = I_BLOCK * np.atleast_1d(np.asarray(x, float))
    out = np.zeros_like(I)
    m = (I >= _RHEO_I) & (I <= _BLOCK_I)
    out[m] = np.interp(I[m], _I[_rise], _r[_rise])
    return out


_xg = np.linspace(0.0, 1.55, 60000); _rg = surrogate(_xg)
surr_inv = lambda rho: float(_xg[np.argmin(np.abs(_rg - rho))])      # what drive the surrogate says gives rho

# ============================ PART 1: single cell ============================
print("PART 1  single cell -- request rate rho; what does the surrogate claim vs real HH deliver?")
print(f"  {'rho(want)':>9} {'surrogate(claims)':>18} {'realHH(actual)':>16} {'':>4}")
rhos = [0, 15, 30, 45, 55, 70, 85, 100, 115, 125, 140]
for rho in rhos:
    x = surr_inv(rho); a = float(true_hh(x)[0])
    tag = "OK" if abs(a - rho) < 8 else "SILENTLY WRONG"
    print(f"  {rho:9.0f} {surrogate(x):18.1f} {a:16.1f}   {tag}")

# ============================ PART 2: network write ============================
print("\nPART 2  network write, full 20 electrodes, mixed rate code (gap + band + block cells):")
rng = np.random.default_rng(1)
N = 20
W = (2.0 / np.sqrt(N)) * rng.standard_normal((N, N))
target = np.empty(N)
perm = rng.permutation(N)
target[perm[:8]] = 30.0        # in the FORBIDDEN gap (0,55): HH cannot fire here
target[perm[8:16]] = 85.0      # in band [55,115]: HH is faithful (control)
target[perm[16:]] = 125.0      # above depol block: HH cannot fire here either

xt = np.array([surr_inv(t) for t in target])
x0 = np.linalg.lstsq(W, xt, rcond=None)[0]
loss = lambda s: float(np.sum((surrogate(W @ s) - target) ** 2))
res = minimize(loss, x0, method="L-BFGS-B", bounds=[(-15, 15)] * N, options={"maxiter": 10000})
x_star = W @ res.x
claim = surrogate(x_star)
real = true_hh(x_star)
res_surr = float(np.sqrt(np.sum((claim - target) ** 2)))
res_true = float(np.sqrt(np.sum((real - target) ** 2)))
n_wrong = int(np.sum(np.abs(real - target) > 20))
print(f"  surrogate residual  (pipeline CLAIMS): {res_surr:6.2f} Hz   -> reports SUCCESS")
print(f"  true HH residual    (ACTUAL neuron):   {res_true:6.2f} Hz   -> write FAILED")
print(f"  cells >20 Hz wrong on real HH: {n_wrong}/{N}; nothing in the pipeline flags it")
for region, val in [("gap 30Hz", 30.0), ("band 85Hz", 85.0), ("block 125Hz", 125.0)]:
    idx = np.abs(target - val) < 1
    print(f"    {region:>11}: mean surrogate={claim[idx].mean():6.1f}  mean realHH={real[idx].mean():6.1f}")

# ============================ figure ============================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# panel 1: claimed vs actual sweep
rr = np.linspace(0, 145, 400)
actual = np.array([float(true_hh(surr_inv(r))[0]) for r in rr])
ax1.axvspan(55, 115, color="green", alpha=0.08)
ax1.plot(rr, rr, "--", color="gray", label="surrogate certifies (= request)")
ax1.plot(rr, actual, "-", lw=2.4, color="C3", label="real HH delivers")
ax1.set_xlabel("requested rate (Hz)"); ax1.set_ylabel("resulting rate (Hz)")
ax1.set_title("surrogate certifies writes real HH cannot perform")
ax1.text(20, 120, "forbidden\ngap", fontsize=8, color="C3", ha="center")
ax1.text(132, 120, "depol\nblock", fontsize=8, color="C3", ha="center")
ax1.text(85, 30, "band\n[55,115]\nfaithful", fontsize=8, color="green", ha="center")
ax1.legend(fontsize=8, loc="upper left"); ax1.set_xlim(0, 145); ax1.set_ylim(-5, 150)

# panel 2: network per-cell
order = np.argsort(target)
ax2.plot(range(N), target[order], "ks", ms=6, label="target code", mfc="none")
ax2.plot(range(N), claim[order], "o", ms=5, color="C0", label="surrogate claims")
ax2.plot(range(N), real[order], "x", ms=7, color="C3", label="real HH actual")
ax2.set_xlabel("cell (sorted by target rate)"); ax2.set_ylabel("rate (Hz)")
ax2.set_title(f"one write: surrogate says 0.0 Hz off, real HH is {res_true:.0f} Hz off")
ax2.legend(fontsize=8, loc="center left"); ax2.set_ylim(-5, 150)

fig.tight_layout(); fig.savefig("../figures/surrogate_blindness.png", dpi=130)
print("\nsaved ../figures/surrogate_blindness.png")
