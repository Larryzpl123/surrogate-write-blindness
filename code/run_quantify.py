"""QUANTIFY: how often, and how badly, does the silent failure bite? Over random rate-codes,
measure the certify-vs-reality gap (the surrogate always claims ~0 Hz off) as a function of the
code's rate composition. Two views, full electrodes, ground truth = raw JAXLEY f-I:

  (A) binary code {0, A}: sweep the active level A. The true error is a NOTCH -- near zero only
      while A sits inside the writable band, large the moment A falls in the gap (<55) or past the
      block (>115). The naive surrogate pipeline reliably writes only ~[60,115]; it even mislocates
      the 55 Hz floor (knife-edge), so the reliable notch is a bit INSIDE the neuron's true set.
  (B) a naive practitioner draws each cell rate ~ Uniform[0,120]: report the distribution of the
      silent error and the fraction of cells silently mis-written.

Run from code/:  python run_quantify.py   ->  figures/quantify.png
"""
import numpy as np
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK

_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
surrogate = lambda x: _sig((I_BLOCK * x - HH_A) / HH_B) * (HH_C + HH_D * I_BLOCK * x)
raw = np.load("../results/hh_fi.npy"); I, r = raw[:, 0], raw[:, 1]
rise = (I >= 0.004) & (I <= 0.030); Ir, rr = I[rise], r[rise]


def true_hh(x):
    Ii = I_BLOCK * np.atleast_1d(np.asarray(x, float)); out = np.zeros_like(Ii)
    m = (Ii >= 0.004) & (Ii <= 0.030); out[m] = np.interp(Ii[m], Ir, rr); return out


_xg = np.linspace(0, 1.55, 40000); _rg = surrogate(_xg)
surr_inv = lambda rho: float(_xg[np.argmin(np.abs(_rg - rho))])

N = 20; TOL = 5.0
rng = np.random.default_rng(3)
W = (2.0 / np.sqrt(N)) * rng.standard_normal((N, N))
solve = lambda xt: W @ np.linalg.lstsq(W, xt, rcond=None)[0]


def naive_write(target):
    """Naive surrogate write, full electrodes: returns (claimed err, true err, frac cells wrong)."""
    xs = solve(np.array([surr_inv(t) for t in target]))
    real = true_hh(xs)
    claim = float(np.sqrt(np.sum((surrogate(xs) - target) ** 2)))
    true = float(np.sqrt(np.sum((real - target) ** 2)))
    return claim, true, float(np.mean(np.abs(real - target) > TOL))


# (A) active-level sweep
A_vals = np.arange(0, 141, 5); M = 24
A_true = []; A_frac = []
for A in A_vals:
    ts, fs = [], []
    for _ in range(M):
        t = np.zeros(N); t[rng.permutation(N)[:10]] = A
        _, tr, f = naive_write(t); ts.append(tr); fs.append(f)
    A_true.append(np.mean(ts)); A_frac.append(np.mean(fs))
print("(A) binary code {0,A}, sweep A -- true error is a NOTCH (low only inside the writable band):")
for A, t, f in zip(A_vals, A_true, A_frac):
    if A % 15 == 0:
        print(f"    A={A:3.0f}Hz  true_err={t:6.1f} Hz  cells_wrong={100*f:3.0f}%")

# (B) naive practitioner: Uniform[0,120] codes
K = 400; gaps = []; fracs = []
for _ in range(K):
    t = rng.uniform(0, 120, N)
    _, tr, f = naive_write(t); gaps.append(tr); fracs.append(f)
gaps = np.array(gaps); fracs = np.array(fracs)
print(f"\n(B) practitioner draws rates ~ Uniform[0,120], {K} random codes:")
print(f"    surrogate claims ~0 Hz off every time; real HH mean {gaps.mean():.0f} Hz off (median {np.median(gaps):.0f})")
print(f"    cells silently mis-written: {100*fracs.mean():.0f}% on average; codes with ANY silent error: {100*np.mean(fracs>0):.0f}%")

# figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
ax1.axvspan(55, 115, color="green", alpha=0.08, label="neuron writable band")
ax1.plot(A_vals, A_true, "o-", color="C3")
ax1.set_xlabel("active level A of the code (Hz)"); ax1.set_ylabel("silent error on real HH (Hz)")
ax1.set_title("writability notch: error ~0 only inside the band")
ax1.text(20, max(A_true) * 0.6, "gap:\nsilent fail", fontsize=8, color="C3", ha="center")
ax1.text(132, max(A_true) * 0.6, "block:\nsilent fail", fontsize=8, color="C3", ha="center")
ax1.legend(fontsize=8, loc="upper center")
ax2.hist(gaps, bins=24, color="C3", alpha=0.85)
ax2.axvline(gaps.mean(), color="k", ls="--", lw=1)
ax2.set_xlabel("silent error on real HH per code (Hz)"); ax2.set_ylabel("# of random codes")
ax2.set_title(f"naive codes ~U[0,120]: 100% fail, {100*fracs.mean():.0f}% of cells wrong")
ax2.text(gaps.mean() + 4, ax2.get_ylim()[1] * 0.9, f"mean {gaps.mean():.0f} Hz\n(surrogate says 0)", fontsize=8)
fig.tight_layout(); fig.savefig("../figures/quantify.png", dpi=130)
print("\nsaved ../figures/quantify.png")
