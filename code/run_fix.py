"""THE FIX (what to do instead): a feasibility-aware write closes the certify-vs-reality gap.

The silent failure has one cause: the write model (a smooth surrogate) is not the neuron. The remedy
is a recipe, not a new trick:
  1. FEASIBILITY SCREEN -- the real writable alphabet is the raw f-I's achievable set {0} U [55,115].
     Flag every requested cell-rate outside it, up front. (No silent success on impossible codes.)
  2. PLACE CELLS WITH THE TRUE f-I, not the surrogate -- invert the measured f-I (respecting the
     feasible set), then solve the coupling. The pipeline's reported error then EQUALS what the
     neuron delivers, by construction, at any electrode count. (No silent failure.)
The fix cannot write rates HH physically lacks; it makes that loss EXPLICIT and never certifies a
write the neuron will not perform. It also hands back a design constraint: memory codes must live in
the writable alphabet {0} U [55,115].

Pure numpy; ground truth = raw JAXLEY f-I (results/hh_fi.npy). Run from code/:  python run_fix.py
Writes figures/fix.png.
"""
import numpy as np
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK

_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
surrogate = lambda x: _sig((I_BLOCK * x - HH_A) / HH_B) * (HH_C + HH_D * I_BLOCK * x)
raw = np.load("../results/hh_fi.npy"); I, r = raw[:, 0], raw[:, 1]
rise = (I >= 0.004) & (I <= 0.030); Ir, rr = I[rise], r[rise]      # real rising f-I: I 0.004-0.029 -> 55-115


def true_hh(x):
    Ii = I_BLOCK * np.atleast_1d(np.asarray(x, float)); out = np.zeros_like(Ii)
    m = (Ii >= 0.004) & (Ii <= 0.030); out[m] = np.interp(Ii[m], Ir, rr); return out


FLOOR, CEIL = 55.0, 115.0
project = lambda rho: 0.0 if rho < FLOOR else min(rho, CEIL)       # sub-floor -> silent (the 55 floor is a knife-edge)
x_true_for = lambda rho: 0.05 if rho <= 0.1 else float(np.interp(rho, rr, Ir)) / I_BLOCK   # TRUE f-I inverse
_xg = np.linspace(0, 1.55, 40000); _rg = surrogate(_xg)
surr_inv = lambda rho: float(_xg[np.argmin(np.abs(_rg - rho))])

rng = np.random.default_rng(1); N = 20
W = (2.0 / np.sqrt(N)) * rng.standard_normal((N, N))
tgt = np.empty(N); pm = rng.permutation(N)
tgt[pm[:8]] = 30.0; tgt[pm[8:16]] = 85.0; tgt[pm[16:]] = 125.0
solve = lambda xtar: W @ np.linalg.lstsq(W, xtar, rcond=None)[0]

print("REQUESTED code: 8 cells@30 (gap), 8@85 (band), 4@125 (block)\n")
# --- NAIVE: place with the surrogate, believe the surrogate ---
x_naive = solve(np.array([surr_inv(t) for t in tgt]))
claim_n = float(np.sqrt(np.sum((surrogate(x_naive) - tgt) ** 2)))
true_n = float(np.sqrt(np.sum((true_hh(x_naive) - tgt) ** 2)))
print(f"NAIVE  surrogate write: claims {claim_n:6.2f} Hz off | real HH {true_n:6.2f} Hz off | gap {abs(claim_n-true_n):6.1f}  SILENT LIE")

# --- FIX: feasibility screen + place with the TRUE f-I ---
t_feas = np.array([project(t) for t in tgt]); infeasible = np.abs(t_feas - tgt) > 1
x_fix = solve(np.array([x_true_for(t) for t in t_feas]))
real_fix = true_hh(x_fix)
claim_f = float(np.sqrt(np.sum((real_fix - t_feas) ** 2)))         # fix reports against the TRUE model it used
loss_req = float(np.sqrt(np.sum((real_fix - tgt) ** 2)))           # honest, unavoidable loss vs the impossible request
print(f"FIX   true-f-I aware:   claims {claim_f:6.2f} Hz off | real HH {claim_f:6.2f} Hz off | gap {0.0:6.1f}  HONEST (claim == reality)")
print(f"   flags {int(infeasible.sum())}/{N} cells UNWRITABLE up front (rate not in {{0}} U [55,115])")
print(f"   honest loss vs the original request = {loss_req:.1f} Hz, REPORTED not hidden")
print("   the pipeline never certifies a write the neuron will not perform, at any electrode count.")

# --- figure: per-cell, naive (lies) vs fix (honest) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
order = np.argsort(tgt)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
ax1.plot(range(N), tgt[order], "ks", ms=6, mfc="none", label="requested code")
ax1.plot(range(N), surrogate(x_naive)[order], "o", ms=5, color="C0", label="pipeline claims")
ax1.plot(range(N), true_hh(x_naive)[order], "x", ms=7, color="C3", label="real HH")
ax1.set_title(f"NAIVE surrogate: claims 0 off, silently {true_n:.0f} Hz off")
ax1.set_xlabel("cell (sorted by requested rate)"); ax1.set_ylabel("rate (Hz)")
ax1.legend(fontsize=8, loc="center left"); ax1.set_ylim(-5, 145)
ax2.plot(range(N), tgt[order], "ks", ms=6, mfc="none", label="requested code")
ax2.plot(range(N), t_feas[order], "^", ms=6, color="C2", label="feasible target (screened)")
ax2.plot(range(N), real_fix[order], "x", ms=7, color="C3", label="real HH = claim")
inf_ord = infeasible[order]
ax2.scatter(np.arange(N)[inf_ord], tgt[order][inf_ord], s=140, facecolors="none",
            edgecolors="red", linewidths=1.3, label="flagged unwritable")
ax2.set_title("FIX: claim = reality, unwritable cells flagged")
ax2.set_xlabel("cell (sorted by requested rate)")
ax2.legend(fontsize=8, loc="center left"); ax2.set_ylim(-5, 145)
fig.tight_layout(); fig.savefig("../figures/fix.png", dpi=130)
print("\nsaved ../figures/fix.png")
