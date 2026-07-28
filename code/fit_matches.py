"""Re-derive every frozen constant in matched_params.py from results/hh_fi.npy and ASSERT it
matches, then redraw figures/match_overlay.png. Pure numpy/scipy (no jax): matched_params is the
single source of truth, reproduced here so no hand-copied number can silently drift (CA1 rule).

Run from code/ :   python fit_matches.py
Exits non-zero if any frozen constant has drifted from what the raw f-I implies.
"""
import numpy as np
from scipy.optimize import curve_fit
import matched_params as MP

sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
sp  = lambda x: np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)   # stable softplus

# --- 1. HH surrogate from the raw JAXLEY f-I -------------------------------------------------
a = np.load("../results/hh_fi.npy"); I, r = a[:, 0], a[:, 1]
band = I <= 0.03                                   # rising region (below-rheobase 0s + firing)
hh_form = lambda I, A, B, C, D: sig((I - A) / B) * (C + D * I)
(pA, pB, pC, pD), _ = curve_fit(hh_form, I[band], r[band], p0=[0.003, 0.0008, 50, 2000], maxfev=40000)
hh_r2 = 1 - np.sum((hh_form(I[band], pA, pB, pC, pD) - r[band]) ** 2) / np.sum((r[band] - r[band].mean()) ** 2)
print(f"HH surrogate:  A={pA:.5f} B={pB:.6f} C={pC:.2f} D={pD:.1f}   R^2={hh_r2:.4f}")

# --- 2. toy 2-knob matches to HH over the band [55,115] Hz (x in [RHEOBASE_X, 1]) ------------
hh_x = lambda x: hh_form(MP.I_BLOCK * x, pA, pB, pC, pD)
xa = np.linspace(MP.RHEOBASE_X, 1.0, 60); ya = hh_x(xa)
enc = lambda x, g, b: sp(g * x + b)
(eg, eb), _ = curve_fit(enc, xa, ya, p0=[70, -10], maxfev=80000)
def lif_u(u):
    g = np.maximum(MP.LIF_SMOOTH * sp((u - MP.LIF_THETA) / MP.LIF_SMOOTH), 1e-9)
    return 1.0 / (MP.LIF_TREF + MP.LIF_TAU * np.log1p(MP.LIF_THETA / g))
lif = lambda x, g, b: lif_u(g * x + b)
(lg, lb), _ = curve_fit(lif, xa, ya, p0=[12, 1.0], maxfev=80000)
er2 = 1 - np.sum((enc(xa, eg, eb) - ya) ** 2) / np.sum((ya - ya.mean()) ** 2)
lr2 = 1 - np.sum((lif(xa, lg, lb) - ya) ** 2) / np.sum((ya - ya.mean()) ** 2)
print(f"encoder match: gain={eg:.4f} base={eb:.4f}   R^2={er2:.4f}")
print(f"LIF match:     gain={lg:.4f} base={lb:.4f}   R^2={lr2:.4f}")
print(f"baseline rate x=0:  HH={hh_x(0.0):.2f}  enc={enc(0.0, eg, eb):.2f}  LIF={lif(0.0, lg, lb):.2f} Hz")

# --- 3. ASSERT the frozen constants reproduce (guards against silent drift) ------------------
def chk(name, got, want, rtol=2e-2):
    ok = bool(np.isclose(got, want, rtol=rtol))
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: fit={got:.5g}  frozen={want:.5g}")
    assert ok, f"{name} drifted from matched_params: fit {got} vs frozen {want}"
print("assert vs matched_params.py:")
for nm, g, w in [("HH_A", pA, MP.HH_A), ("HH_B", pB, MP.HH_B), ("HH_C", pC, MP.HH_C), ("HH_D", pD, MP.HH_D),
                 ("ENC_GAIN", eg, MP.ENC_GAIN), ("ENC_BASE", eb, MP.ENC_BASE),
                 ("LIF_GAIN", lg, MP.LIF_GAIN), ("LIF_BASE", lb, MP.LIF_BASE)]:
    chk(nm, g, w)
assert hh_r2 > 0.99 and er2 > 0.98 and lr2 > 0.98, "fit quality regressed"
print("ALL OK: matched_params.py reproduced from results/hh_fi.npy")

# --- 4. redraw the overlay figure -----------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xe = np.linspace(-0.15, 1.2, 300); hhe = np.where(xe <= 1, hh_x(np.clip(xe, -1, 1)), np.nan)
plt.figure(figsize=(6.2, 4.2)); plt.axvspan(MP.RHEOBASE_X, 1, alpha=0.08, color="gray")
plt.plot(xe, enc(xe, eg, eb), label="encoder (toy)"); plt.plot(xe, lif(xe, lg, lb), label="LIF")
plt.plot(xe, hhe, lw=2.4, label="HH (surrogate)")
plt.axhline(55, ls=":", c="k", lw=.6); plt.axhline(115, ls=":", c="k", lw=.6)
plt.axvline(0, ls="--", c="r", lw=.8); plt.text(0.01, 150, "baseline\n(stim=0)", fontsize=7, color="r")
plt.text(1.02, 10, "HH block", fontsize=7)
plt.xlabel("common drive x  (0=silent baseline, 1=block)"); plt.ylabel("rate (Hz)")
plt.title("2-knob match to HH: in-band aligned, baseline+edges diverge")
plt.legend(fontsize=8, loc="center right"); plt.ylim(-6, 205); plt.tight_layout()
plt.savefig("../figures/match_overlay.png", dpi=130); print("saved ../figures/match_overlay.png")
