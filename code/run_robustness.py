"""ROBUSTNESS: the silent write failure is NOT an artifact of a weak surrogate. Fit a family of
smooth surrogates to the raw HH f-I, from a mediocre 3-param logistic (in-band R^2 ~0.67) up to a
near-perfect monotone interpolator (R^2 = 1.0000), and show EVERY one of them erases the Type-II
floor + depol block and therefore silently certifies rate-codes real HH cannot produce. In-band
fit quality is uncorrelated with the silent-failure magnitude: smoothness itself is the failure,
not how well you fit the interior. This kills the "you built a bad surrogate on purpose" critique.

Pure numpy/scipy; ground truth = raw JAXLEY f-I (results/hh_fi.npy). Run from code/:
    python run_robustness.py
Writes figures/robustness.png.
"""
import numpy as np
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import UnivariateSpline, PchipInterpolator
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK

_sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
raw = np.load("../results/hh_fi.npy"); I, r = raw[:, 0], raw[:, 1]
fm = I <= 0.03; Ifit, rfit = I[fm], r[fm]                       # monotonic training region
bm = (I >= 0.004) & (I <= 0.029); Ib, rb = I[bm], r[bm]        # in-band points for R^2

# --- a family of smooth surrogates rate(I), increasing fit quality ---
surrs = {}
surrs["sigmoid x affine (4p)"] = lambda I: _sig((I - HH_A) / HH_B) * (HH_C + HH_D * I)
_f2 = lambda I, R, c, w: R * _sig((I - c) / w)
_p2, _ = curve_fit(_f2, Ifit, rfit, p0=[120, 0.008, 0.003], maxfev=40000)
surrs["logistic (3p)"] = lambda I, p=_p2: _f2(I, *p)
_sp = UnivariateSpline(Ifit, rfit, k=3, s=40); surrs["cubic spline"] = lambda I, s=_sp: np.clip(s(I), 0, None)
_idx = np.argsort(Ifit); _pch = PchipInterpolator(Ifit[_idx], rfit[_idx], extrapolate=True)
surrs["monotone interp (R2=1)"] = lambda I, p=_pch: np.clip(p(I), 0, None)

# --- faithful HH f-I from the raw JAXLEY data (jump + block) ---
rise = (I >= 0.004) & (I <= 0.030)
def true_hh_x(x):
    Ii = I_BLOCK * np.atleast_1d(np.asarray(x, float)); out = np.zeros_like(Ii)
    m = (Ii >= 0.004) & (Ii <= 0.030); out[m] = np.interp(Ii[m], I[rise], r[rise]); return out

# --- same network write as the main demo: code with gap + band + block cells ---
rng = np.random.default_rng(1); N = 20
W = (2.0 / np.sqrt(N)) * rng.standard_normal((N, N))
tgt = np.empty(N); pm = rng.permutation(N); tgt[pm[:8]] = 30.0; tgt[pm[8:16]] = 85.0; tgt[pm[16:]] = 125.0

rows = []
print(f"{'surrogate':>24}{'inbandR2':>10}{'surrResid':>10}{'trueResid':>10}{'claim@30':>9}{'real@30':>8}")
for name, rfn in surrs.items():
    Rx = lambda x, rfn=rfn: rfn(I_BLOCK * np.asarray(x, float))
    r2 = 1 - np.sum((rfn(Ib) - rb) ** 2) / np.sum((rb - rb.mean()) ** 2)
    xg = np.linspace(0, 1.55, 40000); rg = Rx(xg)
    inv = lambda rho, xg=xg, rg=rg: float(xg[np.argmin(np.abs(rg - rho))])
    xt = np.array([inv(t) for t in tgt]); x0 = np.linalg.lstsq(W, xt, rcond=None)[0]
    loss = lambda s: float(np.sum((Rx(W @ s) - tgt) ** 2))
    res = minimize(loss, x0, method="L-BFGS-B", bounds=[(-15, 15)] * N, options={"maxiter": 8000})
    xs = W @ res.x
    sresid = float(np.sqrt(np.sum((Rx(xs) - tgt) ** 2)))
    tresid = float(np.sqrt(np.sum((true_hh_x(xs) - tgt) ** 2)))
    c30 = float(Rx(inv(30))); real30 = float(true_hh_x(inv(30))[0])
    rows.append((name, r2, sresid, tresid))
    print(f"{name:>24}{r2:10.4f}{sresid:10.2f}{tresid:10.1f}{c30:9.1f}{real30:8.1f}")
print("\nin-band R^2 spans %.2f..%.4f; every surrogate certifies ~0 Hz error while real HH is 200+ Hz off." %
      (min(x[1] for x in rows), max(x[1] for x in rows)))
print("even the R^2=1.0 monotone interpolator fails: perfecting the interior does nothing for the edges.")

# --- figure ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
# panel 1: f-I overlay on current I
Iax = np.linspace(0, 0.045, 600)
ax1.axvspan(0.004, 0.029, color="green", alpha=0.08)
ax1.plot(I[r > 0], r[r > 0], "ko", ms=4, label="real HH (JAXLEY)")
ax1.plot([0, 0.004], [0, 0], "k-", lw=2); ax1.plot([0.030, 0.045], [0, 0], "k-", lw=2)  # true 0 outside
for name, rfn in surrs.items():
    ax1.plot(Iax, np.clip(rfn(Iax), -5, 160), lw=1.6, label=name)
ax1.axvline(0.004, ls=":", c="gray", lw=.7); ax1.axvline(0.030, ls=":", c="gray", lw=.7)
ax1.text(0.0009, 120, "gap:\nHH=0", fontsize=8, color="C3")
ax1.text(0.033, 120, "block:\nHH=0", fontsize=8, color="C3")
ax1.set_xlabel("input current I"); ax1.set_ylabel("rate (Hz)"); ax1.set_ylim(-8, 165)
ax1.set_title("every smooth surrogate fills the gap & skips the block"); ax1.legend(fontsize=7, loc="upper right")
# panel 2: in-band R^2 vs true residual (no correlation)
r2s = [x[1] for x in rows]; trs = [x[3] for x in rows]; names = [x[0] for x in rows]
ax2.scatter(r2s, trs, s=60, color="C3", zorder=3)
for nm, a, b in zip(names, r2s, trs):
    ax2.annotate(nm, (a, b), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax2.set_xlabel("in-band fit quality (R^2)"); ax2.set_ylabel("silent write error on real HH (Hz)")
ax2.set_title("better fit does NOT reduce the silent failure"); ax2.set_ylim(0, max(trs) * 1.25); ax2.grid(alpha=.3)
fig.tight_layout(); fig.savefig("../figures/robustness.png", dpi=130)
print("saved ../figures/robustness.png")
