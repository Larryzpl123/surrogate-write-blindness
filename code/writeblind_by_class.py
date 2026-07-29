"""Tie the excitability class to the surrogate-write blindness. For each measured cell, fit a
smooth monotone surrogate to its firing branch (as the write pipeline would), then REQUEST a set
of target rates: the surrogate inverts each to a current, and the REAL cell (its measured f-I)
delivers the actual rate. On Type-II cells a low request lands in the unwritable gap and the real
cell delivers 0 (silent write); on Type-I cells the same request is faithful. A feasibility screen
against the true f-I flags exactly the Type-II low requests. Reads results/fi_*.npy. Pure scipy.
Prints a table and writes figures/fig6_writeblind_by_class.png/pdf.
Run from code/:   python3 writeblind_by_class.py
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "legend.fontsize": 8,
})
BLUE, VERM, GREY, BLACK = "#0072B2", "#D55E00", "0.55", "#000000"
sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

MODELS = {"HH": "Type-II", "CSII": "Type-II", "CSI": "Type-I", "WB": "Type-I"}


def load(k):
    a = np.load(f"../results/fi_{k}.npy"); return a[:, 0], a[:, 1]


def fit_surrogate(I, r):
    """BEST-case smooth surrogate: monotone PCHIP interpolation of the firing branch, extrapolated.
    A smooth surrogate has no gap by construction, so it happily maps any requested rate to a
    current. Using the most accurate surrogate makes the point stronger: even it is blind to the gap."""
    fir = r > 0
    If, rf = I[fir], r[fir]
    idx = np.argsort(If)
    # anchor at the origin so the smooth surrogate rises 0 -> floor across the sub-rheobase region
    # (this IS the gap-filling blindness), and DO NOT extrapolate (avoids pathological tails).
    Ik = np.r_[0.0, If[idx]]; rk = np.r_[0.0, rf[idx]]
    pc = PchipInterpolator(Ik, rk, extrapolate=False)
    return lambda x: np.nan_to_num(np.clip(pc(np.asarray(x, float)), 0.0, None))


def true_fi(I, r):
    """measured writable set: interpolate the firing branch; 0 below rheobase / above block."""
    fir = r > 0
    If, rf = I[fir], r[fir]
    return If, rf, (lambda x: np.interp(x, If, rf, left=0.0, right=0.0))


def deliver(smon, Igrid, tfi, rho):
    """surrogate (monotone on Igrid) inverts rho -> current; real cell delivers true rate there."""
    Iwant = float(np.interp(rho, smon, Igrid))     # clean monotone inverse (no argmin jitter)
    return float(tfi(Iwant)), Iwant


# ---- table: request 30 / 40 / 100 Hz on every cell ----
print(f"{'cell':5} {'class':8} {'request':>8} {'surr.claims':>12} {'real delivers':>14} {'silent err':>11}")
data = {}
for k, cls in MODELS.items():
    I, r = load(k); surr = fit_surrogate(I, r); If, rf, tfi = true_fi(I, r)
    floor = rf.min()
    Ig = np.linspace(0, If.max(), 8000)
    smon = np.maximum.accumulate(surr(Ig))          # monotone surrogate for a clean inverse
    data[k] = (cls, I, r, smon, If, rf, tfi, floor, Ig)
    for rho in (30.0, 40.0, 100.0):
        dl, _ = deliver(smon, Ig, tfi, rho)
        print(f"{k:5} {cls:8} {rho:7.0f}  {rho:11.0f}  {dl:13.1f}  {rho - dl:10.1f}")

# ---- figure: requested vs delivered, one Type-II vs one Type-I ----
PAN = [("A", "CSII", "Connor-Stevens (Type-II)"), ("B", "CSI", "Connor-Stevens (Type-I)")]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
for (letter, k, title), ax in zip(PAN, axes):
    cls, I, r, smon, If, rf, tfi, floor, Ig = data[k]
    req = np.linspace(0, min(rf.max(), 130), 300)
    dlv = np.array([deliver(smon, Ig, tfi, rho)[0] for rho in req])
    if cls == "Type-II":
        ax.axvspan(0, floor, color=VERM, alpha=0.13, zorder=0)
        ax.text(floor * 0.5, 120, "requests in\nthe gap", ha="center", va="top", fontsize=7.5, color=VERM)
    ax.plot(req, req, "--", color=BLUE, lw=1.4, label="surrogate claims (= request)")
    ax.plot(req, dlv, "-", color=VERM, lw=1.9, label="real cell delivers")
    ax.set_title(f"{letter}   {title}", loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel("requested rate (Hz)"); ax.set_ylabel("delivered rate (Hz)")
    ax.set_xlim(0, req.max()); ax.set_ylim(-6, req.max() * 1.02)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
              handletextpad=0.5, borderaxespad=0.0)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"../figures/fig6_writeblind_by_class.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig6_writeblind_by_class")
