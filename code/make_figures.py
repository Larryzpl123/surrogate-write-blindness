"""make_figures.py -- all paper figures, ONE consistent style. Color (colorblind-safe Okabe-Ito),
serif, no grid, top/right spines off. EVERY legend is boxed and sits OUTSIDE the axes in a fixed
spot (below the panel), never shuffled to chase empty space. Panel labels A/B above-left. 300-dpi
PNG + vector PDF, bbox_inches="tight".

Fixed color roles:  real HH / ground truth = black or vermillion(outcome),  surrogate = blue,
feasible = green,  the four surrogate variants = blue/orange/green/purple.  Run from code/:
    python make_figures.py
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matched_params import HH_A, HH_B, HH_C, HH_D, I_BLOCK

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.framealpha": 1.0, "legend.fontsize": 8,
})
BLACK, BLUE, ORANGE, GREEN, VERM, PURPLE = "#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
BAND = "#EDEDED"


def panel(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=11)


def leg_below(ax, ncol):                          # fixed, outside, consistent: centered below the panel
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=ncol,
              columnspacing=1.1, handletextpad=0.5, borderaxespad=0.0)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"../figures/{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", stem)


# ---------------- shared model ----------------
sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
surrI = lambda I: sig((I - HH_A) / HH_B) * (HH_C + HH_D * I)
surrx = lambda x: surrI(I_BLOCK * x)
raw = np.load("../results/hh_fi.npy"); Iraw, rraw = raw[:, 0], raw[:, 1]
rise = (Iraw >= 0.004) & (Iraw <= 0.030)


def true_x(x):
    Ii = I_BLOCK * np.atleast_1d(np.asarray(x, float)); out = np.zeros_like(Ii)
    m = (Ii >= 0.004) & (Ii <= 0.030); out[m] = np.interp(Ii[m], Iraw[rise], rraw[rise]); return out


_xg = np.linspace(0, 1.55, 40000); _rg = surrx(_xg)
sinv = lambda rho: float(_xg[np.argmin(np.abs(_rg - rho))])
rng0 = np.random.default_rng(1); N = 20
W20 = (2.0 / np.sqrt(N)) * rng0.standard_normal((N, N))
CODE = np.empty(N); _pm = rng0.permutation(N)
CODE[_pm[:8]] = 30.0; CODE[_pm[8:16]] = 85.0; CODE[_pm[16:]] = 125.0
solve = lambda W, xt: W @ np.linalg.lstsq(W, xt, rcond=None)[0]
FIG = (7.4, 3.6)


# ---------------- Fig 1: silent blindness ----------------
def fig_blindness():
    fig, (a, b) = plt.subplots(1, 2, figsize=FIG)
    rr = np.linspace(0, 145, 400); act = np.array([float(true_x(sinv(r))[0]) for r in rr])
    a.axvspan(55, 115, color=BAND, zorder=0)
    a.plot(rr, rr, "--", color=BLUE, lw=1.4, label="surrogate certifies")
    a.plot(rr, act, "-", color=VERM, lw=1.8, label="real HH delivers")
    a.set_xlabel("requested rate (Hz)"); a.set_ylabel("resulting rate (Hz)")
    a.set_xlim(0, 145); a.set_ylim(-6, 150); panel(a, "A"); leg_below(a, 1)

    xs = solve(W20, np.array([sinv(t) for t in CODE])); o = np.argsort(CODE)
    b.plot(range(N), CODE[o], "s", mfc="none", mec=BLACK, mew=1.1, ms=7, label="target")
    b.plot(range(N), surrx(xs)[o], "o", color=BLUE, ms=5, label="surrogate claims")
    b.plot(range(N), true_x(xs)[o], "x", color=VERM, ms=6, mew=1.6, label="real HH")
    b.set_xlabel("cell (sorted by target rate)"); b.set_ylabel("rate (Hz)"); b.set_ylim(-6, 145)
    panel(b, "B"); leg_below(b, 3)
    save(fig, "fig1_blindness")


# ---------------- Fig 2: robustness ----------------
def fig_robustness():
    fm = Iraw <= 0.03; Ifit, rfit = Iraw[fm], rraw[fm]
    bm = (Iraw >= 0.004) & (Iraw <= 0.029); Ib, rb = Iraw[bm], rraw[bm]
    _f2 = lambda I, R, c, w: R * sig((I - c) / w)
    _p2, _ = curve_fit(_f2, Ifit, rfit, p0=[120, 0.008, 0.003], maxfev=40000)
    _sp = UnivariateSpline(Ifit, rfit, k=3, s=40)
    _i = np.argsort(Ifit); _pc = PchipInterpolator(Ifit[_i], rfit[_i], extrapolate=True)
    surrs = [("sigmoid x affine", surrI, BLUE, "o"),
             ("logistic", lambda I: _f2(I, *_p2), ORANGE, "s"),
             ("cubic spline", lambda I: np.clip(_sp(I), 0, None), GREEN, "^"),
             ("monotone interp", lambda I: np.clip(_pc(I), 0, None), PURPLE, "D")]
    fig, (a, b) = plt.subplots(1, 2, figsize=FIG)
    a.axvspan(0.004, 0.029, color=BAND, zorder=0)
    Iax = np.linspace(0, 0.045, 500)
    for nm, f, c, mk in surrs:
        a.plot(Iax, np.clip(f(Iax), -5, 165), color=c, lw=1.5, label=nm, zorder=2)
    a.plot([0, 0.004], [0, 0], "-", color=BLACK, lw=1.6, zorder=3)
    a.plot([0.030, 0.045], [0, 0], "-", color=BLACK, lw=1.6, zorder=3)
    a.plot(Iraw[rraw > 0], rraw[rraw > 0], "o", mfc="white", mec=BLACK, mew=0.9, ms=4, label="real HH", zorder=5)
    a.set_xlabel("input current  I (nA)"); a.set_ylabel("rate (Hz)"); a.set_ylim(-8, 168)
    panel(a, "A"); leg_below(a, 3)

    for nm, f, c, mk in surrs:
        r2 = 1 - np.sum((f(Ib) - rb) ** 2) / np.sum((rb - rb.mean()) ** 2)
        rgi = f(I_BLOCK * _xg); inv = lambda rho: float(_xg[np.argmin(np.abs(rgi - rho))])
        xs = solve(W20, np.array([inv(t) for t in CODE]))
        tr = float(np.sqrt(np.sum((true_x(xs) - CODE) ** 2)))
        b.scatter([r2], [tr], s=70, color=c, edgecolor="black", linewidth=0.5, marker=mk, zorder=3, label=nm)
    b.set_xlabel("in-band fit quality  ($R^2$)"); b.set_ylabel("silent error on real HH (Hz)")
    b.set_xlim(0.6, 1.05); b.set_ylim(0, 320); panel(b, "B"); leg_below(b, 2)
    save(fig, "fig2_robustness")


# ---------------- Fig 3: the fix ----------------
def fig_fix():
    FLOOR, CEIL = 55.0, 115.0
    project = lambda rho: 0.0 if rho < FLOOR else min(rho, CEIL)
    xtrue = lambda rho: 0.05 if rho <= 0.1 else float(np.interp(rho, rraw[rise], Iraw[rise])) / I_BLOCK
    o = np.argsort(CODE)
    x_naive = solve(W20, np.array([sinv(t) for t in CODE]))
    tfeas = np.array([project(t) for t in CODE]); infeas = np.abs(tfeas - CODE) > 1
    x_fix = solve(W20, np.array([xtrue(t) for t in tfeas]))

    fig, (a, b) = plt.subplots(1, 2, figsize=FIG, sharey=True)
    a.plot(range(N), CODE[o], "s", mfc="none", mec=BLACK, mew=1.1, ms=7, label="requested")
    a.plot(range(N), surrx(x_naive)[o], "o", color=BLUE, ms=5, label="pipeline claims")
    a.plot(range(N), true_x(x_naive)[o], "x", color=VERM, ms=6, mew=1.6, label="real HH")
    a.set_xlabel("cell (sorted by requested rate)"); a.set_ylabel("rate (Hz)"); a.set_ylim(-6, 145)
    panel(a, "A"); leg_below(a, 3)

    reqo = CODE[o]; io = infeas[o]; RED = "#D7191C"
    b.plot(np.arange(N)[~io], reqo[~io], "s", mfc="none", mec=BLACK, mew=1.1, ms=7, label="requested (writable)")
    b.plot(np.arange(N)[io], reqo[io], "s", mfc=RED, mec=RED, ms=7, label="requested (unwritable)")
    b.plot(range(N), tfeas[o], "^", color=GREEN, ms=6, label="feasible target")
    b.plot(range(N), true_x(x_fix)[o], "x", color=VERM, ms=6, mew=1.6, label="real HH = claim")
    b.set_xlabel("cell (sorted by requested rate)"); panel(b, "B"); leg_below(b, 2)
    save(fig, "fig3_fix")


# ---------------- Fig 4: prevalence ----------------
def fig_quantify():
    rng = np.random.default_rng(3); W = (2.0 / np.sqrt(N)) * rng.standard_normal((N, N))
    A_vals = np.arange(0, 141, 5); Aerr = []
    for A in A_vals:
        ts = []
        for _ in range(24):
            t = np.zeros(N); t[rng.permutation(N)[:10]] = A
            xs = solve(W, np.array([sinv(v) for v in t])); ts.append(np.sqrt(np.sum((true_x(xs) - t) ** 2)))
        Aerr.append(np.mean(ts))
    gaps = []
    for _ in range(400):
        t = rng.uniform(0, 120, N); xs = solve(W, np.array([sinv(v) for v in t]))
        gaps.append(np.sqrt(np.sum((true_x(xs) - t) ** 2)))
    gaps = np.array(gaps)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    a.axvspan(55, 115, color=BAND, zorder=0)
    a.plot(A_vals, Aerr, "-o", color=VERM, lw=1.5, ms=4)
    a.text(85, max(Aerr) * 0.95, "writable\nband", ha="center", va="top", fontsize=8)
    a.set_xlabel("active level of the code (Hz)"); a.set_ylabel("silent error on real HH (Hz)")
    a.set_xlim(-3, 143); a.set_ylim(0, max(Aerr) * 1.06); panel(a, "A")

    b.hist(gaps, bins=24, color=BLUE, alpha=0.85, edgecolor="black", linewidth=0.4)
    b.axvline(gaps.mean(), color=BLACK, ls="--", lw=1.2)
    b.set_xlabel("silent error per code on real HH (Hz)"); b.set_ylabel("number of codes")
    b.annotate(f"mean {gaps.mean():.0f} Hz\n(surrogate: 0)", (gaps.mean(), b.get_ylim()[1] * 0.9),
               xytext=(7, 0), textcoords="offset points", fontsize=8, va="top")
    panel(b, "B")
    save(fig, "fig4_quantify")


if __name__ == "__main__":
    fig_blindness(); fig_robustness(); fig_fix(); fig_quantify()
    print("done")
