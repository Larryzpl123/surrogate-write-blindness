"""Supplementary figure: the low-rate gap is an onset-bifurcation signature in the canonical minimal
model. Morris-Lecar in its Hopf (Type-II) vs SNIC (Type-I) regime, a single-parameter switch in a
2-variable model. Reads results/fi_ML{II,I}.npy (measure_ml_fi.py). Same house style as make_excitability_figure.py.
Run from code/:  python3 make_ml_supp_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "legend.framealpha": 1.0, "legend.fontsize": 8,
})
BLUE, VERM, GREY = "#0072B2", "#D55E00", "0.55"

D = {k: np.load(f"../results/fi_ML{k}.npy") for k in ("II", "I")}

def floor_of(arr):
    fired = arr[arr[:, 1] > 0]
    return float(fired[:, 1].min())

PAN = [("A", "Morris-Lecar, Hopf regime", "Type-II", "II", VERM),
       ("B", "Morris-Lecar, SNIC regime", "Type-I", "I", BLUE)]

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
for (letter, title, cls, key, color), ax in zip(PAN, axes):
    arr = D[key]; I, r = arr[:, 0], arr[:, 1]
    floor = floor_of(arr)
    top = float(max(r)) * 1.15
    sil = r == 0
    ax.plot(I[sil], r[sil], "-", color=GREY, lw=1.6, zorder=2)
    fir = r > 0
    ax.plot(I[fir], r[fir], "-o", color=color, ms=3.6, lw=1.5, zorder=4)
    if cls == "Type-II":
        rheo = float(I[fir][0])
        ax.axhspan(0, floor, color=VERM, alpha=0.13, zorder=0)
        ax.plot([rheo, rheo], [0, floor], ":", color=color, lw=1.3, zorder=3)
        ax.axhline(floor, color=color, lw=0.7, ls="--", zorder=1)
        note = f"unwritable gap\n(0, {floor:.0f} Hz)"
    else:
        note = f"no gap:\nfires down to {floor:.1f} Hz"
    ax.text(0.05 * max(I), top * 0.93, note, ha="left", va="top", fontsize=7.5, color=color)
    ax.set_title(f"{letter}   {title} ({cls})", loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel(r"applied current  $I$ ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_xlim(0, max(I)); ax.set_ylim(-top * 0.05, top)

handles = [plt.Line2D([], [], color=VERM, marker="o", ms=4, lw=1.5, label="Hopf onset (Type-II): low-rate gap"),
           plt.Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.5, label="SNIC onset (Type-I): continuous, no gap")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2,
           columnspacing=1.4, handletextpad=0.5, frameon=True, edgecolor="black",
           fancybox=False, framealpha=1.0, fontsize=8)
fig.tight_layout(rect=[0, 0.07, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"../figures/figS1_morris_lecar.{ext}", dpi=300, bbox_inches="tight")
print("wrote figS1_morris_lecar | floors:", {k: round(floor_of(D[k]), 1) for k in ("II", "I")})
