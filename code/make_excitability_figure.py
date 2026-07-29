"""Excitability-class figure: the low-rate GAP the write-surrogate is blind to is a signature
of Type-II (Hopf) onset, absent in Type-I (SNIC). Four single-compartment models, measured f-I
read LIVE from results/fi_*.npy (gen_canonical_fi.py); nothing transcribed. Same house style as
make_figures.py (Okabe-Ito color, serif, no grid, boxed legend below, A/B above-left, 300 dpi).
Run from code/:   python3 make_excitability_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.linewidth": 0.8, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.edgecolor": "black",
    "legend.fancybox": False, "legend.framealpha": 1.0, "legend.fontsize": 8,
})
BLUE, VERM, GREY = "#0072B2", "#D55E00", "0.55"

# measured f-I: columns [I (uA/cm^2), rate (Hz), Vlate (mV)]
D = {k: np.load(f"../results/fi_{k}.npy") for k in ("HH", "CSII", "CSI", "WB")}

def floor_rheo(arr, xmax):
    m = arr[:, 0] <= xmax
    fired = arr[m & (arr[:, 1] > 0)]
    return float(fired[:, 1].min()), float(fired[0, 0])   # floor rate, rheobase current

# (letter, title, class, key, xmax for the onset-focused window)
PAN = [("A", "Classic Hodgkin-Huxley", "Type-II", "HH",   16),
       ("B", "Connor-Stevens",         "Type-II", "CSII", 42),
       ("C", "Connor-Stevens",         "Type-I",  "CSI",  42),
       ("D", "Wang-Buzsaki interneuron","Type-I", "WB",    5)]

fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
for (letter, title, cls, key, xmax), ax in zip(PAN, axes.flat):
    arr = D[key]; I, r = arr[:, 0], arr[:, 1]
    m = I <= xmax
    floor, rheo = floor_rheo(arr, xmax)
    top = float(max(r[m])) * 1.12
    color = VERM if cls == "Type-II" else BLUE
    # silent branch (rate 0 below rheobase) as a flat grey line
    sil = m & (r == 0)
    ax.plot(I[sil], r[sil], "-", color=GREY, lw=1.6, zorder=2)
    # firing branch
    fir = m & (r > 0)
    ax.plot(I[fir], r[fir], "-o", color=color, ms=3.6, lw=1.5, zorder=4)
    if cls == "Type-II":
        # the unwritable low-rate gap: no current yields a rate in (0, floor)
        ax.axhspan(0, floor, color=VERM, alpha=0.13, zorder=0)
        ax.plot([rheo, rheo], [0, floor], ":", color=color, lw=1.3, zorder=3)  # the jump
        ax.axhline(floor, color=color, lw=0.7, ls="--", zorder=1)
        note = f"unwritable gap\n(0, {floor:.0f} Hz)"
    else:
        note = f"no gap:\nfires to ~0\n(floor {floor:.0f} Hz)"
    # all four annotations in the SAME fixed spot: upper-left (the only empty corner in all panels)
    ax.text(xmax * 0.05, top * 0.93, note, ha="left", va="top", fontsize=7.5, color=color)
    ax.set_title(f"{letter}   {title} ({cls})", loc="left", fontweight="bold", fontsize=9)
    ax.set_xlabel(r"input current  I ($\mu$A/cm$^2$)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-top * 0.05, top)

# one shared, boxed legend below the whole figure
handles = [plt.Line2D([], [], color=VERM, marker="o", ms=4, lw=1.5, label="Type-II f-I (has low-rate gap)"),
           plt.Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.5, label="Type-I f-I (no gap)"),
           plt.Line2D([], [], color=GREY, lw=1.6, label="silent (below rheobase)"),
           mpatches.Patch(facecolor=VERM, alpha=0.13, edgecolor="none", label="unwritable rate gap")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           ncol=4, columnspacing=1.3, handletextpad=0.5, frameon=True, edgecolor="black",
           fancybox=False, framealpha=1.0, fontsize=8)
fig.tight_layout(rect=[0, 0.05, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"../figures/fig5_excitability_class.{ext}", dpi=300, bbox_inches="tight")
print("wrote fig5_excitability_class  | floors:",
      {k: round(floor_rheo(D[k], xm)[0], 1) for (_, _, _, k, xm) in PAN})
