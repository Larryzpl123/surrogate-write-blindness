"""Conductance robustness: does the excitability-class STRUCTURE survive changing g_Na / g_K?
For CS Type-II and Type-I, scale g_Na and g_K by +/-20% (one at a time) and report rheobase,
floor (min sustained rate at onset) and depol-block onset. The claim: the NUMBERS move but the
STRUCTURE is invariant -- Type-II keeps a finite low-rate gap, Type-I stays continuous (floor ~0).
So the gap is not a parameter special case. Pure numpy/scipy. Appends results/robustness_gNaK.csv.

Run one class per call:   python3 robustness_sweep.py CSII   |   CSI
"""
import sys, os, csv, numpy as np
from scipy.integrate import odeint
from measure_cs_fi import CS

COMBOS = [(1.0, 1.0), (0.8, 1.0), (1.2, 1.0), (1.0, 0.8), (1.0, 1.2)]


def rate(model, I, rest, T=600.0, dt=0.05, last=400.0):
    t = np.arange(0, T, dt)
    V = odeint(model.ode, rest, t, args=(float(I),))[:, 0]
    seg = V[int((T - last) / dt):]
    ab = seg > 0.0
    return int(np.sum((~ab[:-1]) & (ab[1:]))) / (last / 1000.0)


def rheo_floor(model, rest):
    a, b = 0.0, None
    for I in [4, 6, 7, 8, 9, 10, 12, 16, 22, 30, 40]:      # coarse bracket
        if rate(model, I, rest) > 0:
            b = I; break
        a = I
    if b is None:
        return None, None
    while b - a > 0.1:                                     # bisect to 0.1 uA/cm2
        m = (a + b) / 2
        if rate(model, m, rest) > 0:
            b = m
        else:
            a = m
    return b, rate(model, b, rest)                         # rheobase, floor (rate at onset)


def block_onset(model, rest):
    for I in [80, 130, 200, 300]:
        if rate(model, I, rest) == 0:
            return I
    return None


if __name__ == "__main__":
    typ = "II" if sys.argv[1] == "CSII" else "I"
    out = "../results/robustness_gNaK.csv"
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["class", "sNa", "sK", "gNa", "gK", "rheobase", "floor_Hz", "block_uA", "structure"])
        for sNa, sK in COMBOS:
            m = CS(typ); m.g_na *= sNa; m.g_k *= sK
            rest = m.rest()
            rheo, floor = rheo_floor(m, rest)
            blk = block_onset(m, rest)
            struct = "GAP (Type-II)" if (floor and floor > 25) else "continuous (Type-I)"
            row = [f"CS-{typ}", sNa, sK, round(m.g_na, 1), round(m.g_k, 1),
                   round(rheo, 2) if rheo else None, round(floor, 1) if floor else None, blk, struct]
            w.writerow(row)
            print(f"CS-{typ}  gNa={m.g_na:5.1f}(x{sNa}) gK={m.g_k:4.1f}(x{sK})  "
                  f"rheo={rheo:5.2f}  floor={floor:5.1f} Hz  block~{blk}  -> {struct}", flush=True)
    print(f"appended {out}")
