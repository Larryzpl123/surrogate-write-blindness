"""Measure the steady-state f-I curve of a Connor-Stevens (CS) neuron for Type-I and Type-II,
to test the paper's core claim that the low-rate GAP (and thus the surrogate-write blindness) is a
signature of the onset bifurcation class: Type-II (Hopf) jumps from 0 to a finite rate (gap present);
Type-I (saddle-node/SNIC) fires at arbitrarily low rates (no gap).

Kinetics + parameters are transcribed verbatim from Fehrman & Meliza (2024), melizalab/mpc-hh,
neuron_scripts/connor_stevens.py + config/config_cs_type_{I,II}.yaml, so this is directly comparable
to that work. Pure numpy/scipy. Run:  python3 measure_cs_fi.py
"""
import numpy as np
from scipy.integrate import odeint


class CS:
    def __init__(self, sim_type):
        self.sim_type = sim_type
        self.C_m = 1.0
        self.E_na, self.E_k, self.E_a = 50.0, -77.0, -80.0
        self.g_na, self.g_k, self.g_a, self.g_l = 120.0, 20.0, 47.7, 0.3
        self.E_l = -22.0 if sim_type == "I" else -72.8   # leak reversal differs by type

    def a_inf(self, V): return (0.0761 * np.exp((V + 99.22) / 31.84) / (1 + np.exp((V + 6.17) / 28.93))) ** (1 / 3)
    def b_inf(self, V): return 1.0 / (1 + np.exp((V + 58.3) / 14.54)) ** 4
    def tau_a(self, V): return 0.3632 + 1.158 / (1 + np.exp((V + 60.96) / 20.12))
    def tau_b(self, V): return 1.24 + 2.678 / (1 + np.exp((V - 55) / 16.027))
    def a_m(self, V): return 3.8 * (-0.1 * (V + 34.7)) / (np.exp(-(V + 34.7) / 10) - 1)
    def a_h(self, V): return 3.8 * 0.07 * np.exp(-(V + 53) / 20)
    def a_n(self, V): return (3.8 / 2) * (-0.01 * (V + 50.7)) / (np.exp(-(V + 50.7) / 10) - 1)
    def b_m(self, V): return 3.8 * 4 * np.exp(-(V + 59.7) / 18)
    def b_h(self, V): return 3.8 / (np.exp(-(V + 23) / 10) + 1)
    def b_n(self, V): return (3.8 / 2) * 0.125 * np.exp(-(V + 60.7) / 80)

    def ode(self, s, t, I):
        V, a, b, m, h, n = s
        I_a = self.g_a * a ** 3 * b * (V - self.E_a) if self.sim_type == "I" else 0.0
        I_na = self.g_na * m ** 3 * h * (V - self.E_na)
        I_k = self.g_k * n ** 4 * (V - self.E_k)
        I_l = self.g_l * (V - self.E_l)
        dV = (I - I_na - I_k - I_a - I_l) / self.C_m
        da = (self.a_inf(V) - a) / self.tau_a(V)
        db = (self.b_inf(V) - b) / self.tau_b(V)
        dm = self.a_m(V) * (1 - m) - self.b_m(V) * m
        dh = self.a_h(V) * (1 - h) - self.b_h(V) * h
        dn = self.a_n(V) * (1 - n) - self.b_n(V) * n
        return [dV, da, db, dm, dh, dn]

    def rest(self):
        V0 = -70.0
        s0 = [V0, self.a_inf(V0), self.b_inf(V0),
              self.a_m(V0) / (self.a_m(V0) + self.b_m(V0)),
              self.a_h(V0) / (self.a_h(V0) + self.b_h(V0)),
              self.a_n(V0) / (self.a_n(V0) + self.b_n(V0))]
        t = np.arange(0, 400, 0.05)
        return odeint(self.ode, s0, t, args=(0.0,))[-1]


def fi(cs, I, rest, T=1000.0, dt=0.1, last=700.0):
    t = np.arange(0, T, dt)
    V = odeint(cs.ode, rest, t, args=(I,))[:, 0]
    seg = V[int((T - last) / dt):]
    above = seg > 0.0
    spikes = int(np.sum((~above[:-1]) & (above[1:])))
    return spikes / (last / 1000.0)


def sweep(typ, I_grid, out=None):
    import sys
    cs = CS(typ)
    rest = cs.rest()                      # computed once, reused across I
    lines = []
    for I in I_grid:
        r = fi(cs, float(I), rest)
        line = f"Type-{typ}  I={I:6.2f}  rate={r:6.1f} Hz"
        print(line); sys.stdout.flush()
        lines.append((float(I), r))
    nz = [r for _, r in lines if r > 0]
    print(f"=== Type-{typ} DONE  min nonzero rate = {min(nz) if nz else 0:.1f} Hz ===")
    if out:
        np.save(out, np.array(lines))
    return lines


if __name__ == "__main__":
    import sys
    typ = sys.argv[1] if len(sys.argv) > 1 else "II"
    # optional custom grid: python3 measure_cs_fi.py I 6 9 0.25  -> arange(6,9,0.25)
    if len(sys.argv) >= 5:
        I_grid = np.round(np.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])), 3)
    else:
        I_grid = np.round(np.arange(0, 42, 2.0), 1)
    sweep(typ, I_grid, out=f"cs_fi_type{typ}.npy")
