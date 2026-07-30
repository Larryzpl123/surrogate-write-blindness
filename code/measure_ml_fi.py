"""Morris-Lecar minimal model: the canonical 2-variable model in which a single parameter change
switches the onset bifurcation between Type-I (SNIC, continuous f-I to arbitrarily low rate) and
Type-II (Andronov-Hopf, f-I jumps from zero to a finite floor). This is the analytic minimal case
that grounds the excitability-class result of the main paper: the low-rate gap is a signature of
the Hopf onset, not of any particular biophysical model.

Equations (Morris & Lecar 1981; Rinzel & Ermentrout 1989/1998; see Scholarpedia, Lecar 2007):
    C V' = I - gCa Mss(V)(V-VCa) - gK W (V-VK) - gL(V-VL)
    W' = phi (Wss(V) - W) cosh((V-V3)/(2 V4))
    Mss(V) = 0.5(1+tanh((V-V1)/V2)),  Wss(V) = 0.5(1+tanh((V-V3)/V4))
Parameter sets are the standard SNIC / Hopf regimes (Ermentrout & Terman 2010; Izhikevich 2007).
Verification is by BEHAVIOR: Type-I must give a continuous f-I to low rate, Type-II a jump.

Pure numpy/scipy. Run:  python3 measure_ml_fi.py I   |   II   (optionally: start stop step)
"""
import sys, numpy as np
from scipy.integrate import odeint


class ML:
    # common parameters
    C, gL, gK = 20.0, 2.0, 8.0
    VL, VCa, VK = -60.0, 120.0, -84.0
    V1, V2 = -1.2, 18.0

    def __init__(self, typ):
        if typ == "I":            # SNIC -> Type-I (continuous f-I)
            self.gCa, self.V3, self.V4, self.phi = 4.0, 12.0, 17.4, 1.0 / 15.0
        else:                      # Andronov-Hopf -> Type-II (gap)
            self.gCa, self.V3, self.V4, self.phi = 4.4, 2.0, 30.0, 0.04
        self.typ = typ

    def mss(self, V): return 0.5 * (1 + np.tanh((V - self.V1) / self.V2))
    def wss(self, V): return 0.5 * (1 + np.tanh((V - self.V3) / self.V4))

    def ode(self, s, t, I):
        V, W = s
        dV = (I - self.gCa * self.mss(V) * (V - self.VCa) - self.gK * W * (V - self.VK)
              - self.gL * (V - self.VL)) / self.C
        dW = self.phi * (self.wss(V) - W) * np.cosh((V - self.V3) / (2 * self.V4))
        return [dV, dW]

    def rest(self):
        V0 = -60.0
        return odeint(self.ode, [V0, self.wss(V0)], np.arange(0, 500, 0.1), args=(0.0,))[-1]


def fi(m, I, T=2200.0, dt=0.05, last=1600.0, thresh=0.0):
    t = np.arange(0, T, dt)
    V = odeint(m.ode, m.rest(), t, args=(float(I),))[:, 0]
    seg = V[int((T - last) / dt):]
    above = seg > thresh
    spikes = int(np.sum((~above[:-1]) & (above[1:])))
    return spikes / (last / 1000.0)


if __name__ == "__main__":
    typ = sys.argv[1] if len(sys.argv) > 1 else "II"
    if len(sys.argv) >= 5:
        grid = np.round(np.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])), 3)
    elif typ == "I":     # coarse + FINE onset at 0.5 (SNIC rheobase ~38-40; resolve it so the
                         # continuous onset is visible rather than straddling a coarse grid step)
        grid = np.unique(np.r_[0, 10, 20, 28, 34, 36, np.arange(37.5, 42.5, 0.5),
                              44, 46, 48, 60, 80, 100.0])
    else:                # coarse + fine onset (Hopf onset ~88-90)
        grid = np.unique(np.r_[0, 20, 40, 60, 75, 80, 84, 86, np.arange(87.5, 91.0, 0.5),
                              92, 105, 120, 140.0])
    m = ML(typ)
    rates = []
    for I in grid:
        r = fi(m, float(I))
        rates.append((float(I), r))
        np.save(f"../results/fi_ML{typ}.npy", np.array(rates))   # incremental: survives a timeout
        print(f"ML-{typ}  I={I:6.1f}  rate={r:6.1f} Hz", flush=True)
    nz = [r for _, r in rates if r > 0]
    print(f"=== ML-{typ}: min nonzero rate = {min(nz) if nz else 0:.1f} Hz "
          f"(Type-I expect ~0 continuous; Type-II expect a finite floor = gap) ===")
