"""Single-compartment f-I of REAL published cells, measured with the SAME scipy tool as
measure_cs_fi.py, so HH / Connor-Stevens / Wang-Buzsaki sit on one footing.

C = classic Hodgkin-Huxley (1952) squid axon (modern -65 mV convention). This is the same
    model family JAXLEY's `HH` channel implements, i.e. the paper's reference plant. An
    independent scipy cross-check that it is Type-II (0 -> finite floor = gap) + depol block.
A = Wang-Buzsaki (1996) hippocampal fast-spiking interneuron. A real published HIPPOCAMPAL
    cell -> measure its excitability class (Type-I vs Type-II) and whether it has a low-rate gap.
    Kinetics filled in only AFTER verification against a source (never from memory).

Pure numpy/scipy. Run:  python3 measure_real_cells.py HH   |   python3 measure_real_cells.py WB
"""
import numpy as np
from scipy.integrate import odeint


class HH:
    """Classic Hodgkin-Huxley (1952), textbook -65 mV convention. uA/cm^2, mS/cm^2, mV, ms."""
    Cm = 1.0
    gNa, gK, gL = 120.0, 36.0, 0.3
    ENa, EK, EL = 50.0, -77.0, -54.387

    def am(self, V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    def bm(self, V): return 4 * np.exp(-(V + 65) / 18)
    def ah(self, V): return 0.07 * np.exp(-(V + 65) / 20)
    def bh(self, V): return 1 / (1 + np.exp(-(V + 35) / 10))
    def an(self, V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    def bn(self, V): return 0.125 * np.exp(-(V + 65) / 80)

    def ode(self, s, t, I):
        V, m, h, n = s
        Iion = self.gNa * m**3 * h * (V - self.ENa) + self.gK * n**4 * (V - self.EK) + self.gL * (V - self.EL)
        dV = (I - Iion) / self.Cm
        dm = self.am(V) * (1 - m) - self.bm(V) * m
        dh = self.ah(V) * (1 - h) - self.bh(V) * h
        dn = self.an(V) * (1 - n) - self.bn(V) * n
        return [dV, dm, dh, dn]

    def rest(self):
        V0 = -65.0
        s0 = [V0, self.am(V0)/(self.am(V0)+self.bm(V0)),
              self.ah(V0)/(self.ah(V0)+self.bh(V0)),
              self.an(V0)/(self.an(V0)+self.bn(V0))]
        return odeint(self.ode, s0, np.arange(0, 400, 0.05), args=(0.0,))[-1]


def _exprel(x):
    # (exp(x)-1)/x, -> 1 as x -> 0 (removable singularity guard, matches Brian2's exprel)
    return 1.0 if abs(x) < 1e-9 else np.expm1(x) / x


class WB:
    """Wang-Buzsaki (1996) hippocampal fast-spiking interneuron, single compartment.
    Kinetics transcribed verbatim from the Brian2 'frompapers' example (which encodes the
    original J Neurosci 16(20):6402 equations), using exprel to guard the m/n singularities.
    m is instantaneous (m = m_inf); state = [V, h, n]. phi = 5. uA/cm^2, mS/cm^2, mV, ms."""
    Cm = 1.0
    gNa, gK, gL = 35.0, 9.0, 0.1
    ENa, EK, EL = 55.0, -90.0, -65.0
    phi = 5.0

    def alpha_m(self, V): return 1.0 / _exprel(-(V + 35) / 10)
    def beta_m(self, V): return 4 * np.exp(-(V + 60) / 18)
    def m_inf(self, V): return self.alpha_m(V) / (self.alpha_m(V) + self.beta_m(V))
    def alpha_h(self, V): return 0.07 * np.exp(-(V + 58) / 20)
    def beta_h(self, V): return 1.0 / (np.exp(-0.1 * (V + 28)) + 1)
    def alpha_n(self, V): return 0.1 / _exprel(-(V + 34) / 10)
    def beta_n(self, V): return 0.125 * np.exp(-(V + 44) / 80)

    def ode(self, s, t, I):
        V, h, n = s
        m = self.m_inf(V)
        Iion = self.gNa * m**3 * h * (V - self.ENa) + self.gK * n**4 * (V - self.EK) + self.gL * (V - self.EL)
        dV = (I - Iion) / self.Cm
        dh = self.phi * (self.alpha_h(V) * (1 - h) - self.beta_h(V) * h)
        dn = self.phi * (self.alpha_n(V) * (1 - n) - self.beta_n(V) * n)
        return [dV, dh, dn]

    def rest(self):
        V0 = -65.0
        s0 = [V0, self.alpha_h(V0)/(self.alpha_h(V0)+self.beta_h(V0)),
              self.alpha_n(V0)/(self.alpha_n(V0)+self.beta_n(V0))]
        return odeint(self.ode, s0, np.arange(0, 400, 0.05), args=(0.0,))[-1]


MODELS = {"HH": HH, "WB": WB}


def fi(model, I, T=1000.0, dt=0.05, last=700.0, thresh=0.0):
    t = np.arange(0, T, dt)
    V = odeint(model.ode, model.rest(), t, args=(I,))[:, 0]
    seg = V[int((T - last) / dt):]
    above = seg > thresh
    spikes = int(np.sum((~above[:-1]) & (above[1:])))
    v_late = float(np.mean(V[int((T - 30.0) / dt):]))
    return spikes / (last / 1000.0), v_late


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "HH"
    if len(sys.argv) >= 5:
        grid = np.round(np.arange(float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])), 3)
    else:
        grid = np.round(np.arange(0, 22, 1.0), 1)
    model = MODELS[name]()
    rows = []
    for I in grid:
        r, vlate = fi(model, float(I))
        block = "  [block]" if (r < 1 and vlate > -40) else ""
        print(f"{name}  I={I:7.2f}  rate={r:6.1f} Hz  Vlate={vlate:6.1f}{block}")
        rows.append((float(I), r))
    nz = [r for _, r in rows if r > 0]
    print(f"=== {name} DONE  min nonzero rate = {min(nz) if nz else 0:.1f} Hz ===")
    np.save(f"cellfi_{name}.npy", np.array(rows))
