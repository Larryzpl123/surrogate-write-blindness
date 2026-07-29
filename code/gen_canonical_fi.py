"""Canonical f-I archive for the excitability-class result. Measures each model over a clean
grid (silent -> onset -> rise -> depolarization block) with ONE shared measurement, and saves
results/fi_<MODEL>.npy with columns [I, rate_Hz, Vlate_mV]. The figure reads these .npy files;
no number is ever transcribed by hand (CA1 rule). All four classes expose .ode(s,t,I) and .rest().

Run one model per call (stays under the shell time limit):
    python3 gen_canonical_fi.py HH | WB | CSI | CSII
"""
import sys, numpy as np
from scipy.integrate import odeint
from measure_cs_fi import CS
from measure_real_cells import HH, WB

MODELS = {"HH": HH(), "WB": WB(), "CSI": CS("I"), "CSII": CS("II")}

# per-model current grid (uA/cm^2): FINE onset segment (to resolve the jump/floor) + main
# sweep + high-I points to catch depol block. np.unique sorts + de-dups.
GRIDS = {
    "HH":   np.unique(np.r_[np.arange(0, 20, 1.0), np.arange(6.0, 7.5, 0.1), 25, 30, 40, 60, 80, 100, 150, 200, 300]),
    "WB":   np.unique(np.r_[np.arange(0, 5, 0.25), np.arange(0.14, 0.30, 0.02), 6, 8, 12, 20, 30, 40, 80, 150]),
    "CSI":  np.unique(np.r_[np.arange(0, 42, 2.0), np.arange(8.0, 9.2, 0.1), 60, 100, 150, 175, 200, 250, 350, 500]),
    "CSII": np.unique(np.r_[np.arange(0, 42, 2.0), np.arange(7.10, 7.30, 0.02), 60, 100, 150, 175, 200, 250, 350, 500]),
}


def fi(model, I, T=1000.0, dt=0.05, last=700.0, thresh=0.0):
    t = np.arange(0, T, dt)
    V = odeint(model.ode, model.rest(), t, args=(float(I),))[:, 0]
    seg = V[int((T - last) / dt):]
    above = seg > thresh
    spikes = int(np.sum((~above[:-1]) & (above[1:])))
    v_late = float(np.mean(V[int((T - 30.0) / dt):]))
    return spikes / (last / 1000.0), v_late


if __name__ == "__main__":
    name = sys.argv[1]
    model, grid = MODELS[name], GRIDS[sys.argv[1]]
    rows = []
    for I in grid:
        r, vlate = fi(model, I)
        blk = (r < 1) and (vlate > -40)
        rows.append((float(I), r, vlate))
        np.save(f"../results/fi_{name}.npy", np.array(rows))   # incremental: survives a timeout
        print(f"{name:4}  I={I:7.2f}  rate={r:6.1f} Hz  Vlate={vlate:6.1f}{'  [block]' if blk else ''}", flush=True)
    arr = np.array(rows)
    # summary
    fired = arr[arr[:, 1] > 0]
    rheo = fired[0, 0] if len(fired) else np.nan
    floor = fired[:, 1].min() if len(fired) else np.nan
    blocked = arr[(arr[:, 1] < 1) & (arr[:, 2] > -40) & (arr[:, 0] > rheo)]
    blk_onset = blocked[0, 0] if len(blocked) else np.nan
    print(f"=== {name}: rheobase~{rheo:.2f}  floor(min nonzero rate)={floor:.1f} Hz  "
          f"block onset~{blk_onset} uA/cm2  -> saved ../results/fi_{name}.npy ===")
