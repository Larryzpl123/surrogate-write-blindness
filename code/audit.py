"""Number audit: every headline number in the manuscript must be reproduced by a script here.

It (1) checks the writable band directly from the raw f-I, and (2) runs the analysis scripts and
verifies each claimed result number appears in their output within tolerance. Exits non-zero if any
claim is unsupported. This is the CA1-style discipline: the paper's credibility rests on the numbers,
so a script proves each one is produced, not transcribed. (It cannot catch a wrong sentence with no
digit in it -- read the prose for those.)

Run from code/:  python3 audit.py
"""
import subprocess, re, sys, os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1) writable band, straight from the measured f-I
raw = np.load("../results/hh_fi.npy"); rates = raw[:, 1]; nz = rates[rates > 0]
band = (round(float(nz.min())), round(float(nz.max())))
assert band == (55, 115), f"writable band drifted from (55,115): {band}"
print(f"data check OK: writable band = {band[0]}-{band[1]} Hz")

# 2) result numbers vs script output
SCRIPTS = ["fit_matches.py", "run_surrogate_blindness.py", "run_robustness.py", "run_fix.py", "run_quantify.py"]
out = ""
for s in SCRIPTS:
    print("running", s, "...")
    out += subprocess.run([sys.executable, s], capture_output=True, text=True).stdout
vals = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", out)]
has = lambda v, tol: any(abs(n - v) <= tol for n in vals)

# (value, description, tolerance)
CLAIMS = [
    (0.965, "sigmoid x affine in-band R^2", 0.01),
    (1.0,   "monotone interpolator in-band R^2", 0.005),
    (0.67,  "logistic in-band R^2", 0.01),
    (264,   "network true error (Hz)", 3),
    (233,   "robustness min true error (Hz)", 3),
    (53,    "percent of cells silently miswritten", 1),
    (109,   "mean per-code error, U[0,120] (Hz)", 2),
    (104,   "median per-code error (Hz)", 2),
    (87,    "fix true error (Hz)", 1.5),
    (15,    "per-cell mean error (Hz)", 1),
    (12,    "cells wrong out of 20", 0.5),
]
bad = [(v, c) for v, c, t in CLAIMS if not has(v, t)]
if bad:
    print("\nUNSUPPORTED (not reproduced by any script output):")
    for v, c in bad:
        print(f"  {v}  ({c})")
    sys.exit(1)
print(f"\nAUDIT OK: writable band + all {len(CLAIMS)} result numbers reproduced by the scripts.")
