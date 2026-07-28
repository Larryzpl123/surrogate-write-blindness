#!/usr/bin/env bash
# Reproduce every result and figure from results/hh_fi.npy. Fails loudly on any error (the gate).
# measure_hh_fi.py (which needs jaxley and regenerates results/hh_fi.npy) is intentionally NOT run
# here: the measured f-I is committed, so this runs with only numpy/scipy/matplotlib.
set -euo pipefail
cd "$(dirname "$0")"
echo "[1/6] fit_matches.py        (GATE: re-derives + ASSERTS the frozen constants from hh_fi.npy)"
python3 fit_matches.py
echo "[2/6] make_figures.py"        ; python3 make_figures.py
echo "[3/6] run_surrogate_blindness.py"; python3 run_surrogate_blindness.py
echo "[4/6] run_robustness.py"      ; python3 run_robustness.py
echo "[5/6] run_fix.py"             ; python3 run_fix.py
echo "[6/6] run_quantify.py"        ; python3 run_quantify.py
echo "ALL OK"
