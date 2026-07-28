"""Frozen model constants for the realism ladder. PURE PYTHON (no jax) so it is the SINGLE
source of truth, imported by BOTH plants.py (the jax plants) and fit_matches.py. fit_matches.py
RE-DERIVES every value here from results/hh_fi.npy and asserts they match, so no hand-copied
number can silently drift (CA1 rule: no file carries an unreproduced result). Do not hand-edit;
change fit_matches.py, re-run it, and paste what it prints.
"""
# HH single-compartment surrogate: rate(I) = sigmoid((I-HH_A)/HH_B) * (HH_C + HH_D*I),
# fit to the JAXLEY f-I in results/hh_fi.npy (R^2 = 0.994 on the firing band).
HH_A, HH_B, HH_C, HH_D = 0.00379, 0.000144, 59.99, 1953.6
I_BLOCK = 0.029                    # depol-block onset current -> common drive x = 1
RHEOBASE_X = 0.004 / I_BLOCK       # ~0.138 : HH's Type-II jump to ~55 Hz

# LIF fixed biophysics (the model's identity; NOT fit)
LIF_THETA, LIF_TAU, LIF_TREF, LIF_SMOOTH = 1.0, 0.05, 0.005, 0.05

# FROZEN 2-knob (gain, base) least-squares matches of the toys to HH's f-I over the writable
# band [55,115] Hz (common drive x in [RHEOBASE_X, 1]). x=0 = silent baseline.
ENC_GAIN, ENC_BASE = 58.1680, 58.9034
LIF_GAIN, LIF_BASE = 9.8020, 3.7804
