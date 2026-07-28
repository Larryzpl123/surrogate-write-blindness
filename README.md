# Smooth Surrogate Models Approve Impossible Firing Patterns

Code, figures, and data for the preprint *"Smooth Surrogate Models Approve Impossible Firing
Patterns: A Hidden Flaw in Writing Brain Codes"* (Peilin (Larry) Zhong, 2026).

Preprint (Zenodo, latest version): https://doi.org/10.5281/zenodo.21645655

## What this shows
A smooth, differentiable surrogate of a Hodgkin-Huxley neuron's firing-rate (f-I) curve silently
certifies target firing rates the real neuron cannot produce: the Type-II onset gap below ~55 Hz and
depolarization block above ~115 Hz. The failure is invisible to the surrogate's own loss and is
invariant to fit quality (even a perfect interpolator fails). A feasibility screen against the true
f-I removes it.

## Reproduce
Python 3 with `numpy`, `scipy`, `matplotlib` (plus `jax` + `jaxley` only to re-measure the HH f-I).

    cd code
    python fit_matches.py            # re-derives + asserts the frozen constants from ../results/hh_fi.npy
    python make_figures.py           # regenerates ../figures/ (the four paper figures)
    python run_surrogate_blindness.py   # the core result (surrogate certifies impossible codes)
    python run_robustness.py            # smoothness is the disease, not fit quality
    python run_fix.py                   # the feasibility-screen fix
    python run_quantify.py              # how often / how badly it bites

`results/hh_fi.npy` is the Hodgkin-Huxley f-I curve measured in JAXLEY (`code/measure_hh_fi.py`).

## Layout
    code/       analysis + figure code (plants, testbed, run_*, make_figures, fit_matches, measure_hh_fi)
    figures/    the four paper figures (.png + .pdf)
    results/    hh_fi.npy (measured HH f-I) and run logs

## License
Code: MIT (see `LICENSE`). The manuscript is CC-BY-4.0.
