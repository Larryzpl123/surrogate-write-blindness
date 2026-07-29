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

**Excitability class (added in v1.1.0).** The failure is a signature of the target cell's onset
bifurcation class. It is present in Type-II models (Hodgkin-Huxley; Connor-Stevens in its Type-II
configuration, gap to ~67 Hz) and absent in Type-I models (Connor-Stevens Type-I; the Wang-Buzsaki
hippocampal interneuron), which fire continuously toward zero. The effect is robust to +/-20%
variation in sodium and potassium conductance. Figures 5 and 6 show the f-I curves by class and the
resulting write outcomes.

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

### Excitability-class experiments (figures 5-6, v1.1.0)
Pure `numpy`/`scipy`/`matplotlib` (no `jaxley` needed):

    cd code
    python gen_canonical_fi.py HH        # measure + archive each model's f-I -> ../results/fi_HH.npy
    python gen_canonical_fi.py WB        #   (repeat for WB, CSI, CSII)
    python gen_canonical_fi.py CSI
    python gen_canonical_fi.py CSII
    python make_excitability_figure.py   # figure 5 (f-I by class, gap vs no gap) from the fi_*.npy
    python writeblind_by_class.py        # figure 6 (requested vs delivered rate, Type-II vs Type-I)
    python robustness_sweep.py CSII      # conductance sweep -> ../results/robustness_gNaK.csv
    python robustness_sweep.py CSI

`measure_cs_fi.py` (Connor-Stevens Type-I/II) and `measure_real_cells.py` (classic HH, Wang-Buzsaki)
hold the model kinetics, transcribed from their cited primary sources.

## Layout
    code/       analysis + figure code (plants, testbed, run_*, make_figures, fit_matches, measure_hh_fi;
                measure_cs_fi, measure_real_cells, gen_canonical_fi, make_excitability_figure,
                writeblind_by_class, robustness_sweep)
    figures/    the six paper figures (.png + .pdf)
    results/    hh_fi.npy, fi_{HH,WB,CSI,CSII}.npy (measured f-I), robustness_gNaK.csv, run logs

## License
Code: MIT (see `LICENSE`). The manuscript is CC-BY-4.0.
