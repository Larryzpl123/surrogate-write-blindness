"""Plants: the 'brain' being written to. Each returns a differentiable
    plant_fn(stim) -> rates            # rates: (n_cells,) per-cell firing rate (Hz)

ALL three rungs share a COMMON per-cell drive  x = W @ stim  (x=0 = silent baseline,
x=1 = HH depol-block onset), then apply their OWN frozen f-I on x. The two toy plants'
2 knobs (gain, base) were set ONCE by least-squares to HH's f-I over the writable band
[55,115] Hz (x in [RHEOBASE_X, 1]); constants live in matched_params.py and are re-derived
+ asserted from results/hh_fi.npy by fit_matches.py. The knob COUNT (=2) is the guardrail:
2 dof align level+slope to HH but cannot erase shape, so in-band the three agree and at the
edges they diverge -- HH is silent below x=RHEOBASE_X (Type-II floor) and blocks above x=1,
the toys do neither. The optimizer only ever chooses stim; these params never move, so there
is no per-target refit.
"""
import jax
import jax.numpy as jnp

from matched_params import (
    HH_A, HH_B, HH_C, HH_D, I_BLOCK, RHEOBASE_X,
    LIF_THETA, LIF_TAU, LIF_TREF, LIF_SMOOTH,
    ENC_GAIN, ENC_BASE, LIF_GAIN, LIF_BASE,
)

__all__ = ["make_coupling", "encoder_plant", "lif_plant", "hh_plant",
           "RHEOBASE_X", "I_BLOCK"]


def make_coupling(key, n_cells, n_stim_max, w_scale=2.0):
    """Electrode->cell coupling W (n_cells x n_stim_max) on the COMMON drive x = W @ stim.
    Nested columns (using n_stim <= n_stim_max = the first n_stim COLUMNS) so adding an
    electrode never removes control authority -> reachability-vs-electrode-count is clean and
    monotonic. w_scale sets how far a unit stim pushes x; foundation_check.py reports coverage."""
    return (w_scale / jnp.sqrt(n_stim_max)) * jax.random.normal(key, (n_cells, n_stim_max))


def _hh_rate_from_x(x):
    I = I_BLOCK * x
    return jax.nn.sigmoid((I - HH_A) / HH_B) * (HH_C + HH_D * I)


def encoder_plant(W):
    """Rung 0 toy: rate = softplus(ENC_GAIN * x + ENC_BASE), x = W @ stim. Frozen match to HH.
    Threshold-free: no true silent state (baseline ~59 Hz), so a cell must be driven NEGATIVE
    to be silenced -- the lower-edge realism the band-only match exposes."""
    def plant_fn(stim):
        return jax.nn.softplus(ENC_GAIN * (W @ stim) + ENC_BASE)
    return plant_fn


def lif_plant(W):
    """Rung 1: LIF steady-state f-I on u = LIF_GAIN * x + LIF_BASE, x = W @ stim. Frozen match.
    Closed-form (exact for the one-time-bin rate target: no ODE, no spike non-differentiability):
        g = smooth * softplus((u - theta)/smooth);   rate = 1 / (t_ref + tau * log1p(theta/g))."""
    def plant_fn(stim):
        u = LIF_GAIN * (W @ stim) + LIF_BASE
        g = jnp.maximum(LIF_SMOOTH * jax.nn.softplus((u - LIF_THETA) / LIF_SMOOTH), 1e-9)
        return 1.0 / (LIF_TREF + LIF_TAU * jnp.log1p(LIF_THETA / g))
    return plant_fn


def hh_plant(W):
    """Rung 3: single-compartment HH surrogate (the reference), rate = HH_fI(I_BLOCK*(W@stim)).
    Silent for x < RHEOBASE_X (Type-II). Keep x <= 1 (target active rate <= 115 Hz + amp budget)
    so the surrogate stays valid; real HH's depol block above x=1 is a documented target CEILING,
    deliberately NOT modeled here (it would make the f-I non-monotonic and trap the optimizer)."""
    def plant_fn(stim):
        return _hh_rate_from_x(W @ stim)
    return plant_fn
