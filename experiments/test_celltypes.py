"""Test experiment: one excitatory (Pyr) and two inhibitory (PV, SST) cell types.

Identical to ``tutorial_CN.py`` except for the neuron types. It exists to
exercise several neuron types with different synaptic time constants and
refractory periods in one network; it is not calibrated, so its working point
is not a target.

Pyr is the base excitatory neuron, with the repository defaults. Every
inhibitory population is split into PV and SST with layer-dependent fractions
that sum to 1, so (up to rounding) no base inhibitory neuron is created. The
base inhibitory parameters still matter: the synaptic weights (PSC amplitudes)
onto and the external drive of the inhibitory populations are computed from
them. They are therefore left at the repository default, an "average
interneuron", and PV and SST deviate from it in opposite directions.
"""
import os
import runpy
from copy import deepcopy

_HERE = os.path.dirname(os.path.abspath(__file__))
_CN = runpy.run_path(os.path.join(_HERE, "tutorial_CN.py"))

net_params = deepcopy(_CN["net_params"])
sim_params = deepcopy(_CN["sim_params"])
ana_params = deepcopy(_CN["ana_params"])

# ---- cell types -------------------------------------------------------------
# Every type receives the PSC amplitudes computed for the base neuron, so a
# type with longer synaptic time constants receives more charge per spike. The
# deviations below are kept moderate for that reason: in this
# inhibition-dominated network a GABA_A time constant of 4 ms onto Pyr (instead
# of 2 ms) already silences the excitatory populations. The working point is
# very sensitive to the I -> I time constants: with these values the network
# settles in an active state (E ~20 Hz, I ~49 Hz), while PV tau_syn_in = 1 ms
# instead of 1.5 ms silences the excitatory populations again.
#
# Pyr: regular-spiking pyramidal cell, the base excitatory neuron
# (neuron_params_E, repository default: tau_syn 2 / 2 ms, t_ref 2 ms).

# Fraction of SST among the interneurons of each layer (PV takes the rest).
_SST_FRACTION = {"II/III": 0.35, "IV": 0.25, "V": 0.40, "VI": 0.40}

net_params["neuron_types"] = {
    # PV: fast-spiking basket cell. Low input resistance (short tau_m), fast
    # synapses, short refractory period.
    "PV": {
        "ratio_I": {"default": {l: 1.0 - f for l, f in _SST_FRACTION.items()}},
        "neuron_params_I": {
            "tau_m": 8.0, "tau_syn_ex": 1.5, "tau_syn_in": 1.5, "t_ref": 1.0,
        },
    },
    # SST: slower, larger cell with slower excitatory and inhibitory synapses
    # and some cell-to-cell spread of the threshold.
    "SST": {
        "ratio_I": {"default": dict(_SST_FRACTION)},
        "neuron_params_I": {
            "C_m": 120.0, "tau_m": 12.0,
            "tau_syn_ex": 3.0, "tau_syn_in": 4.0, "t_ref": 2.0,
        },
        "neuron_param_dist_I": {
            "V_th": {"distribution": "lognormal", "rel_sd": 0.1},
        },
    },
}

# The full-scale rates of THIS configuration, from which the downscaling DC
# compensation is computed:
#   python tools/fullscale_rates.py experiments/test_celltypes.py --iters 40 \
#       --out experiments/fullscale_rates/test_celltypes.pkl
net_params["fullscale_rates"] = os.path.join(
    _HERE, "fullscale_rates", "test_celltypes.pkl")
