"""Tutorial experiment: amyloid-burdened (AD) cortex, laptop scale.

Identical to ``tutorial_CN.py`` in every respect - same anatomy, same
connectivity, same drive, same seed - except that a fraction of the excitatory
neurons in each area is replaced by amyloid-affected ones. Any difference in the
results is therefore attributable to that substitution alone.
"""
import os

# Results go next to the tutorial, never inside the model repository.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")


# ===========================================================================
#  net_params - what the network IS
# ===========================================================================
net_params = {
    "outpath": _RESULTS,
    "mode": "INFO",

    # ---- size -------------------------------------------------------------
    # The full model is 3.47 M neurons and 42.8 G synapses. Downscaling keeps
    # N_scaling of the neurons and K_scaling of the in-degree; synapses (and so
    # memory) scale as the product. K_scaling is the one that changes the
    # dynamics, so we keep it at the value the project validates and shrink
    # N_scaling instead. See Part 3 of the notebook.
    "N_scaling": 0.005,      # 17 k neurons
    "K_scaling": 0.1,        # in-degree ~1250 instead of ~12500
    "scaling_type": "sqrt",  # weights scale as 1/sqrt(K_scaling)

    # Mean-field rates of the full-scale model, needed to compute the DC current
    # that compensates the mean drive lost by downscaling. Required whenever
    # N_scaling or K_scaling differ from 1.
    "fullscale_rates": "./simulated_data/base_theory_rates.pkl",

    # ---- external drive ---------------------------------------------------
    # eta_ext = mean external input as a multiple of the distance to threshold.
    # 1.1 is the calibrated FULL-SCALE value; a downscaled network needs more,
    # because the sqrt weight rule preserves input variance but not its mean.
    # 3.45 was measured with tools/eta_ext_search.py for exactly this N/K, and
    # puts the network at E ~3.0 Hz, I ~12.1 Hz.
    "input_params": {"eta_ext": 3.45},

    # ---- connection strength ----------------------------------------------
    # Cortico-cortical excitatory weights, relative to the local ones. These two
    # are the main knobs the project uses to move the network between a quiet
    # ground state and a metastable one.
    "scaling_factors_recurrent": {
        "cc_scalingEtoE": 2.5,
        "cc_scalingEtoI": 2.0 * 2.5,
    },

    # ---- pathology --------------------------------------------------------
    # Replace a fraction of the excitatory neurons with amyloid-affected ones.
    "Abeta": True,
}

# The fraction per area comes from an amyloid-PET map of ADNI participant
# 002_S_0729, parcellated into Desikan-Killiany areas and shipped with the
# repository. A scalar (0.15), a dict ({"precuneus": 0.3, "default": 0.05}) or a
# per-layer dict work just as well.
net_params["Abeta_ratio_E"] = os.path.join(
    os.getcwd(), "experimental_data", "abeta", "002_S_0729_abeta_ratio.csv")
net_params["Abeta_ratio_I"] = 0          # inhibitory neurons left intact

net_params["neuron_model_Abeta_E"] = "iaf_psc_exp"
net_params["neuron_model_Abeta_I"] = "iaf_psc_exp"

# Fitted to Allen Brain Cell Atlas GLIF recordings of amyloid-burdened cells.
# The decisive change is the distance to threshold: V_th - E_L = 15 mV here,
# against 25 mV for a healthy excitatory neuron. Everything else is unchanged.
net_params["neuron_params_Abeta_E"] = {
    "E_L": -65.0, "V_th": -50.0, "V_reset": -70.0, "C_m": 220.0,
    "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0,
}
net_params["neuron_params_Abeta_I"] = {
    "E_L": -70.0, "V_th": -45.0, "V_reset": -70.0, "C_m": 100.0,
    "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0,
}

# Cell-to-cell heterogeneity of those parameters. rel_sd = 0 means every
# affected neuron is identical; the commented values are the fitted spreads.
_no_spread = {
    "V_th":  {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.21
    "C_m":   {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.22
    "tau_m": {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.55
}
net_params["neuron_param_dist_Abeta_E"] = _no_spread
net_params["neuron_param_dist_Abeta_I"] = _no_spread


# ===========================================================================
#  sim_params - what HAPPENS to it
# ===========================================================================
sim_params = {
    # Biological time, in ms. The first second is discarded as a transient, so
    # this leaves 4 s of usable signal - enough for spectra above ~2 Hz, not
    # enough for BOLD. Cluster runs use tens of seconds.
    "t_sim": 5000.0,

    # Seeds connectivity, weights, delays, the Poisson background and the
    # initial membrane potentials. Change it and you get a different sample of
    # the same model - which is how you measure your own noise floor.
    "master_seed": 2903,

    # Recording devices. "I_syn" additionally records synaptic currents for an
    # exact LFP proxy, at the cost of very large output files.
    "rec_dev": ["spike_recorder"],

    "mode": "INFO",

    # Oscillatory stimulation, e.g. 40 Hz into the precuneus. Uncomment to try.
    # "stimuli": {
    #     ("precuneus", "II/III", "E"): {
    #         "type": "ac",
    #         "params": {"amplitude": 400.0, "offset": 0.0, "frequency": 40.0,
    #                    "phase": 0.0, "start": 1000.0, "stop": 5000.0},
    #     },
    # },
}


# ===========================================================================
#  ana_params - what is MEASURED afterwards
# ===========================================================================
# Discard the settling transient everywhere. Every analysis takes its own
# window parameter, so they are set together here.
_TRANSIENT = 1000.0          # ms

ana_params = {
    "mode": "INFO",

    # Kernel LFP proxy from the 1 ms rate histogram: fast, and its 500 Hz
    # Nyquist limit is far above the 120 Hz we look at. Set False for the exact
    # (slower) version built from raw spike times.
    "fast_lfp_kernel": True,

    # Figure resolution. The default 600 dpi produces very large PNGs.
    "dpi": 200,

    "functconn_corr": {"exclude_diagonal": False},

    # Raster window: the last second of the run. At this population size a small
    # sample is too sparse to look like a raster, so show half of every
    # population.
    "plotRasterArea": {"fraction": 0.5, "low": 4000, "high": 5000},

    "meanFiringRate":          {"t_start": _TRANSIENT},
    "individualFiringRate":    {"t_start": _TRANSIENT},
    "rate_histogram":          {"t_start": _TRANSIENT},
    "cv":                      {"t_start": _TRANSIENT},
    "lvr":                     {"t_start": _TRANSIENT},
    "correlation_coefficient": {"tmin": _TRANSIENT},
    "compute_louvain":         {"tmin": _TRANSIENT},
    "calcLfpCoherence":        {"tmin": _TRANSIENT},
    "calcLfpPlv":              {"tmin": _TRANSIENT},
    "calcAreaLfpCorrelation":  {"tmin": _TRANSIENT},
    "calcAreaLfpCoherence":    {"tmin": _TRANSIENT},
    "plotAllLfpProxyPSD":      {"tmin": _TRANSIENT},
    "plotAreaLfpPsd":          {"tmin": _TRANSIENT},
    "plotLfpArea":             {"tmin": _TRANSIENT},
    "plotLfpPsdArea":          {"tmin": _TRANSIENT},
    "plotLfpSpectrogramArea":  {"tmin": _TRANSIENT},
    "plotLfpWaveletArea":      {"tmin": _TRANSIENT},
    "plotConnectivities":      {"tmin": _TRANSIENT},
    "plotBOLD":                {"tmin": _TRANSIENT},
}
