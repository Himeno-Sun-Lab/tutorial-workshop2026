"""Tutorial experiment: amyloid-burdened cortex, laptop scale.

Identical to ``tutorial_CN.py`` in every respect - same anatomy, same
connectivity, same drive, same seed - except that a fraction of the excitatory
neurons in each area is replaced by amyloid-affected ones. Any difference in the
results is therefore attributable to that substitution alone.
"""
import os
import runpy

# Results go next to the tutorial, never inside the model repository.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")

# The healthy control this experiment is compared with.
_CN = runpy.run_path(os.path.join(_HERE, "tutorial_CN.py"))


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
    # N_scaling instead. See Appendix A of the notebook.
    "N_scaling": 0.01,      # 17 k neurons
    "K_scaling": 0.1,        # in-degree ~1250 instead of ~12500
    "scaling_type": "sqrt",  # weights scale as 1/sqrt(K_scaling)

    # Rates of the full-scale version of THIS experiment, from which the DC
    # current that restores the mean input lost by downscaling is computed (see
    # tutorial_CN.py). The DC current stands in for recurrent input, so it must
    # be computed from the amyloid network's own rates, not the healthy ones:
    #   python tools/fullscale_rates.py experiments/tutorial_amyloid.py \
    #       --out experiments/fullscale_rates/tutorial_amyloid.pkl
    "fullscale_rates": os.path.join(_HERE, "fullscale_rates", "tutorial_amyloid.pkl"),

    # ---- external drive ---------------------------------------------------
    # eta_ext = mean external input as a multiple of the distance to threshold.
    # Read from tutorial_CN.py, where it is calibrated for the healthy network,
    # so the two cannot drift apart. It is deliberately NOT recalibrated here:
    # pulling the amyloid network back to the healthy working point would
    # remove the very effect under study. Recalibrate tutorial_CN.py instead.
    "input_params": {"eta_ext": _CN["net_params"]["input_params"]["eta_ext"]},

    # ---- connection strength ----------------------------------------------
    # Cortico-cortical excitatory weights, relative to the local ones. These two
    # are the main knobs the project uses to move the network between a quiet
    # ground state and a metastable one.
    "scaling_factors_recurrent": {
        "cc_scalingEtoE": 2.5,
        "cc_scalingEtoI": 2.0 * 2.5,
    },
}

# ---- pathology ------------------------------------------------------------
# Replace a fraction of the excitatory neurons with amyloid-affected ones.
# Each entry of neuron_types is one custom-named type that takes its own
# fraction of every population; fields it leaves out come from the healthy
# ("base") neuron, and ratio_I left out means 0: interneurons stay intact.
#
# ratio_E: amyloid-PET map of an ADNI participant, per Desikan-Killiany area. A
# scalar (0.15) or a dict ({"precuneus": 0.3, "default": 0.05}) works too.
# Parameters: Allen GLIF fits of amyloid-burdened cells. The decisive change is
# V_th - E_L = 15 mV instead of 25 mV. rel_sd = 0: no cell-to-cell spread.
net_params["neuron_types"] = {
    "Abeta": {
        "ratio_E": os.path.join(
            os.getcwd(), "experimental_data", "amyloid", "amyloid_ratio.csv"),
        "neuron_model_E": "iaf_psc_exp",
        "neuron_params_E": {
            "E_L": -65.0, "V_th": -50.0, "V_reset": -70.0, "C_m": 220.0,
            "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0,
        },
        "neuron_param_dist_E": {
            "V_th":  {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.21
            "C_m":   {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.22
            "tau_m": {"distribution": "lognormal", "rel_sd": 0.0},   # fitted: 0.55
        },
    },
}


# ===========================================================================
#  sim_params - what HAPPENS to it
# ===========================================================================
sim_params = {
    # Biological time, in ms: 4000, read from tutorial_CN.py. Every analysis
    # window ends at t_sim, so the two runs are only comparable at the same
    # length. The first second is discarded as a transient, so this leaves 3 s
    # of usable signal - enough for spectra above ~2 Hz, not enough for BOLD.
    # Cluster runs use tens of seconds.
    "t_sim": _CN["sim_params"]["t_sim"],

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

    # Figure resolution of every figure the analysis stage writes.
    "dpi": 120,

    "functconn_corr": {"exclude_diagonal": True},

    # Raster window: the last second of the run, derived from t_sim so that the
    # two cannot drift apart. At this population size a small sample is too
    # sparse to look like a raster, so show half of every population. The bounds
    # must be ints: the analysis stage rejects a value whose type differs from
    # its default (500 / 750).
    "plotRasterArea": {"fraction": 0.5,
                       "low": int(sim_params["t_sim"]) - 1000,
                       "high": int(sim_params["t_sim"])},

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
