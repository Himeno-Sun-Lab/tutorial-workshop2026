"""Tutorial experiment: healthy (CN) cortex, laptop scale.

An experiment is exactly this: one Python file defining three
dictionaries. Everything not mentioned here keeps the repository default from
``src/default_net_params.py``, ``src/default_sim_params.py`` and
``src/default_ana_params.py``.

The pipeline reads this file three times - once per stage - so it must stay a
plain, side-effect-free description of *what to run*, never of *how to run it*.
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
    # N_scaling instead. See Appendix A of the notebook.
    "N_scaling": 0.01,        # 34.7 k neurons, 4.3e7 synapses (~5.9 GB)
    "K_scaling": 0.1,        # in-degree ~1250 instead of ~12500
    "scaling_type": "sqrt",  # weights scale as 1/sqrt(K_scaling)

    # Rates of the full-scale version of THIS experiment. The sqrt weight rule
    # preserves the variance of the synaptic input but keeps only sqrt(K_scaling)
    # of its mean; the rest is restored as a DC current computed from these
    # rates. They must match every parameter below (drive, weights, pathology),
    # so they were generated for this file by self-consistent iteration:
    #   python tools/fullscale_rates.py experiments/tutorial_CN.py --iters 40 \
    #       --out experiments/fullscale_rates/tutorial_CN.pkl
    # (the default 10 iterations stop short of the fixed point at this scale)
    # Re-run that whenever you change eta_ext, the weights or the anatomy.
    # Required whenever N_scaling or K_scaling differ from 1.
    "fullscale_rates": os.path.join(_HERE, "fullscale_rates", "tutorial_CN.pkl"),

    # ---- external drive ---------------------------------------------------
    # eta_ext = mean external input as a multiple of the distance to threshold.
    # 1.1 is the repository default and the drive of the full-scale runs
    # (notebook Part 6). The downscaled network does not reproduce the
    # full-scale working point at the same drive (notebook Appendix A.5): at 1.1
    # it sits at E ~0.3 Hz, so its drive is calibrated on its own. Of the scan
    # 4.5 / 5.0 / 5.5, only 5.0 holds both target bands (4.5: E 2.3 Hz; 5.5:
    # I 17.9 Hz), and puts the network at E ~3.5 Hz, I ~13.4 Hz - which is the
    # state the fullscale_rates file above holds (E mean 3.54 Hz, I mean
    # 13.41 Hz, 26 damped iterations at this drive), so the two must be
    # changed together: the DC compensation is computed from those rates, and
    # pairing them with a weaker drive leaves every neuron ~50 mV below
    # threshold and the network silent after the transient.
    "input_params": {"eta_ext": 5.0},

    # ---- connection strength ----------------------------------------------
    # Cortico-cortical excitatory weights, relative to the local ones. These two
    # are the main knobs the project uses to move the network between a quiet
    # ground state and a metastable one. 2.5 / 5.0 equals the repository
    # default; it is set here so that the control panel shows it.
    "scaling_factors_recurrent": {
        "cc_scalingEtoE": 2.5,
        "cc_scalingEtoI": 2.0 * 2.5,
    },

    # ---- pathology --------------------------------------------------------
    # No neuron_types: healthy control; see tutorial_amyloid.py
}


# ===========================================================================
#  sim_params - what HAPPENS to it
# ===========================================================================
sim_params = {
    # Biological time, in ms. The first second is discarded as a transient, so
    # this leaves 3 s of usable signal - enough for spectra above ~2 Hz, not
    # enough for BOLD. Cluster runs use tens of seconds.
    "t_sim": 4000.0,

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
