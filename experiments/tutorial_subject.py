"""Tutorial experiment: one individual, from their own MRI and PET.

Everything in the other two experiments is group-average anatomy: the connectome
comes from the HCP reference and the amyloid map from one participant applied to
a generic brain. This one is individualised end to end:

    structural connectivity  <- this person's diffusion MRI tractography
    amyloid distribution     <- this person's amyloid-PET
    comparison target        <- this person's own resting-state BOLD

so the functional connectivity the analysis reports is a prediction about *this
person*, checked against *their own* measurement rather than a group average.

Run it the same way as the others (Part 5 of the notebook), or set
RUN_SUBJECT = True in Part 7.4.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_TUTORIAL = os.path.dirname(_HERE)
_RESULTS = os.path.join(_TUTORIAL, "results")
_ADNI = os.path.join(_TUTORIAL, "model", "experimental_data", "ADNI")

SUBJECT = "023_S_6661"


# ===========================================================================
#  net_params - this person's anatomy and pathology
# ===========================================================================
net_params = {
    "outpath": _RESULTS,
    "mode": "INFO",

    "N_scaling": 0.005,
    "K_scaling": 0.1,
    "scaling_type": "sqrt",
    "fullscale_rates": "./simulated_data/base_theory_rates.pkl",
    "input_params": {"eta_ext": 3.45},
    "scaling_factors_recurrent": {
        "cc_scalingEtoE": 2.5,
        "cc_scalingEtoI": 2.0 * 2.5,
    },

    # ---- this person's connectome ----------------------------------------
    # Same variable layout as the group-average HCP file it replaces, so only
    # the path changes. Everything downstream - in-degrees, delays, laminar
    # target patterns - is rebuilt from it.
    "predictive_connectomic_params": {
        "con_path": os.path.join(_ADNI, SUBJECT + "_sc.mat"),
    },
}

# ---- this person's amyloid-PET map ----------------------------------------
net_params["Abeta"] = True
net_params["Abeta_ratio_E"] = os.path.join(_ADNI, SUBJECT + "_abeta_ratio.csv")
net_params["Abeta_ratio_I"] = 0
net_params["neuron_model_Abeta_E"] = "iaf_psc_exp"
net_params["neuron_model_Abeta_I"] = "iaf_psc_exp"
net_params["neuron_params_Abeta_E"] = {
    "E_L": -65.0, "V_th": -50.0, "V_reset": -70.0, "C_m": 220.0,
    "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0,
}
net_params["neuron_params_Abeta_I"] = {
    "E_L": -70.0, "V_th": -45.0, "V_reset": -70.0, "C_m": 100.0,
    "tau_m": 10.0, "tau_syn_ex": 2.0, "tau_syn_in": 2.0, "t_ref": 2.0,
}
_no_spread = {
    "V_th":  {"distribution": "lognormal", "rel_sd": 0.0},
    "C_m":   {"distribution": "lognormal", "rel_sd": 0.0},
    "tau_m": {"distribution": "lognormal", "rel_sd": 0.0},
}
net_params["neuron_param_dist_Abeta_E"] = _no_spread
net_params["neuron_param_dist_Abeta_I"] = _no_spread


# ===========================================================================
#  sim_params
# ===========================================================================
sim_params = {
    "t_sim": 5000.0,
    "master_seed": 2903,
    "rec_dev": ["spike_recorder"],
    "mode": "INFO",
}


# ===========================================================================
#  ana_params - compare against this person's own fMRI
# ===========================================================================
_TRANSIENT = 1000.0

ana_params = {
    "mode": "INFO",
    "fast_lfp_kernel": True,
    "dpi": 120,

    # The default is the sentinel "rsDATA_7T_DKparcel.npy", which selects the
    # group-averaged 7T reference. Any other value is read as a path to one
    # person's BOLD, here 835 frames x 68 DK areas.
    "experimental_fc": os.path.join(_ADNI, SUBJECT + "_bold.npy"),

    "functconn_corr": {"exclude_diagonal": False},
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
