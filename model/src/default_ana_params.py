"""
Provides default parameters used in the analysis.
"""

params = {}
# Logging level. One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
params['mode'] = 'DEBUG'
params['experimental_fc'] = 'rsDATA_7T_DKparcel.npy'

params['mplstyles'] = 'misc/mplstyles/report_plots_master.mplstyle'
params['extension'] = 'png'
# Resolution (dots per inch) for every saved figure. Passed explicitly to each
# savefig() call so it does not depend on matplotlib's ambient rcParams (which
# plt.style.use() mutates globally and, in a few places, never resets).
params['dpi'] = 120
params['seed'] = 2903
params['python_sort'] = True
params["I_syn_python_sort"] = False

# The rate histogram always starts at t = 0 (see analysis.calc_rates); every
# analysis built on it applies its own window.
params['rate_histogram'] = {
    'binsize': 1.,    # in ms
    't_stop': None    # in ms
}

params['meanFiringRate'] = {
    't_start': 500.,  # in ms
    't_stop': None    # in ms
}

params['individualFiringRate'] = {
    't_start': 500.,  # in ms
    't_stop': None    # in ms
}

params['correlation_coefficient'] = {
    'subsample': 2000,
    'tbin': 1.,    # in ms
    'tmin': 500.,  # in ms
    'tmax': None   # in ms
}

params['compute_louvain'] = {
    'tmin': 500.  # in ms
}

params['cv'] = {
    't_start': 500.,  # in ms
    't_stop': None,   # in ms
    'min_spikes': 10
}

params['lvr'] = {
    't_start': 500.,  # in ms
    't_stop': None,   # in ms
    'min_spikes': 10
}

params['functconn_corr'] = {
    'exclude_diagonal': True
}

params['plotAllFiringRatesSummary'] = {
    'num_col': 4,
    't_start': 500.  # in ms, start of the plotted window
    }

params['plotAllSynapticCurrentsSummary'] = {
    'num_col': 4
    }

params['plotAllBOLDSignalSummary'] = {
    'num_col': 4
    }

params['plotAllLfpProxyPSD'] = {
    'num_col': 4,
    'tmin': 500.,    # in ms
    'fmax': 120.,    # in Hz
    # Welch segment length in samples. None -> full analysis window (finest
    # resolution the data allows, Δf = fs/N); set smaller (e.g. N // 4) to
    # trade resolution for a smoother, lower-variance estimate.
    'nperseg': None
    }

params['plotAreaLfpPsd'] = {
    'num_col': 6,
    'tmin': 500.,    # in ms
    'fmax': 300.,    # in Hz
    # See plotAllLfpProxyPSD['nperseg'].
    'nperseg': None
    }

params['calcAreaLfpCorrelation'] = {
    'tmin': 500.,    # in ms
    }

params['plotAreaLfpCorrelation'] = {
    'vmax': 1.0,             # diverging colorbar bound (r in [-1, 1])
    'cmap': 'RdBu_r',
    'cell_size_inches': 0.4  # inches per area cell; controls figure size
    }

params['calcAreaLfpCoherence'] = {
    'tmin': 500.,    # in ms
    'fmin': 0.01,    # in Hz
    'fmax': 80.,     # in Hz
    # Welch / CSD segment length in samples, the same for both. None -> 500 ms
    # segments: 2 Hz resolution at any sampling rate, so the band holds the
    # bins 2, 4, ..., 80 Hz. Longer segments resolve finer but average fewer
    # segments, which biases coherence upward.
    'nperseg': None
    }

params['plotAreaLfpCoherence'] = {
    'vmax': 1.0,             # colorbar upper bound (coherence is in [0, 1])
    'cmap': 'YlOrRd',
    'cell_size_inches': 0.4  # inches per area cell; controls figure size
    }

params['plotLfpArea'] = {
    'tmin': 500.    # in ms
    }

params['plotLfpPsdArea'] = {
    'tmin': 500.,    # in ms
    'fmax': 120.,    # in Hz
    # See plotAllLfpProxyPSD['nperseg'].
    'nperseg': None
    }

params['calcLfpCoherence'] = {
    'tmin': 500.,  # in ms
    'fmin': 0.01,    # in Hz
    'fmax': 80.,   # in Hz
    # See calcAreaLfpCoherence['nperseg'].
    'nperseg': None
    }

params['plotLayerLfpCoherence'] = {
    'vmax': 1.0,             # colorbar upper bound (coherence is in [0, 1])
    'cell_size_inches': 0.4  # inches per area cell; controls figure size
    }

params['plotFullLayerLfpCoherence'] = {
    'vmax': 1.0,             # colorbar upper bound
    'cell_size_inches': 0.18 # inches per (area, layer) cell; controls figure size
    }

params['calcLfpPlv'] = {
    'tmin': 500.,           # in ms
    # Cross-frequency θ–β phase locking: phase of the θ-filtered signal (rows)
    # vs phase of the β-filtered signal (columns), using n:m phase locking
    # PLV_nm(i, j) = |mean(exp(1j * (n*phi_theta_i - m*phi_beta_j)))|.
    'theta_band': (4., 8.),   # slow band [Hz] (rows / first phase)
    'beta_band':  (12., 30.), # fast band [Hz] (cols / second phase)
    'nm': (3, 1),           # (n_theta, m_beta): θ:β cycle ratio 1:3 → 3·φθ − 1·φβ (f_β≈3·f_θ); None = auto from band centres
    'n_max': 5,             # max harmonic order when auto-deriving n:m
    }

params['plotFullLayerLfpPlv'] = {
    'vmax': 1.0,             # colorbar upper bound (PLV is in [0, 1])
    'cell_size_inches': 0.4  # inches per area cell; controls figure size
    }

params['plotLfpWaveletArea'] = {
    'tmin': 500.,        # analysis start time in ms
    'fmin': 0.01,         # lower frequency bound in Hz
    'fmax': 120.,        # upper frequency bound in Hz
    'n_scales': 64,      # number of log-spaced frequency scales
    'w': 6.0,            # Morlet wavelet parameter
    'db_range': 60.0,    # dynamic range in dB (None = full range)
    'show_bands': True,  # dashed guide-lines at δ/θ/α/β/γ boundaries
    }

params['plotLfpSpectrogramArea'] = {
    'tmin': 500.,        # analysis start time in ms
    'fmin': 0.01,         # lower frequency bound in Hz
    'fmax': 120.,        # upper frequency bound in Hz
    'nperseg': 256,      # STFT window length in samples
    'db_range': 60.0,    # dynamic range in dB (None = full range)
    'show_bands': True,  # dashed guide-lines at δ/θ/α/β/γ boundaries
    }

params['plotConnectivities'] = {
    'tmin': 500.  # in ms
}

params['plotBOLD'] = {
    'tmin': 500.,   # in ms
    'stepSize': 1.  # in ms
}

params['plotRasterArea'] = {
    'fraction': 0.03,
    'low': 500,
    'high': 750
}

params['binned_spikerates_per_neuron'] = {
    'bins': 250,
    'num_col': 4
}

params['rate_distribution_similarity'] = {
    'remove_zeros': True,
}

params['plot_binned_spikerates_per_neuron'] = {
    'max_rate': 40.
}
