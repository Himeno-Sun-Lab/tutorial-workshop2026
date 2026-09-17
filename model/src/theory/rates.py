import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from nnmt.lif.exp import (_firing_rate_shift, _transfer_function,
                          _effective_connectivity)
from nnmt.lif._general import _mean_input, _std_input
from nnmt.network_properties import _delay_dist_matrix


def initial_rates_uniform(net_dict, trials, min, max, seed=None):
    """
    Draw uniformly distributed initial rates between min and max.
    Supports multiple trials.
    """
    # remove empty popultions
    N = net_dict['neuron_numbers'].sort_index()
    mask = N > 0
    N = N.loc[mask]
    # create uniform initial rates
    initial_rates = {}
    initial_rate_params = dict(size=N.shape[0], low=min, high=max)
    np.random.seed(seed)
    for trial in range(trials):
        initial_rates[trial] = np.random.uniform(**initial_rate_params)
    return pd.DataFrame(initial_rates, index=N.index)


def _build_inputs(net_dict):
    """
    Assemble the nnmt mean-field input parameters from a network dict.

    Returns a dict with the (empty-population-masked) index ``N``, the input
    parameter dicts ``input_prms`` / ``input_prms_scld`` (as consumed by
    ``nnmt.lif._general._mean_input`` / ``_std_input``), and the per-population
    neuron parameters ``tau_m``, ``tau_syn``, ``t_ref``, ``V_th``, ``V_r`` (mV,
    relative to the leak potential). Shared by ``solve`` and ``working_point``.
    """
    # collect network parameters, make sure everything is sorted the same way
    N = net_dict['neuron_numbers'].sort_index()
    nu_ext = net_dict['rate_ext'].sort_index()  # 1/s
    K = net_dict['synapses_internal'].sort_index(axis=0).sort_index(axis=1)
    K = K.div(N, axis=0)
    K_ext = net_dict['synapses_external'].sort_index()
    K_ext = K_ext.div(N)
    J = net_dict['weights'].sort_index(axis=0).sort_index(axis=1)
    J_ext = net_dict['weights_ext'].sort_index()

    # remove empty populations
    mask = N > 0
    N = N.loc[mask]
    nu_ext = nu_ext.loc[mask]
    K = K.loc[mask, mask]
    K_ext = K_ext.loc[mask]
    J = J.loc[mask, mask]
    J_ext = J_ext.loc[mask]

    # collect neuron parameters
    lif_params_E = net_dict['neuron_params_E']
    lif_params_I = net_dict['neuron_params_I']
    ix_E = (slice(None), slice(None), 'E')
    ix_I = (slice(None), slice(None), 'I')
    tau_m = pd.Series(data=0., index=N.index)
    tau_m.loc[ix_E] = 1e-3*lif_params_E['tau_m']  # s
    tau_m.loc[ix_I] = 1e-3*lif_params_I['tau_m']  # s
    tau_syn = pd.DataFrame(data=0., index=N.index, columns=N.index)
    tau_syn.loc[ix_E, ix_E] = 1e-3*lif_params_E['tau_syn_ex']  # s
    tau_syn.loc[ix_E, ix_I] = 1e-3*lif_params_E['tau_syn_in']  # s
    tau_syn.loc[ix_I, ix_E] = 1e-3*lif_params_I['tau_syn_ex']  # s
    tau_syn.loc[ix_I, ix_I] = 1e-3*lif_params_I['tau_syn_in']  # s
    tau_syn_ext = pd.Series(data=0., index=N.index)
    tau_syn_ext.loc[ix_E] = 1e-3*lif_params_E['tau_syn_ex']  # s
    tau_syn_ext.loc[ix_I] = 1e-3*lif_params_I['tau_syn_ex']  # s
    t_ref = pd.Series(data=0., index=N.index)
    t_ref.loc[ix_E] = 1e-3*lif_params_E['t_ref']  # s
    t_ref.loc[ix_I] = 1e-3*lif_params_I['t_ref']  # s
    C_m = pd.DataFrame(data=0., index=N.index, columns=N.index)
    C_m.loc[ix_E, :] = lif_params_E['C_m']  # pF
    C_m.loc[ix_I, :] = lif_params_I['C_m']  # pF
    C_m_ext = pd.Series(data=0., index=N.index)
    C_m_ext.loc[ix_E] = lif_params_E['C_m']  # pF
    C_m_ext.loc[ix_I] = lif_params_I['C_m']  # pF
    V_th = pd.Series(data=0., index=N.index)
    V_th.loc[ix_E] = lif_params_E['V_th'] - lif_params_E['E_L']  # mV
    V_th.loc[ix_I] = lif_params_I['V_th'] - lif_params_I['E_L']  # mV
    V_r = pd.Series(data=0., index=N.index)
    V_r.loc[ix_E] = lif_params_E['V_reset'] - lif_params_E['E_L']  # mV
    V_r.loc[ix_I] = lif_params_I['V_reset'] - lif_params_I['E_L']  # mV

    # scale weights
    J *= 1e3*tau_syn/C_m  # mV
    J_ext *= 1e3*tau_syn_ext/C_m_ext  # mV

    input_prms = dict(tau_m=tau_m, J=J, K=K, J_ext=np.diag(J_ext),
                      K_ext=np.diag(K_ext), nu_ext=nu_ext)
    input_prms_scld = dict(tau_m=tau_m, J=J, K=K/tau_syn, J_ext=np.diag(J_ext),
                           K_ext=np.diag(K_ext/tau_syn_ext), nu_ext=nu_ext)
    return dict(N=N, input_prms=input_prms, input_prms_scld=input_prms_scld,
                tau_m=tau_m, tau_syn=tau_syn, t_ref=t_ref, V_th=V_th, V_r=V_r)


def solve(net_dict, t, initial_rates):
    r"""
    Solves the auxiliary ODE system
      \dot{\nu}_i = -\nu_i + \phi(\mu_i, \sigma_i)
    with initial condition initial_rates at the times t.
    mu and sigma are determined by the recurrent input, hence a fixed point
    of the dynamics corresponds to a self-consistent solution.
    """
    p = _build_inputs(net_dict)
    N, input_prms, input_prms_scld = p['N'], p['input_prms'], p['input_prms_scld']
    tau_m, t_ref, V_th, V_r, tau_syn = (p['tau_m'], p['t_ref'], p['V_th'],
                                        p['V_r'], p['tau_syn'])

    #  ODE for (pseudotime) fixed point iteration
    def aux_ode(_, nu_in):
        mu = _mean_input(nu=nu_in, **input_prms)
        sig = _std_input(nu=nu_in, **input_prms)
        tau_eff = sig**2 / _std_input(nu=nu_in, **input_prms_scld)**2
        nu_out = _firing_rate_shift(V_0_rel=V_r, V_th_rel=V_th,
                                    mu=mu, sigma=sig,
                                    tau_m=tau_m, tau_r=t_ref, tau_s=tau_eff)
        return -nu_in + nu_out

    rates = {}
    for trial in initial_rates:
        sol_ivp = solve_ivp(fun=aux_ode, t_span=(t[0], t[-1]),
                            y0=initial_rates[trial], method='LSODA', t_eval=t)
        assert sol_ivp.success
        for i, (area, layer, pop) in enumerate(N.index):
            rates[(area, layer, pop, trial)] = sol_ivp.y[i, :]
    return pd.DataFrame(rates)


def working_point(net_dict, rates):
    r"""
    Mean-field working point of every population at the given firing rates.

    Computes the mean input ``mu`` and its standard deviation ``sigma`` (mV,
    relative to the leak potential) via nnmt, and the distance of the mean input
    from threshold in units of the fluctuation:

        working_point(p) = (V_th - mu) / sigma.

    Interpretation: ``> ~1`` the mean input sits well below threshold, spikes
    are fluctuation-driven ⇒ asynchronous-irregular; ``<= 0`` the mean input is
    at/above threshold, neurons are mean-driven ⇒ regular firing / synchronous-
    regular. Uses the network structure, neuron parameters and (given) rates.

    Parameters
    ----------
    net_dict : dict
        Network dict from ``networkDictFromDump`` (needs synapse counts,
        weights, neuron numbers, external drive and ``neuron_params_E/I``).
    rates : pd.Series
        Firing rate (1/s) per (area, layer, population); typically the simulated
        rates, or a mean-field self-consistent solution. Missing populations are
        treated as silent.

    Returns
    -------
    pd.DataFrame
        Columns ``mu``, ``sigma``, ``working_point`` indexed by the non-empty
        (area, layer, population) targets.
    """
    p = _build_inputs(net_dict)
    N, input_prms = p['N'], p['input_prms']
    V_th, V_r = p['V_th'], p['V_r']

    nu = rates.copy()
    nu.index = nu.index.set_names(N.index.names)
    nu = nu.reindex(N.index).astype(float).fillna(0.0)

    mu = pd.Series(_mean_input(nu=nu, **input_prms), index=N.index)
    sigma = pd.Series(_std_input(nu=nu, **input_prms), index=N.index)
    wp = (V_th - mu) / sigma.replace(0, np.nan)
    return pd.DataFrame({'mu': mu, 'sigma': sigma, 'working_point': wp})


def _population_delay_matrix(net_dict, index):
    """
    Population-resolved delay mean/sd (seconds) for the mean-field analysis.

    Inter-area entries use the area-level cortico-cortical delays
    (``delay_cc``); intra-area entries use the local dendritic delays
    (``delay_e``/``delay_i`` by presynaptic type). The sd is ``delay_rel`` times
    the mean (as in the network's delay model).
    """
    dp = net_dict.get('network_params', {}).get('delay_params') \
        or net_dict['delay_params']
    delay_cc = net_dict['delay_cc']                       # area x area, ms
    areas = index.get_level_values('area')
    pops = index.get_level_values('population')
    d_area = delay_cc.reindex(index=areas, columns=areas).values     # ms
    same = areas.values[:, None] == areas.values[None, :]
    local = np.where(pops.values[None, :] == 'E', dp['delay_e'], dp['delay_i'])
    local = np.broadcast_to(local, d_area.shape)
    delay = np.where(same, local, d_area) * 1e-3          # s
    return delay, dp['delay_rel'] * delay


def oscillation_spectrum(net_dict, rates, freqs=None,
                         delay_dist='truncated_gaussian'):
    r"""
    Eigenvalue spectrum of the effective connectivity vs frequency (Tier-2).

    Builds the effective connectivity ``M_eff(w) = tau_m * J * K * H(w) * D(w)``
    at the mean-field working point (:cite:t:`bos2016`, Eq. 12), where ``H`` is
    the LIF transfer function evaluated at the input ``mu``/``sigma`` implied by
    ``rates`` and ``D`` the delay-distribution matrix, then returns its
    eigenvalues at every analysis frequency.

    The asynchronous state loses linear stability when an eigenvalue reaches the
    point ``1 + 0j``: ``max_w max_i Re(lambda_i(w)) >= 1`` ⇒ unstable, with the
    peak frequency giving the emerging oscillation. The distance
    ``margin = 1 - max Re(lambda)`` is a stability margin (``<= 0`` unstable).

    Parameters
    ----------
    net_dict : dict
        Network dict (see ``working_point``); additionally needs ``delay_cc``
        and the ``delay_params``.
    rates : pd.Series
        Firing rate (1/s) per population defining the working point.
    freqs : np.ndarray, optional
        Analysis frequencies in Hz (default ``linspace(1, 300, 300)``).
    delay_dist : str
        Delay distribution for ``nnmt`` (default ``'truncated_gaussian'``).

    Returns
    -------
    dict with keys
        ``freqs``          analysis frequencies (Hz);
        ``eigenvalues``    complex array (len(freqs), n_pop);
        ``max_real``       max Re(eigenvalue) per frequency;
        ``critical_freq``  frequency of the overall max Re;
        ``peak``           the overall max Re(eigenvalue);
        ``margin``         1 - peak  (<= 0 ⇒ asynchronous state unstable).
    """
    if freqs is None:
        freqs = np.linspace(1., 300., 300)
    p = _build_inputs(net_dict)
    N, input_prms = p['N'], p['input_prms']

    nu = rates.copy()
    nu.index = nu.index.set_names(N.index.names)
    nu = nu.reindex(N.index).astype(float).fillna(0.0).values

    # working point in SI units (V, s); tau_m and tau_s are uniform ⇒ scalars
    mu_V = _mean_input(nu=nu, **input_prms) * 1e-3
    sig_V = _std_input(nu=nu, **input_prms) * 1e-3
    V_th_V, V_0_V = p['V_th'].values * 1e-3, p['V_r'].values * 1e-3
    J_V = input_prms['J'].values * 1e-3
    K = input_prms['K'].values
    tau_m = float(p['tau_m'].iloc[0])
    tau_s = float(p['tau_syn'].values.max())
    t_ref = p['t_ref'].values

    omegas = 2 * np.pi * freqs
    tf = _transfer_function(mu_V, sig_V, tau_m, tau_s, t_ref, V_th_V, V_0_V,
                            omegas)
    delay, delay_sd = _population_delay_matrix(net_dict, N.index)
    D = _delay_dist_matrix(delay, delay_sd, delay_dist, omegas)
    # The delay distribution is singular (NaN) for zero-delay entries; those are
    # unconnected area pairs (K = 0) so they contribute nothing — zero them so
    # the 0 * NaN does not poison the matrix.
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
    M_eff = _effective_connectivity(tf, D, J_V, K, tau_m)

    # A pathological population (e.g. a non-finite transfer function) would make
    # its whole row non-finite; drop such entries (treated as decoupled) so the
    # eigenvalue solve stays well-posed, and report how many populations.
    n_nonfinite = int(np.unique(np.nonzero(~np.isfinite(M_eff))[1]).size)
    M_eff = np.nan_to_num(M_eff, nan=0.0, posinf=0.0, neginf=0.0)

    eig = np.array([np.linalg.eigvals(M_eff[i]) for i in range(len(freqs))])
    max_real = eig.real.max(axis=1)
    ic = int(np.argmax(max_real))
    peak = float(max_real[ic])
    return dict(freqs=freqs, eigenvalues=eig, max_real=max_real,
                critical_freq=float(freqs[ic]), peak=peak, margin=1.0 - peak,
                n_nonfinite=n_nonfinite)
