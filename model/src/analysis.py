import os
import random
from ast import literal_eval
from datetime import datetime
import glob
import math
import time
import subprocess

import yaml
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.stats import ks_2samp
from scipy.signal import convolve, hilbert, welch, csd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from dicthash import dicthash

from helpers.resting_state_networks import left_ordering
from helpers.abeta import resolve_abeta_ratio, abeta_ratio_is_zero

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Pool

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        passed_time = round(te - ts, 3)
        msg = f'{method.__name__} took {passed_time} s'
        self = args[0] if args else None
        if self is not None and hasattr(self, 'logger') and self.logger:
            self.logger.debug(msg)
        else:
            print(msg)
        return result
    return timed


class Analysis():
    """
    Class that provides functionality to analyze simulation results.
    """

    def __init__(self, ana_params, net_dict, sim_dict, sim_folder,
                base_path, logger=None):
        self.ana_dict = ana_params
        self.net_dict = net_dict
        self.sim_dict = sim_dict
        self.base_path = base_path
        self.sim_folder = sim_folder
        self.ana_folder = os.path.join(sim_folder, self.getHash())
        self.plot_folder = os.path.join(self.ana_folder, 'plots')
        self.logger = logger

    def setup(self):
        random.seed(self.ana_dict['seed'])

        if not os.path.isdir(self.ana_folder):
            os.mkdir(self.ana_folder)
        if not os.path.isdir(self.plot_folder):
            os.mkdir(self.plot_folder)
        self.logger.info('Results will be written to %s', self.plot_folder)
        self.logger.info('Plots will be written to %s', self.plot_folder)

        self.logger.info('Reading popGids')
        self.popGids = self._readPopGids()
        self.logger.info('Reading spikes')
        self.spikes = self._readSpikes()
        # if self.net_dict["Abeta"]:
        #     self.logger.info('Reading Abeta-popGids')
        #     self.AbetaGids = self._readAbetaPopGids()
        if "I_syn" in self.sim_dict["rec_dev"]:
            self.logger.info('Reading post-synaptic currents')
            self.I_syn = self._readIsyn()

    def _category_plot_folder(self, category):
        """Return (and create if needed) a per-category subdirectory of plot_folder."""
        path = os.path.join(self.plot_folder, category)
        os.makedirs(path, exist_ok=True)
        return path

    def _abeta_active(self):
        """
        Whether the network actually contains amyloid-affected neurons.

        True only if the ``Abeta`` flag is set *and* at least one population has
        a non-zero Abeta ratio. When ``Abeta`` is enabled but every ratio
        resolves to 0 (scalar 0, an all-zero dict, or both E and I disabled) no
        Abeta neurons were created, so the analysis must behave as the plain
        (non-Abeta) case — otherwise it would read an empty Abeta GID file or
        blend a redundant kernel.
        """
        if not self.net_dict.get('Abeta', False):
            return False
        return not (
            abeta_ratio_is_zero(self.net_dict.get('Abeta_ratio_E', 0.0))
            and abeta_ratio_is_zero(self.net_dict.get('Abeta_ratio_I', 0.0))
        )

    @timeit
    def fullAnalysis(self):
        """
        Execute the full analysis.
        """
        self.logger.info('Calculating rate')
        self.rate = self.meanFiringRate()
        self.logger.info('Calculating population CV ISI')
        self.pop_cv_isi = self.popCvIsi()
        self.logger.info('Calculating population LV ISI')
        self.pop_lv = self.popLv()
        self.logger.info('Calculating correlation coefficients')
        self.pop_cc = self.popCorrCoeff()
        self.logger.info('Calculating rate histogram')
        self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        self.logger.info('Calculating synaptic input currents')
        self.curr_in = self.synapticInputCurrent()
        if "I_syn" in self.sim_dict["rec_dev"]:
            self.logger.info('Calculating LFP proxy (NEST)')
            self.lfp = self.calc_lfp_proxy()
        else:
            self.logger.info('Calculating LFP proxy (Kernel, Fast=%s)',
                             self.ana_dict.get('fast_lfp_kernel', False))
            self.lfp = self.calc_lfp_proxy_kernel()
        self.logger.info('Calculating LFP coherence')
        self.calcLfpCoherence()
        self.logger.info('Calculating LFP PLV')
        self.calcLfpPlv()
        self.logger.info('Calculating ROI (area-level) LFP proxy')
        self.calcAreaLfp()
        self.logger.info('Calculating ROI LFP correlation (functional connectivity)')
        self.calcAreaLfpCorrelation()
        self.logger.info('Calculating ROI LFP coherence')
        self.calcAreaLfpCoherence()

        self.logger.info('Calculating rates of all neurons')
        self.individual_rates = self.individualFiringRate()
        self.logger.info('Calculating binned spike rates per neuron')
        self.spikes_per_neuron_population_resolved, self.spikes_per_neuron_area_resolved = self.binned_spikerates_per_neuron()

        self.logger.info('Calculating BOLD')
        self.BOLD = self.computeBOLD()
        self.logger.info('Calculating BOLD connectivity')
        self.BOLD_correlation = self.calculateBOLDConnectivity()
        self.logger.info('Calculating connectivity correlations')
        self.calculateFuncionalConnectivityCorrelations()
        self.calculateRateDistributionSimilarity()

        self.logger.info('Plotting Boxplot')
        self.plotBoxPlot()
        self.logger.info('Plotting Cv Isi')
        self.plotPopCVIsi()
        self.logger.info('Plotting Lv Isi')
        self.plotPopLV()
        self.logger.info('Plotting Correlation coefficients')
        self.plotCorrCoff()
        self.logger.info('Plotting binned spike rates per neuron')
        self.plot_all_binned_spike_rates_area()

        self.logger.info('Plotting functional connectivity based on synaptic input currents')
        # The experimental reference (group-averaged vs. single subject) is
        # picked automatically from ana_dict['experimental_fc'].
        self.plot_functional_connectivity(save_fig=True)

        self.logger.info('Plotting area averaged spike rates')
        self.plotAllFiringRatesSummary()
        self.logger.info('Plotting Synaptic currents')
        self.plotAllSynapticCurrentsSummary()
        self.logger.info('Plotting BOLD signal')
        self.plotAllBOLDSignalSummary()
        self.logger.info('Plotting LFP proxy PSD summary')
        self.plotAllLfpProxyPSD()
        self.logger.info('Plotting ROI LFP proxy PSD')
        self.plotAreaLfpPsd()
        self.logger.info('Plotting ROI LFP correlation matrix')
        self.plotAreaLfpCorrelation()
        self.logger.info('Plotting ROI LFP coherence matrix')
        self.plotAreaLfpCoherence()
        self.logger.info('Plotting layer LFP coherence matrices')
        self.plotLayerLfpCoherence()
        self.logger.info('Plotting full-layer LFP coherence matrix')
        self.plotFullLayerLfpCoherence()
        self.logger.info('Plotting full-layer LFP PLV matrix')
        self.plotFullLayerLfpPlv()
        for area in self.popGids.index.levels[0]:
            self.logger.info('Plotting %s', area)
            self.plotRasterArea(area)
            if self._abeta_active():
                self.logger.info('Plotting Abeta-affected nodes %s', area)
                self.plotRasterAbetaArea(area)
            self.logger.info('Plotting %s LFP proxy', area)
            self.plotLfpArea(area)
            self.logger.info('Plotting %s LFP PSD', area)
            self.plotLfpPsdArea(area)
            self.logger.info('Plotting %s LFP spectrogram', area)
            self.plotLfpSpectrogramArea(area)
            self.logger.info('Plotting %s LFP wavelet scalogram', area)
            self.plotLfpWaveletArea(area)
        self.logger.info('Plotting BOLD connectivities')
        self.plotBOLDConnectivity()
        self.logger.info('Plotting Raster statistics')
        self.plot_raster_statistics(save_fig=True)
        plt.close('all')
        self.logger.info('Analysis finished')


    @timeit
    def meanFiringRate(self):
        """
        Calculates the population averaged firing rate.

        Returns
        -------
        rate : Series
        """
        try:
            rate = pd.read_pickle(os.path.join(self.sim_folder, 'rates.pkl'))
        except FileNotFoundError:
            # Start, End and Simulation time
            t_start = self.ana_dict['meanFiringRate']['t_start']  # in ms
            t_stop = self.ana_dict['meanFiringRate']['t_stop']  # in ms
            if t_start is None:
                t_start = 0.
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']
            t_sim_sec = (t_stop - t_start) / 1000.  # Convert ms to s
            # For all populations we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # per population by the number of neurons in this populations and
            # the simulation time.
            rate = self.spikes.apply(
                    lambda sts: sum([st[(st >= t_start) & (st < t_stop)].size for st in sts])
                    ).div(self.popGids['pop_size']) / t_sim_sec
            rate.to_pickle(os.path.join(self.sim_folder, 'rates.pkl'))
        return rate

    @timeit
    def individualFiringRate(self):
        """
        Calculates the firing rate of individual neurons.

        Returns
        -------
        rate : Series
        """
        try:
            rate = pd.read_pickle(os.path.join(self.sim_folder, 'rates_individual.pkl'))
        except FileNotFoundError:
            # Start, End and Simulation time
            t_start = self.ana_dict['individualFiringRate']['t_start']  # in ms
            t_stop = self.ana_dict['individualFiringRate']['t_stop']  # in ms
            if t_start is None:
                t_start = 0.
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']
            t_sim_sec = (t_stop - t_start) / 1000.  # Convert ms to s
            # For all neurons we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # by the simulation time.
            rate = self.spikes.apply(
                    lambda sts: np.array([st[(st >= t_start) & (st < t_stop)].size for st in sts])
                    ) / t_sim_sec
            rate.to_pickle(os.path.join(self.sim_folder, 'rates_individual.pkl'))
        return rate

    @timeit
    def binned_spikerates_per_neuron(self):
        """
        Calculates the distribution of firing rates area and population
        resolved for the complete network.

        Returns
        -------
        spikes_per_neuron_population_resolved : Series
        spikes_per_neuron_area_resolved : Series
        """
        try:
            spikes_per_neuron_population_resolved = pd.read_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_population_resolved.pkl'))
            spikes_per_neuron_area_resolved = pd.read_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_area_resolved.pkl'))
        except FileNotFoundError:
            # Simulation time in seconds
            bins = self.ana_dict['binned_spikerates_per_neuron']['bins']
            # For all populations we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # per population by the number of neurons in this populations and
            # the simulation time.
            spikes_per_neuron_population_resolved_tmp = self.spikes.apply(
                    lambda tmp: np.histogram(np.array([len(i) for i in tmp]), bins=bins)
                    )
            spikes_per_neuron_area_resolved_tmp = self.spikes.apply(list).groupby('area').agg(sum).apply(
                    lambda tmp: np.histogram(np.array([len(i) for i in tmp]), bins=bins)
                    )

            tmp = {}
            for area, (hist, bin_edges) in spikes_per_neuron_population_resolved_tmp.items():
                tmp[area + tuple(['hist'])] = hist
                tmp[area + tuple(['bin_edges'])] = bin_edges
            spikes_per_neuron_population_resolved = pd.Series(tmp)

            tmp = {}
            for area, (hist, bin_edges) in spikes_per_neuron_area_resolved_tmp.items():
                tmp[(area, 'hist')] = hist
                tmp[(area, 'bin_edges')] = bin_edges
            spikes_per_neuron_area_resolved = pd.Series(tmp)

            spikes_per_neuron_population_resolved.to_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_population_resolved.pkl'))
            spikes_per_neuron_area_resolved.to_pickle(os.path.join(self.sim_folder, 'spikes_per_neuron_area_resolved.pkl'))
        return spikes_per_neuron_population_resolved, spikes_per_neuron_area_resolved

    @timeit
    def firingRateHistogram(self):
        """
        Calculates the time-resolved population averaged firing rate.
        Uses np.histogram with a fixed binsize.

        Returns
        -------
        rate_hist : Series of numpy arrays containing time resolved firing
                    rates
        """
        try:
            rate_hist = pd.read_pickle(os.path.join(
                self.ana_folder, 'rate_histogram.pkl'
            ))
            rate_hist_areas = pd.read_pickle(os.path.join(
                self.ana_folder, 'rate_histogram_areas.pkl'
            ))
        except FileNotFoundError:
            rate_hist = self.spikes.apply(
                    calc_rates,
                    args=(self.sim_dict, self.ana_dict,)
                    ).div(self.popGids['pop_size'])

            rate_hist_areas = self.spikes.apply(list).groupby('area').agg(
                    sum
                    ).apply(
                        calc_rates,
                        args=(self.sim_dict, self.ana_dict)
                    ).div(self.popGids.groupby('area').agg(sum)['pop_size'])

            rate_hist.to_pickle(os.path.join(
                self.ana_folder, 'rate_histogram.pkl'
            ))

            rate_hist_areas.to_pickle(os.path.join(
                self.ana_folder, 'rate_histogram_areas.pkl'
            ))
        return rate_hist, rate_hist_areas

    @timeit
    def plot_instantaneous_firing_rate(self, save_fig=False):
        """
        Plots the instantaneous firing rate over simulated areas using a heatmap.
        
        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        """
        if not hasattr(self, 'rate_hist_areas'):
            self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        
        # Convert rate_hist_areas to spikes/s
        rate_hist_areas = self.rate_hist_areas * 1000
        
        # Convert rate_hist_areas to a DataFrame for easier plotting
        rate_hist_areas_df = pd.DataFrame(rate_hist_areas.tolist(), index=rate_hist_areas.index)

        # Plot the heatmap with an orange-yellow color palette
        plt.style.use('default')
        plt.figure(figsize=(12, 5))
        sns.heatmap(rate_hist_areas_df, cmap='YlOrBr', cbar_kws={'label': 'Spikes/s'}, yticklabels=rate_hist_areas_df.index)
        plt.xlabel('Time (ms)')
        plt.ylabel('Area')
        plt.title('Instantaneous firing rate over simulated areas')
        plt.xticks(rotation=0)  # Rotate x-axis labels to make the times horizontal
        plt.xlim(self.sim_dict['t_sim']-500, self.sim_dict['t_sim'])
        
        # Save the plot if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            plt.savefig(os.path.join(self.plot_folder, f'instantaneous_firing_rate.{extension}'),
                        dpi=self.ana_dict['dpi'])
        plt.show()

    @timeit
    def plot_average_rate_per_pop(self, save_fig=False):
        """
        Plots the time-averaged firing rate over simulated populations using a heatmap.
        
        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        """

        # Calculate the time-averaged firing rate if it has not been calculated yet
        if not hasattr(self, 'self.rate'):
            self.rate = self.meanFiringRate()
        mean_rates_per_pop = self.rate

        # Pivot the DataFrame to have areas on the x-axis and layer+pop on the y-axis
        mean_rates_df = mean_rates_per_pop.reset_index().pivot(index=['layer', 'pop'], columns='area', values=0)

        # Create a new index combining layer and pop with layer names converted from Roman to Arabic numerals
        roman_to_arabic = {
            'II/III': '2/3',
            'IV': '4',
            'V': '5',
            'VI': '6',
        }
        mean_rates_df.index = mean_rates_df.index.map(lambda x: f"{roman_to_arabic.get(x[0], x[0])} {x[1]}")

        # Create a mask for NaN values
        mask = mean_rates_df.isna()

        # Plot the heatmap with external grid
        plt.style.use('default')
        plt.figure(figsize=(12, 4.5))
        sns.heatmap(mean_rates_df, cmap='YlOrBr', fmt=".2f", mask=mask, cbar_kws={'label': 'Spikes/s'})
        plt.title('Time-averaged firing rate over simulated populations')
        plt.xlabel('Area')
        plt.ylabel('Population')

        # Rotate x-tick and y-tick labels
        plt.yticks(rotation=0)

        # Plot X marks for NaN values
        for i in range(mean_rates_df.shape[0]):
            for j in range(mean_rates_df.shape[1]):
                if mask.iloc[i, j]:
                    plt.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', color='black', fontsize=12, weight='bold')

        plt.tight_layout()
        
        # Save the plot if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            plt.savefig(os.path.join(self.plot_folder, f'average_rate_per_area.{extension}'),
                        dpi=self.ana_dict['dpi'])

        # Calculate and print the total mean firing rate
        total_mean_rate = mean_rates_per_pop.mean()
        if self.logger:
            self.logger.info('Total mean firing rate: %.2f spikes/s', total_mean_rate)        
        plt.show()

    @timeit
    def synapticInputCurrent(self):
        """
        Calculates the area averaged synaptic input current.
        Compare MAM dynamics paper, page14.

        Returns
        -------
        current : DataFrame (index=index, columns=t)
        """
        try:
            curr = pd.read_pickle(os.path.join(
                self.ana_folder, 'input_current.pkl'
            ))
        except FileNotFoundError:
            # Get parameters
            if not hasattr(self, 'rate_hist'):
                self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
            inst_rates = self.rate_hist
            index = inst_rates.index
            NN = self.net_dict['neuron_numbers'].loc[index]
            weights = self.net_dict['weights'].loc[index, index]
            synapses = self.net_dict['synapses_internal'].loc[index, index]
            indeg = synapses.div(NN, axis=0)
            tau_s_ex_E = self.net_dict['neuron_params_E']['tau_syn_ex']
            tau_s_in_E = self.net_dict['neuron_params_E']['tau_syn_in']
            tau_s_ex_I = self.net_dict['neuron_params_I']['tau_syn_ex']
            tau_s_in_I = self.net_dict['neuron_params_I']['tau_syn_in']
            assert (tau_s_ex_E == tau_s_ex_I) and (tau_s_in_E == tau_s_in_I)
            tau_syn_ex = tau_s_ex_E
            tau_syn_in = tau_s_in_E
            dt = self.sim_dict['sim_resolution']
            binsize = self.ana_dict['rate_histogram']['binsize']
            t_start = 0.
            t_stop = self.ana_dict['rate_histogram']['t_stop']
            if t_stop is None:
                t_stop = self.sim_dict['t_sim']

            kernel_syn_in = kernel_for_psc(tau_syn_in, dt)
            kernel_syn_ex = kernel_for_psc(tau_syn_ex, dt)

            t_in = np.arange(t_start, t_stop, binsize)

            # Apply convolution to all firing rates emitted from a particular
            # population. Depending on whether the population is excitatory or
            # inhibitory different kernels with different synaptic time
            # constants are used. The results are the postynaptic currents
            # originating in a particular population.
            tmp = {}
            for (area, layer, population), dat in inst_rates.items():
                if population == 'E':
                    tau_s = tau_syn_ex
                    kernel = kernel_syn_ex
                else:
                    tau_s = tau_syn_in
                    kernel = kernel_syn_in
                tmp[(area, layer, population)] = convolve(tau_s*dat, kernel,
                                                          mode='same')
            pscs = pd.DataFrame(tmp, index=t_in).T

            # Include weights, indegrees and average
            curr = (weights.abs() * indeg).dot(pscs)
            curr = curr.mul(NN, axis=0).groupby(level=0).sum()
            curr = curr.div(NN.groupby(level=0).sum(), axis=0)
            curr.to_pickle(os.path.join(
                self.ana_folder, 'input_current.pkl'
            ))

        return curr

    @timeit
    def computeBOLD(self):
        # TODO
        # Use rate ( note it's given in spikes/ms, i.e. multiply by 1000)
        # instead of current. This is in agreement with Gustavo Deco and Viktor
        # Jirsa 2012: Ongoing Cortical Activity at Rest: Criticality,
        # Multistability,and Ghost Attractors
        # and Deco 2019: Brain songs framework used for discovering the
        # relevant timescale of the human brain
        # Kringelbach et al. Dynamic coupling of whole-brain neuronal
        # andneurotransmitter systems
        # It seems, at least in deco 2019 and kringelbach 2020, they only used
        # excitatory neurons
        # Take parameters from Stephan 2017
        try:
            BOLDSIGNAL = pd.read_pickle(os.path.join(
                self.ana_folder, 'bold_signal.pkl'
            ))
        except FileNotFoundError:
            if not hasattr(self, 'rate_hist_areas'):
                _, self.rate_hist_areas = self.firingRateHistogram()
            rate_hist_areas = self.rate_hist_areas * 1000.  # convert
                                                            # spikes / [ms] to
                                                            # spikes / [s]

            def calculate_BOLD(rate, area=None):
                # Assuming the rate is binned in [ms]
                t_vals = np.arange(0, len(rate)) / 1000.
                # Interpolate the rate in order to access values which do
                # not lie on the grid. The ode solver might access values
                # which are slightly out of the bounds of the interpolation
                # range, thus extrapolate those values.
                z_val = interp1d(
                        t_vals,
                        rate,
                        bounds_error=False,
                        fill_value='extrapolate'
                        )

                def E(f_in, rho):
                    """
                    fraction of oxygen extracted from the inflowing blood

                    Buxton et al. 1998
                    """
                    return 1 - np.power(1 - rho, 1/f_in)

                def balloon_windkessel(t, w, z):
                    """
                    f_in : inflow from the venouscompartment
                    v : Volume

                    z : neuronal activity

                    TVB implementations:
                    https://github.com/the-virtual-brain/tvb-hpc/blob/master/tvb_hpc/bold.py
                    https://github.com/the-virtual-brain/tvb-root/blob/bb3d3c91fb2ba20273b9a065943a141002b07229/scientific_library/tvb/analyzers/fmri_balloon.py
                    neurolib implementation:
                    https://github.com/neurolib-dev/neurolib/blob/27ea47aa33d83d080952954600380d33b6e38d34/neurolib/models/bold/timeIntegration.py
                    WholeBrain implementation:
                    https://github.com/dagush/WholeBrain/blob/faf3a557410c2f7a5942c0298c7ae332630d3eb5/functions/BOLDHemModel_Friston2003.py
                    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2008.py
                    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2007.py
                    Deco lab
                    https://github.com/decolab/cb-neuromod/blob/master/functions/BOLD.m

                    Important papers:
                    * K.J. Friston, L. Harrison, and W. Penny,
                      Dynamic causal modelling, NeuroImage 19 (2003) 1273-1302
                    * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
                      Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387-401
                   * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
                     Dynamic causal modelling revisited, NeuroImage 199 (2019) 730-744
                    """
                    s, f_in, v, q = w
                    eps = 1.  # Friston 2003; eps = 0.5  Friston 2000
                    kappa = .65  # time-constant for signal decay or
                                 # elimination [1/s], Friston 2003,
                                 # kappa = 1 / tau_s, tau_s = 0.8 in Friston
                                 # 2000
                                 # For some reason some other code sources use
                                 # this value reversed, is this a bug on their
                                 # or our side? cf tvb, wholebrain, deco
                    gamma = .41  # time-constant for autoregulatory feedback
                                 # from blood flow [1/s], Friston 2003,
                                 # gamma = 1 / tau_f
                                 # tau_f = 0.4 in Friston 2000
                                 # For some reason some other code sources use
                                 # this value reversed, is this a bug on their
                                 # or our side? cf tvb, wholebrain, deco
                    tau = .98  # Mean transit time [s], time to traverse the
                               # venous compartment
                    alpha = .32  # Grubb's exponent, Grubb et al 1974
                    rho = .34  # oxygen extraction fraction, sometimes called
                               # E0

                    f_out = np.power(v, 1 / alpha)  # Outflow

                    # ========================================================
                    #  For a description of these equations see Friston 2003
                    # ========================================================

                    # Change of vasodilatory signal s, dependent on neuronal
                    # activity z, which is subject to autoregulatory feedback
                    # Vasodilation is the widening of blood vessels
                    dsdt = eps * z(t) - kappa * s - gamma * (f_in - 1)
                    # assumption of dynamical system linking synaptic activity
                    # and rCBF (regional cerbral blood flow) is linear
                    dfdt = s
                    # Rate of change of volume v
                    dvdt = (f_in - f_out) / tau
                    # change in deoxyhemoglobin q, delivery into venous
                    # compartment minus expelled
                    dqdt = (f_in * E(f_in, rho) / rho - f_out * q / v) / tau

                    w_return = [dsdt, dfdt, dvdt, dqdt]
                    return w_return

                s0 = 0.
                f0 = 1.
                v0 = 1.
                q0 = 1.

                w0 = [s0, f0, v0, q0]

                balloon_windkessel_solution = solve_ivp(
                    lambda t, w: balloon_windkessel(t, w, z_val),
                    (t_vals[0], t_vals[-1]),
                    w0
                    )

                _, _, v, q = balloon_windkessel_solution.y
                time = balloon_windkessel_solution.t

                rho = .34  # oxygen extraction fraction, sometimes called E0
                V0 = 0.02  # resting blood volume fraction

                # Buxton 1998
                k1 = 7 * rho
                k2 = 2.
                k3 = 2 * rho - 0.2

                BOLD = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
                return area, BOLD, time

            bold = Parallel(n_jobs=-1)(
                delayed(calculate_BOLD)(
                    rate,
                    area
                    ) for area, rate in rate_hist_areas.items()
            )

            tmp = {}
            for area, b, t in bold:
                tmp[(area, 'bold')] = b
                tmp[(area, 't')] = t

            BOLDSIGNAL = pd.Series(tmp)
            BOLDSIGNAL.to_pickle(os.path.join(
                self.ana_folder, 'bold_signal.pkl'
            ))
        return BOLDSIGNAL

    def plotBoxPlot(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.rate'):
            self.rate = self.meanFiringRate()
        extension = self.ana_dict['extension']
        dpi = self.ana_dict['dpi']
        fig = plt.figure(1)
        area = np.unique(self.rate.index.get_level_values(0))
        layer = np.unique(self.rate.index.get_level_values(1))
        pop_type = np.unique(self.rate.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        rates = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.rate.items():
            rates.loc[a, l+p] = r
        ax = sns.boxplot(
            data=rates,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Rate (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'boxplot.{0}'.format(extension)
        ), dpi=dpi)
        plt.clf()
        plt.close(fig)

    @timeit
    def popCorrCoeff(self):
        """
        Compute correlation coefficients for a subsample of neurons for the
        entire network. Subsample is set to 200 for duration reasons. Higher
        subsample size might yield more accurate results but is costly timing
        wise.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged correlation coefficients.
        """
        try:
            cc = pd.read_pickle(os.path.join(
                self.sim_folder, 'cc.pkl'
            ))
        except FileNotFoundError:
            # lv = self.spikes.apply(LV, args=(t_ref, t_start, t_stop))
            cc = self.spikes.apply(correlation, args=(self.ana_dict, self.sim_dict))
            # cc = Parallel(n_jobs=-1)(
            #     delayed(correlation)(
            #         sts,
            #         self.ana_dict
            #         ) for sts in self.spikes.values
            # )
            cc = pd.Series(cc, index=self.spikes.index)

            cc.to_pickle(os.path.join(self.sim_folder, 'cc.pkl'))
        return cc

    def plotCorrCoff(self):
        """
        Generates a boxplot of the correlation coefficients of the different
        populations averaged over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cc'):
            self.pop_cc = self.popCorrCoeff()
        extension = self.ana_dict['extension']
        dpi = self.ana_dict['dpi']
        fig = plt.figure(1)
        area = np.unique(self.pop_cc.index.get_level_values(0))
        layer = np.unique(self.pop_cc.index.get_level_values(1))
        pop_type = np.unique(self.pop_cc.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_ccs = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cc.items():
            pop_ccs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_ccs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cc.{0}'.format(extension)
        ), dpi=dpi)
        plt.clf()
        plt.close(fig)

    @timeit
    def popLv(self):
        """
        Compute the Lv value of the spikes.
        See Shinomoto et al. 2009 for details.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged Lv.
        """
        try:
            lv = pd.read_pickle(os.path.join(
                self.sim_folder, 'lv.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated lvs or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['lvr']['t_start']
            t_stop = self.ana_dict['lvr']['t_stop']
            min_spikes = self.ana_dict['lvr']['min_spikes']
            t_ref_E = self.net_dict['neuron_params_E']['t_ref']
            t_ref_I = self.net_dict['neuron_params_I']['t_ref']
            # TODO different refractory periods
            assert(t_ref_E == t_ref_I)
            t_ref = t_ref_E
            lv = self.spikes.apply(LV, args=(t_ref, t_start, t_stop,
                                             min_spikes))
            lv.to_pickle(os.path.join(self.sim_folder, 'lv.pkl'))
        return lv

    def plotPopLV(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_lv'):
            self.pop_lv = self.popLv()
        extension = self.ana_dict['extension']
        dpi = self.ana_dict['dpi']
        fig = plt.figure(1)
        area = np.unique(self.pop_lv.index.get_level_values(0))
        layer = np.unique(self.pop_lv.index.get_level_values(1))
        pop_type = np.unique(self.pop_lv.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_lvs = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_lv.items():
            pop_lvs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_lvs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Lv (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'lv_isi.{0}'.format(extension)
        ), dpi=dpi)
        plt.clf()
        plt.close(fig)

    @timeit
    def popCvIsi(self):
        """
        Calculates the coefficient of variation cv of the interspike intervals
        isi.

        Returns
        -------
        mean : Pandas Dataframe
            Mean CV ISI value of the population
        """
        try:
            cv_isi = pd.read_pickle(os.path.join(
                self.sim_folder, 'cv_isi.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated cv isis or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['cv']['t_start']
            t_stop = self.ana_dict['cv']['t_stop']
            min_spikes = self.ana_dict['cv']['min_spikes']
            cv_isi = self.spikes.apply(cvIsi, args=(t_start, t_stop,
                                                    min_spikes))
            cv_isi.to_pickle(os.path.join(self.sim_folder, 'cv_isi.pkl'))
        return cv_isi

    def plotPopCVIsi(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cv_isi'):
            self.pop_cv_isi = self.popCvIsi()
        extension = self.ana_dict['extension']
        dpi = self.ana_dict['dpi']
        fig = plt.figure(1)
        area = np.unique(self.pop_cv_isi.index.get_level_values(0))
        layer = np.unique(self.pop_cv_isi.index.get_level_values(1))
        pop_type = np.unique(self.pop_cv_isi.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_cv_isis = pd.DataFrame(data=0.0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cv_isi.items():
            pop_cv_isis.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_cv_isis,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.patches[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Cv Isi (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cv_isi.{0}'.format(extension)
        ), dpi=dpi)
        plt.clf()
        plt.close(fig)

    @timeit
    def calc_lfp_proxy(self):
        """
        Calculate local field potential (LFP) proxy from PSCs.

        Saves two pkl files:
        - lfp_proxy.pkl: z-scored LFP proxy, indexed by (area, layer)
        - lfp_proxy_components.pkl: I_syn_ex, I_syn_in, lfp columns, indexed by (area, layer)

        Returns
        ---
        z (pandas.DataFrame): z-scored LFP proxy
        """
        try:
            I_syn = self._readIsyn()
            idx = pd.MultiIndex.from_arrays(
                [I_syn["area"], I_syn["layer"], I_syn["pop"]],
                names=["area", "layer", "pop"]
            )

            # Build arrays first so we know the actual number of recorded steps.
            # The NEST multimeter records at its own interval (default 1.0 ms),
            # which may differ from sim_resolution. Infer recording interval from
            # n_steps and t_sim, then truncate with [:n_steps] to absorb any
            # floating-point drift in np.arange.
            ex_arr = np.stack(I_syn["I_syn_ex"].apply(lambda g: np.sum(g, axis=0)))
            in_arr = np.stack(I_syn["I_syn_in"].apply(lambda g: np.sum(g, axis=0)))
            n_steps = ex_arr.shape[1]
            sim_res = self.sim_dict['sim_resolution']
            t_sim   = self.sim_dict['t_sim']
            rec_interval = max(sim_res, round(t_sim / n_steps / sim_res) * sim_res)
            t = np.arange(rec_interval, t_sim + rec_interval / 2, rec_interval)[:n_steps]
            if self.logger:
                self.logger.debug(
                    'calc_lfp_proxy: n_steps=%d  rec_interval=%.4f ms  t_sim=%.1f ms',
                    n_steps, rec_interval, t_sim
                )

            # Per-population DataFrames: (N_pop, T), index = (area, layer, pop)
            I_syn_ex = pd.DataFrame(ex_arr, index=idx, columns=t)
            I_syn_in = pd.DataFrame(in_arr, index=idx, columns=t)

            # Aggregate to (area, layer) level by summing across populations.
            # LFP proxy formula is applied once per layer, not per population,
            # consistent with Mazzoni et al.  Sign is preserved (no abs) so
            # that oscillatory phase information is retained for coherence.
            I_syn_ex_layer = I_syn_ex.groupby(level=['area', 'layer']).sum()
            I_syn_in_layer = I_syn_in.groupby(level=['area', 'layer']).sum()
            # Mazzoni et al. 2015 (PLoS Comput Biol 11(12):e1004584), Table 1 /
            # Eq 4, "Reference Weighted Sum" (RWS) LFP proxy:
            #   LFP(t) = AMPA(t - 6 ms) [+/-] 1.65 * GABA(t)
            # i.e. the excitatory (AMPA-like) current leads the inhibitory
            # (GABA-like) current by a fixed 6 ms (tau_AMPA = 6 ms, tau_GABA = 0
            # in the paper). This is a paper-derived constant, not a simulation
            # parameter, so it is hardcoded rather than exposed via ana_dict.
            AMPA_DELAY_MS = 6.0
            I_syn_ex_layer_delayed = _delay_signal(
                I_syn_ex_layer.values, rec_interval, AMPA_DELAY_MS)
            lfp = pd.DataFrame(
                I_syn_ex_layer_delayed + 1.65 * I_syn_in_layer.values,
                index=I_syn_ex_layer.index, columns=t
            )
            z = lfp.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

            components = pd.concat(
                {"I_syn_ex": I_syn_ex_layer, "I_syn_in": I_syn_in_layer, "lfp": lfp},
                axis=1
            )
            components.to_pickle(os.path.join(self.ana_folder, "lfp_proxy_components.pkl"))
            z.to_pickle(os.path.join(self.ana_folder, "lfp_proxy.pkl"))
            return z

        except Exception as e:
            if self.logger:
                self.logger.error('calc_lfp_proxy failed: %s', e, exc_info=True)
            raise

    @timeit
    def calc_lfp_proxy_kernel(self):
        """
        Calculate LFP proxy at (area, layer) level using PSC kernel convolution
        on spike data. Controlled by ana_dict['fast_lfp_kernel'] (bool):

            True  (default): convolves population-averaged rate_hist at binsize
                             resolution — fast, lower Nyquist frequency ceiling.
            False           : builds spike trains at sim_resolution from raw
                             spike times — slower, full frequency resolution.

        Returns a signed z-scored oscillatory signal (no abs) suitable for
        coherence analysis. Phase information is preserved.

        Saves result to lfp_proxy_kernel.pkl.

        Returns
        -------
        z : pd.DataFrame
            Z-scored LFP proxy at (area, layer) level. Columns are time in ms.
        """
        pkl_path = os.path.join(self.ana_folder, 'lfp_proxy_kernel.pkl')
        try:
            return pd.read_pickle(pkl_path)
        except FileNotFoundError:
            pass

        lfp_layer, _ = self._kernelLayerCurrent()
        z = lfp_layer.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        z.to_pickle(pkl_path)
        if self.logger:
            cols = np.asarray(z.columns, dtype=float)
            dt_lfp = cols[1] - cols[0] if len(cols) > 1 else float('nan')
            self.logger.info(
                'LFP proxy (kernel) saved: %d layers, dt=%.3f ms, duration=%.1f ms',
                len(z), dt_lfp, cols[-1])
        return z

    @timeit
    def _kernelLayerCurrent(self):
        """
        Per-(area, layer) signed net synaptic current of the kernel LFP proxy,
        *before* z-scoring, together with the population neuron counts.

        Factored out of ``calc_lfp_proxy_kernel`` so that both the per-layer
        proxy and the ROI-level proxy (``calcAreaLfp``) build on one identical
        computation.  Returning the pre-z-score current is essential for ROI
        aggregation: z-scoring is non-linear across rows, so it must happen
        *after* the area-level NN-weighted sum, not before.  Cached on the
        instance to avoid recomputation within a single analysis run.

        Returns
        -------
        (lfp_layer, NN) : tuple
            lfp_layer : pd.DataFrame, MultiIndex (area, layer), columns = time (ms)
            NN        : pd.Series,    MultiIndex (area, layer, pop), neuron counts
        """
        if hasattr(self, '_lfp_layer_current'):
            return self._lfp_layer_current, self._lfp_layer_NN

        fast   = self.ana_dict.get('fast_lfp_kernel', False)
        sim_dt = self.sim_dict['sim_resolution']   # ms
        t_sim  = self.sim_dict['t_sim']            # ms

        # --- Time resolution inference and Nyquist check ---
        if fast:
            dt_lfp = float(self.ana_dict['rate_histogram']['binsize'])
            f_nyq  = 500.0 / dt_lfp
            if self.logger:
                self.logger.info(
                    'LFP kernel mode: fast (rate_hist), dt=%.1f ms, f_Nyquist=%.0f Hz',
                    dt_lfp, f_nyq)
            if f_nyq < 100.0 and self.logger:
                self.logger.warning(
                    'rate_hist binsize=%.1f ms → f_Nyquist=%.0f Hz < 100 Hz; '
                    'gamma band may be underresolved. Consider fast_lfp_kernel=False.',
                    dt_lfp, f_nyq)
        else:
            dt_lfp = sim_dt
            if self.logger:
                self.logger.info(
                    'LFP kernel mode: precise (raw spikes), dt=%.3f ms, f_Nyquist=%.0f Hz',
                    dt_lfp, 500.0 / dt_lfp)

        # --- Common network parameters ---
        if fast:
            if not hasattr(self, 'rate_hist'):
                self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
            index = self.rate_hist.index
        else:
            index = self.spikes.index

        try:
            NN       = self.net_dict['neuron_numbers'].loc[index]
            weights  = self.net_dict['weights'].loc[index, index]
            synapses = self.net_dict['synapses_internal'].loc[index, index]
            indeg    = synapses.div(NN, axis=0)
        except KeyError as e:
            if self.logger:
                self.logger.error(
                    'calc_lfp_proxy_kernel: network parameter index mismatch: %s',
                    e, exc_info=True)
            raise

        is_abeta = self._abeta_active()

        def _effective_kernel(pop_type, synapse_key, dt, area, layer):
            """Weighted exponential kernel; handles Abeta mixed populations.

            The Abeta fraction may be configured per area/layer, so it is
            resolved for the specific ``(area, layer)`` of the population.
            """
            if pop_type == 'E':
                tau_n     = self.net_dict['neuron_params_E'][synapse_key]
                ratio_cfg = self.net_dict.get('Abeta_ratio_E', 0.0)
                tau_a     = self.net_dict.get('neuron_params_Abeta_E', {}).get(synapse_key, tau_n)
            else:
                tau_n     = self.net_dict['neuron_params_I'][synapse_key]
                ratio_cfg = self.net_dict.get('Abeta_ratio_I', 0.0)
                tau_a     = self.net_dict.get('neuron_params_Abeta_I', {}).get(synapse_key, tau_n)
            f = resolve_abeta_ratio(ratio_cfg, area, layer) if is_abeta else 0.0

            k_n = kernel_for_psc(tau_n, dt)
            if f == 0.0 or tau_a == tau_n:
                return tau_n, k_n

            k_a = kernel_for_psc(tau_a, dt)
            n   = max(len(k_n), len(k_a))
            k_n = np.pad(k_n, (0, n - len(k_n)))
            k_a = np.pad(k_a, (0, n - len(k_a)))
            return (1 - f) * tau_n + f * tau_a, (1 - f) * k_n + f * k_a

        # --- Build PSC time series per source population ---
        try:
            pscs = {}

            if fast:
                sample = next(iter(self.rate_hist))
                n_bins = len(sample)
                # Cross-check inferred dt against declared binsize
                n_bins_expected = int(round(t_sim / dt_lfp))
                if abs(n_bins - n_bins_expected) > 1:
                    dt_inferred = t_sim / n_bins
                    if self.logger:
                        self.logger.warning(
                            'rate_hist has %d bins but t_sim/binsize → %d bins expected; '
                            'using inferred dt=%.4f ms instead of declared %.1f ms',
                            n_bins, n_bins_expected, dt_inferred, dt_lfp)
                    dt_lfp = dt_inferred

                t = np.linspace(dt_lfp, n_bins * dt_lfp, n_bins)

                for (area, layer, pop), dat in self.rate_hist.items():
                    if len(dat) != n_bins:
                        raise ValueError(
                            'rate_hist length inconsistency at %s/%s/%s: '
                            'got %d bins, expected %d' % (area, layer, pop, len(dat), n_bins))
                    synapse_key     = 'tau_syn_ex' if pop == 'E' else 'tau_syn_in'
                    tau_eff, kernel = _effective_kernel(pop, synapse_key, dt_lfp, area, layer)
                    pscs[(area, layer, pop)] = convolve(tau_eff * dat, kernel, mode='same')

            else:
                n_bins  = int(round(t_sim / sim_dt))
                t       = np.linspace(sim_dt, t_sim, n_bins)
                t_edges = np.linspace(0.0, t_sim, n_bins + 1)  # linspace avoids float accumulation

                for (area, layer, pop), spike_times in self.spikes.items():
                    n_neurons = int(NN.loc[(area, layer, pop)])

                    # spike_times may be a flat 1D array (all spikes concatenated)
                    # or an object array of per-neuron spike time arrays depending
                    # on how _readSpikesFromDAT stored the data.  Normalise to 1D.
                    arr = np.asarray(spike_times)
                    if arr.dtype == object:
                        flat_spikes = (np.concatenate(arr.tolist()).astype(float)
                                       if len(arr) > 0 else np.array([], dtype=float))
                    else:
                        flat_spikes = arr.astype(float)

                    if flat_spikes.ndim != 1:
                        raise ValueError(
                            'Unexpected spike_times shape %s at %s/%s/%s; '
                            'expected 1D array.' % (flat_spikes.shape, area, layer, pop))

                    # Warn on out-of-window spikes
                    out_of_range = int(np.sum((flat_spikes < 0.0) | (flat_spikes > t_sim)))
                    if out_of_range > 0 and self.logger:
                        self.logger.warning(
                            '%s/%s/%s: %d spike(s) outside [0, %.1f ms] will be discarded',
                            area, layer, pop, out_of_range, t_sim)

                    spike_train, _ = np.histogram(flat_spikes, bins=t_edges)
                    # Convert counts to firing rate in Hz
                    rate = spike_train.astype(float) / (n_neurons * sim_dt * 1e-3)

                    synapse_key     = 'tau_syn_ex' if pop == 'E' else 'tau_syn_in'
                    tau_eff, kernel = _effective_kernel(pop, synapse_key, sim_dt, area, layer)
                    pscs[(area, layer, pop)] = convolve(tau_eff * rate, kernel, mode='same')

        except (KeyError, ValueError, TypeError) as e:
            if self.logger:
                self.logger.error(
                    'calc_lfp_proxy_kernel: PSC computation failed: %s', e, exc_info=True)
            raise

        # Validate uniform PSC length across all populations
        psc_lengths = set(len(v) for v in pscs.values())
        if len(psc_lengths) > 1:
            msg = ('calc_lfp_proxy_kernel: inconsistent PSC array lengths across '
                   'populations: %s' % {k: len(v) for k, v in pscs.items()})
            if self.logger:
                self.logger.error(msg)
            raise ValueError(msg)

        pscs_df = pd.DataFrame(pscs, index=t).T   # (n_pops, n_timesteps)

        # --- E/I current separation ---
        try:
            w_signed = weights * indeg
            I_ex = w_signed.clip(lower=0).dot(pscs_df)   # positive (E sources)
            I_in = w_signed.clip(upper=0).dot(pscs_df)   # negative (I sources)
        except Exception as e:
            if self.logger:
                self.logger.error(
                    'calc_lfp_proxy_kernel: E/I current computation failed: %s',
                    e, exc_info=True)
            raise

        # Mazzoni et al. 2015 (PLoS Comput Biol 11(12):e1004584), Table 1 /
        # Eq 4, "Reference Weighted Sum" (RWS) LFP proxy: the excitatory
        # (AMPA-like) current leads the inhibitory (GABA-like) current by a
        # fixed 6 ms (tau_AMPA = 6 ms, tau_GABA = 0). Paper-derived constant,
        # not a simulation parameter, hence hardcoded rather than exposed via
        # ana_dict.
        AMPA_DELAY_MS = 6.0
        I_ex_delayed = pd.DataFrame(
            _delay_signal(I_ex.values, dt_lfp, AMPA_DELAY_MS),
            index=I_ex.index, columns=I_ex.columns
        )

        # Signed net current: no abs to preserve oscillatory phase for coherence
        lfp_pop   = I_ex_delayed + 1.65 * I_in
        lfp_layer = (lfp_pop.mul(NN, axis=0)
                             .groupby(level=[0, 1]).sum()
                             .div(NN.groupby(level=[0, 1]).sum(), axis=0))

        self._lfp_layer_current = lfp_layer
        self._lfp_layer_NN      = NN
        return lfp_layer, NN

    @timeit
    def calcAreaLfp(self):
        """
        ROI (area)-level LFP-like proxy from the kernel method.

        Aggregates the per-(area, layer) signed synaptic current across *all*
        layers and populations of each area, NN-weighted and *before* any
        z-scoring, then z-scores once per area.  Aggregating before
        normalisation is required: ``lfp_proxy_kernel.pkl`` is z-scored per
        layer (unit variance in every row), so averaging those rows would be
        meaningless.  NN-weighting is required because the kernel current is a
        per-neuron quantity while the field contribution scales with population
        size (see ``_kernelLayerCurrent``).

        Note: this is a whole-network computation reduced to area level — each
        area's current already includes its cortico-cortical afferents; it
        cannot be obtained from that area's spikes alone.

        Saves result to lfp_proxy_area.pkl.

        Returns
        -------
        z : pd.DataFrame
            MultiIndex/Index = area, columns = time (ms); z-scored per area.
        """
        lfp_layer, NN = self._kernelLayerCurrent()

        # NN per (area, layer); aggregating layers with these weights is exactly
        # the NN-weighted mean over all of an area's populations.
        NN_layer = NN.groupby(level=['area', 'layer']).sum()
        lfp_area = (lfp_layer.mul(NN_layer, axis=0)
                             .groupby(level='area').sum()
                             .div(NN_layer.groupby(level='area').sum(), axis=0))

        z = lfp_area.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
        z.to_pickle(os.path.join(self.ana_folder, 'lfp_proxy_area.pkl'))
        if self.logger:
            self.logger.info('ROI LFP proxy saved: %d areas, %d samples',
                             z.shape[0], z.shape[1])
        return z

    @timeit
    def calcAreaLfpCorrelation(self):
        """
        Functional connectivity between ROIs as the Pearson correlation of the
        area-level LFP proxy time series.

        Reads lfp_proxy_area.pkl (computed by calcAreaLfp; computed on demand if
        absent), correlates every area pair over the window t >= tmin, and
        orders rows/columns by resting-state subnetwork (left_ordering).

        Saves result to lfp_area_correlation.pkl.

        Returns
        -------
        corr : pd.DataFrame
            Square DataFrame (area x area), Pearson r in [-1, 1].
        """
        pkl_path = os.path.join(self.ana_folder, 'lfp_proxy_area.pkl')
        try:
            roi = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            roi = self.calcAreaLfp()

        tmin = self.ana_dict['calcAreaLfpCorrelation']['tmin']
        z = roi.loc[:, roi.columns >= tmin]

        # Order areas by subnetwork before correlating
        all_areas     = z.index.tolist()
        ordered_areas = [a for a in left_ordering if a in all_areas]
        ordered_areas += [a for a in all_areas if a not in left_ordering]
        z = z.loc[ordered_areas]

        corr = pd.DataFrame(np.corrcoef(z.values),
                            index=z.index, columns=z.index)
        corr.to_pickle(os.path.join(self.ana_folder, 'lfp_area_correlation.pkl'))
        if self.logger:
            self.logger.info('ROI LFP correlation saved: %d x %d areas',
                             corr.shape[0], corr.shape[1])
        return corr

    @timeit
    def calcAreaLfpCoherence(self):
        """
        Pairwise magnitude-squared coherence between ROI (area-level) LFP
        proxy time series, averaged over a frequency band [fmin, fmax].

        The area-level signal is the kernel-method net synaptic current
        aggregated per area (see calcAreaLfp), so this is the ROI analogue of
        calcLfpCoherence, which operates at (area, layer) level.  Unlike the
        Pearson correlation in calcAreaLfpCorrelation, coherence is computed in
        the frequency domain (Welch PSD + cross-spectral density), capturing
        band-limited phase-consistent coupling rather than broadband linear
        correlation.

        Reads lfp_proxy_area.pkl (computed by calcAreaLfp; computed on demand if
        absent), and orders rows/columns by resting-state subnetwork
        (left_ordering) so the saved matrix matches calcAreaLfpCorrelation.

        Saves result to lfp_area_coherence.pkl.

        Returns
        -------
        coherence : pd.DataFrame
            Square DataFrame (area x area), mean magnitude-squared coherence
            in [fmin, fmax] Hz, values in [0, 1].
        """
        pkl_path = os.path.join(self.ana_folder, 'lfp_proxy_area.pkl')
        try:
            roi = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            roi = self.calcAreaLfp()

        tmin = self.ana_dict['calcAreaLfpCoherence']['tmin']
        fmin = self.ana_dict['calcAreaLfpCoherence']['fmin']
        fmax = self.ana_dict['calcAreaLfpCoherence']['fmax']

        z = roi.loc[:, roi.columns >= tmin]

        # Order areas by subnetwork before computing, matching
        # calcAreaLfpCorrelation so both ROI matrices share row/column order.
        all_areas     = z.index.tolist()
        ordered_areas = [a for a in left_ordering if a in all_areas]
        ordered_areas += [a for a in all_areas if a not in left_ordering]
        z = z.loc[ordered_areas]

        dt = z.columns.values[1] - z.columns.values[0]   # ms
        fs = 1000. / dt                                    # Hz

        areas   = z.index.tolist()
        signals = z.values        # (N, n_t)
        N       = len(areas)

        # Pre-compute Welch PSD for every area once, reuse per pair
        f_ref, _ = welch(signals[0], fs=fs)
        band = (f_ref >= fmin) & (f_ref <= fmax)
        psds = [welch(s, fs=fs)[1] for s in signals]

        coh_matrix = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                _, Pxy = csd(signals[i], signals[j], fs=fs)
                Cxy = np.abs(Pxy[band]) ** 2 / (psds[i][band] * psds[j][band])
                coh_matrix[i, j] = coh_matrix[j, i] = Cxy.mean()

        coherence = pd.DataFrame(coh_matrix, index=areas, columns=areas)
        coherence.to_pickle(os.path.join(self.ana_folder, 'lfp_area_coherence.pkl'))
        if self.logger:
            self.logger.info('ROI LFP coherence saved: %d x %d areas',
                             coherence.shape[0], coherence.shape[1])
        return coherence

    @timeit
    def calcLfpCoherence(self):
        """
        Compute pairwise magnitude-squared coherence between all (area, layer)
        LFP signal pairs, averaged over a frequency band [fmin, fmax].

        Cross-layer pairs (e.g. area_A/II_III vs area_B/V) are included.
        Uses Welch's method: each signal's PSD is pre-computed once, then
        the cross-spectral density is computed per unique pair.

        Saves result to lfp_coherence.pkl.

        Returns
        -------
        coherence : pd.DataFrame
            Square DataFrame with MultiIndex (area, layer) on both rows and
            columns.  Values are mean magnitude-squared coherence in
            [fmin, fmax] Hz.
        """
        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    if self.logger:
                        self.logger.warning(
                            'calcLfpCoherence: %s not found, trying next source.',
                            pkl_name)
            else:
                msg = ('calcLfpCoherence: no LFP proxy pkl found '
                       '(lfp_proxy.pkl, lfp_proxy_kernel.pkl). '
                       'Run calc_lfp_proxy or calc_lfp_proxy_kernel first.')
                if self.logger:
                    self.logger.error(msg)
                raise FileNotFoundError(msg)
        tmin = self.ana_dict['calcLfpCoherence']['tmin']
        fmin = self.ana_dict['calcLfpCoherence']['fmin']
        fmax = self.ana_dict['calcLfpCoherence']['fmax']

        z = self.lfp.loc[:, self.lfp.columns >= tmin]
        dt = z.columns.values[1] - z.columns.values[0]   # ms
        fs = 1000. / dt                                    # Hz

        # Each row is one (area, layer) signal; index order is area-first
        # (preserved from groupby in calc_lfp_proxy / calc_lfp_proxy_kernel)
        index_pairs = list(z.index)   # [(area, layer), ...]
        signals     = z.values        # (N, n_t)
        N           = len(index_pairs)

        # Pre-compute Welch PSD for every signal once
        f_ref, _ = welch(signals[0], fs=fs)
        band = (f_ref >= fmin) & (f_ref <= fmax)
        psds = [welch(s, fs=fs)[1] for s in signals]

        # Full N×N coherence — includes both same-layer and cross-layer pairs
        coh_matrix = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                _, Pxy = csd(signals[i], signals[j], fs=fs)
                Cxy = np.abs(Pxy[band]) ** 2 / (psds[i][band] * psds[j][band])
                coh_matrix[i, j] = coh_matrix[j, i] = Cxy.mean()

        mi = pd.MultiIndex.from_tuples(index_pairs, names=['area', 'layer'])
        result = pd.DataFrame(coh_matrix, index=mi, columns=mi)
        result.to_pickle(os.path.join(self.ana_folder, 'lfp_coherence.pkl'))
        return result

    @timeit
    def _thetaBetaNM(self):
        """
        Resolve the (n, m) harmonic ratio and the θ / β bands used for the
        cross-frequency PLV from ``ana_dict['calcLfpPlv']``.

        n is the multiplier of the θ (slow) phase, m of the β (fast) phase,
        chosen so that n * f_theta ≈ m * f_beta.  Either taken verbatim from
        the ``nm`` config entry, or derived automatically from the band-centre
        ratio with denominator bounded by ``n_max``.

        Returns
        -------
        (n, m, theta_band, beta_band) : tuple
        """
        cfg        = self.ana_dict['calcLfpPlv']
        theta_band = tuple(cfg['theta_band'])
        beta_band  = tuple(cfg['beta_band'])
        nm         = cfg.get('nm', None)
        if nm is not None:
            n, m = int(nm[0]), int(nm[1])
        else:
            from fractions import Fraction
            f_theta_c = 0.5 * (theta_band[0] + theta_band[1])
            f_beta_c  = 0.5 * (beta_band[0] + beta_band[1])
            ratio = Fraction(f_beta_c / f_theta_c).limit_denominator(
                int(cfg.get('n_max', 5)))
            n, m = ratio.numerator, ratio.denominator  # n*f_theta ≈ m*f_beta
        return n, m, theta_band, beta_band

    def calcLfpPlv(self):
        """
        Compute cross-frequency θ–β Phase Locking Value (PLV) between all
        (area, layer) LFP signal pairs.

        Each signal is bandpass-filtered separately in the θ and β bands; the
        Hilbert transform yields the instantaneous phase in each band.  The
        n:m phase locking value between θ-phase of signal i and β-phase of
        signal j is:

            PLV(i, j) = |mean(exp(1j * (n*phi_theta_i - m*phi_beta_j)))|

        with n, m the harmonic ratio (n*f_theta ≈ m*f_beta).  The matrix is
        therefore asymmetric: rows index the θ-band signal, columns the β-band
        signal; the diagonal is the within-region θ–β coupling.

        Saves result to lfp_plv.pkl.

        Returns
        -------
        plv : pd.DataFrame
            Square DataFrame with MultiIndex (area, layer) on both rows
            (θ band) and columns (β band).  Values are n:m PLV in [0, 1].
        """
        from scipy.signal import butter, sosfiltfilt

        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    if self.logger:
                        self.logger.warning(
                            'calcLfpPlv: %s not found, trying next source.',
                            pkl_name)
            else:
                msg = ('calcLfpPlv: no LFP proxy pkl found '
                       '(lfp_proxy.pkl, lfp_proxy_kernel.pkl). '
                       'Run calc_lfp_proxy or calc_lfp_proxy_kernel first.')
                if self.logger:
                    self.logger.error(msg)
                raise FileNotFoundError(msg)

        tmin = self.ana_dict['calcLfpPlv']['tmin']
        n, m, theta_band, beta_band = self._thetaBetaNM()

        z = self.lfp.loc[:, self.lfp.columns >= tmin]
        dt = z.columns.values[1] - z.columns.values[0]  # ms
        fs = 1000. / dt  # Hz

        index_pairs = list(z.index)
        signals     = z.values  # (N, n_t)
        N           = len(index_pairs)

        if self.logger:
            self.logger.info(
                'calcLfpPlv: cross-frequency θ%s–β%s PLV, n:m = %d:%d',
                theta_band, beta_band, n, m)

        nyq = fs / 2.

        def _band_phase(band):
            low  = band[0] / nyq
            high = min(band[1], nyq * 0.99) / nyq
            # Second-order-section form: the transfer-function (b, a) form is
            # numerically unstable at the very small normalised cut-offs that
            # arise when fs >> band (e.g. fs=10 kHz, θ=4-8 Hz pushes a pole to
            # |z|=1.018 > 1), making filtfilt diverge to inf/NaN.  SOS keeps the
            # poles correctly placed and the filter stable.
            sos = butter(4, [low, high], btype='band', output='sos')
            ph = np.empty_like(signals)
            for i, s in enumerate(signals):
                ph[i] = np.angle(hilbert(sosfiltfilt(sos, s)))
            return ph

        phase_theta = _band_phase(theta_band)   # rows
        phase_beta  = _band_phase(beta_band)    # columns

        # Full asymmetric N×N n:m PLV matrix (rows = θ signal, cols = β signal)
        plv_matrix = np.empty((N, N))
        for i in range(N):
            ntheta_i = n * phase_theta[i]
            for j in range(N):
                dphi = ntheta_i - m * phase_beta[j]
                plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * dphi)))

        mi     = pd.MultiIndex.from_tuples(index_pairs, names=['area', 'layer'])
        result = pd.DataFrame(plv_matrix, index=mi, columns=mi)
        result.to_pickle(os.path.join(self.ana_folder, 'lfp_plv.pkl'))
        return result

    @timeit
    def plotLayerLfpCoherence(self):
        """
        Plot pairwise LFP coherence matrices, one per cortical layer.

        Areas are sorted by resting-state subnetwork (left_ordering). Thick
        black lines separate subnetworks; thin lines separate individual areas
        within a subnetwork.  Y-axis area labels are on the right; subnetwork
        labels on the left; colorbar at the bottom.

        Produces one figure per layer saved as lfp_coherence_layer_<layer>.<ext>.
        """
        import matplotlib.transforms as mtransforms
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        pkl_path = os.path.join(self.ana_folder, 'lfp_coherence.pkl')
        try:
            coherence = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            coherence = self.calcLfpCoherence()

        extension = self.ana_dict['extension']
        dpi        = self.ana_dict['dpi']
        vmax       = self.ana_dict['plotLayerLfpCoherence']['vmax']
        cell_size  = self.ana_dict['plotLayerLfpCoherence']['cell_size_inches']

        # ------------------------------------------------------------------ #
        # Fixed font sizes shared across all layer plots                      #
        # Derived from cell_size alone (= figsize/n when the floor is absent),#
        # so they are independent of how many areas each layer happens to have.#
        # ------------------------------------------------------------------ #
        tick_fontsize   = max(10, int(cell_size * 72 * 0.28))
        subnet_fontsize = max(12, int(tick_fontsize * 1.35))
        title_fontsize  = max(16, tick_fontsize + 8)
        cbar_fontsize   = max(12, tick_fontsize)

        layers = coherence.index.get_level_values('layer').unique()
        for layer in layers:
            # Extract within-layer block
            coh_matrix = (coherence
                          .xs(layer, level='layer', axis=0)
                          .xs(layer, level='layer', axis=1))

            # -------------------------------------------------------------- #
            # Sort areas by subnetwork                                        #
            # -------------------------------------------------------------- #
            all_areas     = coh_matrix.index.tolist()
            ordered_areas = [a for a in left_ordering if a in all_areas]
            ordered_areas += [a for a in all_areas if a not in left_ordering]
            coh_matrix = coh_matrix.loc[ordered_areas, ordered_areas]

            areas = coh_matrix.index.tolist()
            n     = len(areas)

            # -------------------------------------------------------------- #
            # Boundary positions and subnetwork blocks                        #
            # -------------------------------------------------------------- #
            area_boundaries   = list(range(1, n))
            subnet_boundaries = []
            subnet_blocks     = []   # (subnet_name, row_start, row_end)

            prev_subnet      = None
            subnet_row_start = 0
            for i, area in enumerate(areas):
                new_subnet = left_ordering.get(area, 'other')
                if prev_subnet is not None and new_subnet != prev_subnet:
                    subnet_boundaries.append(i)
                    subnet_blocks.append((prev_subnet, subnet_row_start, i))
                    subnet_row_start = i
                prev_subnet = new_subnet
            subnet_blocks.append((prev_subnet, subnet_row_start, n))

            # -------------------------------------------------------------- #
            # Figure                                                          #
            # -------------------------------------------------------------- #
            figsize = max(8, n * cell_size)
            fig, ax = plt.subplots(figsize=(figsize, figsize))

            cmap = plt.get_cmap('YlOrRd').copy()
            cmap.set_bad(color='white')

            im = ax.imshow(
                coh_matrix.values,
                aspect='equal',
                cmap=cmap,
                vmin=0.,
                vmax=vmax,
                interpolation='nearest',
            )

            # -------------------------------------------------------------- #
            # Area tick labels — names on the right y-axis                   #
            # -------------------------------------------------------------- #
            tick_pos = list(range(n))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(areas, rotation=90, fontsize=tick_fontsize)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(areas, rotation=0, fontsize=tick_fontsize)

            # -------------------------------------------------------------- #
            # Colorbar — gap computed in inches to clear rotated x-labels    #
            # -------------------------------------------------------------- #
            max_xlab_chars = max(len(a) for a in areas)
            xlab_height_in = tick_fontsize / 72.0 * max_xlab_chars * 0.6
            cbar_pad_in    = xlab_height_in + 0.1

            cbar_shrink = 0.5
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='2%', pad=cbar_pad_in)
            cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            p = cax.get_position()
            cax.set_position([
                p.x0 + p.width * (1 - cbar_shrink) / 2,
                p.y0,
                p.width * cbar_shrink,
                p.height,
            ])
            cbar.set_label('Coherence', fontsize=cbar_fontsize)
            cbar.ax.tick_params(labelsize=max(10, cbar_fontsize - 2))

            # -------------------------------------------------------------- #
            # Subnetwork labels on the left, centred on each block           #
            # -------------------------------------------------------------- #
            left_trans = mtransforms.blended_transform_factory(
                ax.transAxes, ax.transData
            )
            # Compute x so the right edge of the longest label sits 2% outside
            # the axes left boundary, regardless of font size or figure size.
            max_subnet_chars = max(len(name) for name, _, _ in subnet_blocks)
            axes_width_est   = figsize * 0.82   # approx after tight_layout rect
            label_half_width = (subnet_fontsize / 72.0 * 0.6 * max_subnet_chars
                                / axes_width_est)
            subnet_label_x   = -(label_half_width + 0.005)
            for subnet_name, row_start, row_end in subnet_blocks:
                centre = (row_start + row_end - 1) / 2.0
                ax.text(
                    subnet_label_x, centre, subnet_name,
                    transform=left_trans,
                    ha='center', va='center',
                    fontsize=subnet_fontsize, fontweight='bold',
                    clip_on=False,
                )

            # -------------------------------------------------------------- #
            # Separator lines                                                 #
            # -------------------------------------------------------------- #
            subnet_boundary_set = set(subnet_boundaries)
            for b in area_boundaries:
                if b not in subnet_boundary_set:
                    ax.axvline(x=b - 0.5, color='#C0C0C0', linewidth=1.0)
                    ax.axhline(y=b - 0.5, color='#C0C0C0', linewidth=1.0)
            for b in subnet_boundaries:
                ax.axvline(x=b - 0.5, color='black', linewidth=2.5)
                ax.axhline(y=b - 0.5, color='black', linewidth=2.5)

            ax.set_title(f'LFP Coherence — Layer {layer}',
                         fontsize=title_fontsize, pad=title_fontsize * 1.2)
            left_rect = max(0.02, -subnet_label_x - label_half_width + 0.03)
            fig.tight_layout(rect=[left_rect, 0.04, 0.96, 0.94], pad=3.0)
            fname = f'lfp_coherence_layer_{layer.replace("/", "_")}.{extension}'
            fig.savefig(os.path.join(self.plot_folder, fname), dpi=dpi,
                        bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    @timeit
    def plotFullLayerLfpCoherence(self):
        """
        Plot a single combined LFP coherence matrix covering all areas and
        all layers, including cross-layer pairs.

        Rows and columns are ordered as (area, layer), with areas sorted by
        resting-state subnetwork (following left_ordering). Thick black lines
        separate subnetworks; thin lines separate individual areas within a
        subnetwork.  Axis tick labels show area names only, centred on each
        area's block.

        Figure size is chosen adaptively so that every cell and label is
        legible.  Saved as lfp_coherence_fulllayer.<ext> in the plot folder.
        """
        pkl_path = os.path.join(self.ana_folder, 'lfp_coherence.pkl')
        try:
            coherence = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            coherence = self.calcLfpCoherence()

        extension = self.ana_dict['extension']
        dpi       = self.ana_dict['dpi']
        vmax      = self.ana_dict['plotFullLayerLfpCoherence']['vmax']
        cell_size = self.ana_dict['plotFullLayerLfpCoherence']['cell_size_inches']

        # ------------------------------------------------------------------ #
        # Sort areas by subnetwork (following left_ordering key order)        #
        # ------------------------------------------------------------------ #
        all_areas     = coherence.index.get_level_values('area').unique().tolist()
        ordered_areas = [a for a in left_ordering if a in all_areas]
        ordered_areas += [a for a in all_areas if a not in left_ordering]

        new_order = []
        for area in ordered_areas:
            new_order.extend([(a, l) for (a, l) in coherence.index if a == area])
        coherence = coherence.loc[new_order, new_order]

        index_pairs = list(coherence.index)   # [(area, layer), ...]
        combined    = coherence.values        # (N, N) — fully filled, no NaN
        n_total     = len(index_pairs)

        # ------------------------------------------------------------------ #
        # Boundary positions, tick positions, and subnetwork blocks          #
        # ------------------------------------------------------------------ #
        area_boundaries   = []   # dividing lines between every area pair
        subnet_boundaries = []   # subset of area_boundaries at subnet changes
        area_tick_pos     = []   # centre of each area's block
        area_tick_label   = []
        subnet_blocks     = []   # (subnet_name, row_start, row_end)

        current_area     = None
        prev_subnet      = None
        block_start      = 0
        subnet_row_start = 0
        for i, (area, _layer) in enumerate(index_pairs):
            if area != current_area:
                if current_area is not None:
                    area_boundaries.append(i)
                    area_tick_pos.append((block_start + i) / 2.0)
                    area_tick_label.append(current_area)
                new_subnet = left_ordering.get(area, 'other')
                if prev_subnet is not None and new_subnet != prev_subnet:
                    subnet_boundaries.append(i)
                    subnet_blocks.append((prev_subnet, subnet_row_start, i))
                    subnet_row_start = i
                prev_subnet  = new_subnet
                block_start  = i
                current_area = area
        # last area / last subnetwork block
        area_tick_pos.append((block_start + n_total) / 2.0)
        area_tick_label.append(current_area)
        subnet_blocks.append((prev_subnet, subnet_row_start, n_total))

        # ------------------------------------------------------------------ #
        # Figure                                                              #
        # ------------------------------------------------------------------ #
        figsize = max(12, n_total * cell_size)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        cmap = plt.get_cmap('YlOrRd').copy()
        cmap.set_bad(color='white')

        im = ax.imshow(
            combined,
            aspect='equal',
            cmap=cmap,
            vmin=0.,
            vmax=vmax,
            interpolation='nearest',
        )

        # ------------------------------------------------------------------ #
        # Font sizes — scale with figure size so labels stay readable        #
        # ------------------------------------------------------------------ #
        n_areas = len(area_tick_pos)
        # Points per area-block height on the saved figure
        tick_fontsize   = max(10, int(figsize / n_areas * 72 * 0.28))
        subnet_fontsize = max(12, int(tick_fontsize * 1.35))
        title_fontsize  = max(16, tick_fontsize + 8)
        cbar_fontsize   = max(12, tick_fontsize)

        # ------------------------------------------------------------------ #
        # Area tick labels                                                     #
        # ------------------------------------------------------------------ #
        ax.set_xticks(area_tick_pos)
        ax.set_xticklabels(area_tick_label, rotation=90, fontsize=tick_fontsize)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticks(area_tick_pos)
        ax.set_yticklabels(area_tick_label, rotation=0, fontsize=tick_fontsize)

        # ------------------------------------------------------------------ #
        # Colorbar — placed below the axes with a gap that clears rotated     #
        # x-tick labels (computed in inches so it's layout-independent)       #
        # ------------------------------------------------------------------ #
        import matplotlib.transforms as mtransforms
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        max_xlab_chars  = max(len(l) for l in area_tick_label)
        # Approximate height of a rotated-90° label in inches
        xlab_height_in  = tick_fontsize / 72.0 * max_xlab_chars * 0.6
        cbar_pad_in     = xlab_height_in + 0.1   # label height + gap

        divider = make_axes_locatable(ax)
        cbar_shrink = 0.5   # colorbar width as a fraction of the axes width
        cax = divider.append_axes('bottom', size='2%', pad=cbar_pad_in)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        # Centre the colorbar horizontally by shrinking cax
        p = cax.get_position()
        cax.set_position([
            p.x0 + p.width * (1 - cbar_shrink) / 2,
            p.y0,
            p.width * cbar_shrink,
            p.height,
        ])
        cbar.set_label('Coherence', fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=max(10, cbar_fontsize - 2))

        # ------------------------------------------------------------------ #
        # Subnetwork labels on the left, centred on each block                #
        # ------------------------------------------------------------------ #
        left_trans = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        # Compute x so the right edge of the longest label sits 2% outside
        # the axes left boundary, regardless of font size or figure size.
        max_subnet_chars = max(len(name) for name, _, _ in subnet_blocks)
        axes_width_est   = figsize * 0.82   # approx after tight_layout rect
        label_half_width = (subnet_fontsize / 72.0 * 0.6 * max_subnet_chars
                            / axes_width_est)
        subnet_label_x   = -(label_half_width + 0.005)
        for subnet_name, row_start, row_end in subnet_blocks:
            centre = (row_start + row_end) / 2.0
            ax.text(
                subnet_label_x, centre, subnet_name,
                transform=left_trans,
                ha='center', va='center',
                fontsize=subnet_fontsize, fontweight='bold',
                clip_on=False,
            )

        # ------------------------------------------------------------------ #
        # Separator lines                                                      #
        # ------------------------------------------------------------------ #
        subnet_boundary_set = set(subnet_boundaries)
        for b in area_boundaries:
            if b not in subnet_boundary_set:
                ax.axvline(x=b - 0.5, color='#C0C0C0', linewidth=2.)
                ax.axhline(y=b - 0.5, color='#C0C0C0', linewidth=2.)

        for b in subnet_boundaries:
            ax.axvline(x=b - 0.5, color='black', linewidth=3.)
            ax.axhline(y=b - 0.5, color='black', linewidth=3.)

        ax.set_title('LFP Coherence Matrix',
                     fontsize=title_fontsize, pad=title_fontsize * 1.2)

        # left_rect accounts for the actual width of the subnet labels so they
        # are never clipped by the figure boundary.
        left_rect = max(0.02, -subnet_label_x - label_half_width + 0.03)
        fig.tight_layout(rect=[left_rect, 0.04, 0.96, 0.94], pad=3.0)
        fname = f'lfp_coherence_fulllayer.{extension}'
        fig.savefig(os.path.join(self.plot_folder, fname), dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    @timeit
    def plotFullLayerLfpPlv(self):
        """
        Plot the full (area, layer) × (area, layer) PLV matrix, mirroring
        plotFullLayerLfpCoherence.

        All cross-area and cross-layer pairs are shown.  Rows and columns are
        ordered as (area, layer), with areas sorted by resting-state subnetwork
        (following left_ordering).  Thick black lines separate subnetworks;
        thin lines separate individual areas.  Axis tick labels show area names
        only, centred on each area's block.

        Saved as lfp_plv_fulllayer.<ext> in the plot folder.
        """
        import matplotlib.transforms as mtransforms
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        pkl_path = os.path.join(self.ana_folder, 'lfp_plv.pkl')
        try:
            plv = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            plv = self.calcLfpPlv()

        extension = self.ana_dict['extension']
        dpi       = self.ana_dict['dpi']
        vmax      = self.ana_dict['plotFullLayerLfpPlv']['vmax']
        cell_size = self.ana_dict['plotFullLayerLfpPlv']['cell_size_inches']

        # ------------------------------------------------------------------ #
        # Sort areas by subnetwork (following left_ordering key order)         #
        # ------------------------------------------------------------------ #
        all_areas     = plv.index.get_level_values('area').unique().tolist()
        ordered_areas = [a for a in left_ordering if a in all_areas]
        ordered_areas += [a for a in all_areas if a not in left_ordering]

        new_order = []
        for area in ordered_areas:
            new_order.extend([(a, l) for (a, l) in plv.index if a == area])
        plv     = plv.loc[new_order, new_order]

        index_pairs = list(plv.index)   # [(area, layer), ...]
        combined    = plv.values        # (N, N)
        n_total     = len(index_pairs)

        # ------------------------------------------------------------------ #
        # Boundary positions, tick positions, and subnetwork blocks           #
        # ------------------------------------------------------------------ #
        area_boundaries   = []
        subnet_boundaries = []
        area_tick_pos     = []
        area_tick_label   = []
        subnet_blocks     = []

        current_area     = None
        prev_subnet      = None
        block_start      = 0
        subnet_row_start = 0
        for i, (area, _layer) in enumerate(index_pairs):
            if area != current_area:
                if current_area is not None:
                    area_boundaries.append(i)
                    area_tick_pos.append((block_start + i) / 2.0)
                    area_tick_label.append(current_area)
                new_subnet = left_ordering.get(area, 'other')
                if prev_subnet is not None and new_subnet != prev_subnet:
                    subnet_boundaries.append(i)
                    subnet_blocks.append((prev_subnet, subnet_row_start, i))
                    subnet_row_start = i
                prev_subnet  = new_subnet
                block_start  = i
                current_area = area
        area_tick_pos.append((block_start + n_total) / 2.0)
        area_tick_label.append(current_area)
        subnet_blocks.append((prev_subnet, subnet_row_start, n_total))

        # ------------------------------------------------------------------ #
        # Figure                                                               #
        # ------------------------------------------------------------------ #
        figsize = max(12, n_total * cell_size)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        # PLV colormap — perceptually uniform, visually distinct from YlOrRd
        cmap = plt.get_cmap('viridis').copy()   # primary choice
        # Alternative colormaps (uncomment to use):
        # cmap = plt.get_cmap('plasma').copy()   # warm violet→yellow
        # cmap = plt.get_cmap('magma').copy()    # dark→light, good on white bg
        # cmap = plt.get_cmap('cividis').copy()  # accessible blue→yellow
        # cmap = plt.get_cmap('Blues').copy()    # sequential blue
        # cmap = plt.get_cmap('PuRd').copy()     # purple→red
        # cmap = plt.get_cmap('RdPu').copy()     # red→purple
        cmap.set_bad(color='white')

        im = ax.imshow(
            combined,
            aspect='equal',
            cmap=cmap,
            vmin=0.,
            vmax=vmax,
            interpolation='nearest',
        )

        # ------------------------------------------------------------------ #
        # Font sizes                                                           #
        # ------------------------------------------------------------------ #
        n_areas = len(area_tick_pos)
        tick_fontsize   = max(10, int(figsize / n_areas * 72 * 0.28))
        subnet_fontsize = max(12, int(tick_fontsize * 1.35))
        title_fontsize  = max(16, tick_fontsize + 8)
        cbar_fontsize   = max(12, tick_fontsize)

        # ------------------------------------------------------------------ #
        # Area tick labels                                                     #
        # ------------------------------------------------------------------ #
        ax.set_xticks(area_tick_pos)
        ax.set_xticklabels(area_tick_label, rotation=90, fontsize=tick_fontsize)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticks(area_tick_pos)
        ax.set_yticklabels(area_tick_label, rotation=0, fontsize=tick_fontsize)

        # ------------------------------------------------------------------ #
        # Colorbar                                                             #
        # ------------------------------------------------------------------ #
        max_xlab_chars = max(len(l) for l in area_tick_label)
        xlab_height_in = tick_fontsize / 72.0 * max_xlab_chars * 0.6
        cbar_pad_in    = xlab_height_in + 0.1

        divider = make_axes_locatable(ax)
        cbar_shrink = 0.5
        cax = divider.append_axes('bottom', size='2%', pad=cbar_pad_in)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        p = cax.get_position()
        cax.set_position([
            p.x0 + p.width * (1 - cbar_shrink) / 2,
            p.y0,
            p.width * cbar_shrink,
            p.height,
        ])
        cbar.set_label('n:m PLV', fontsize=cbar_fontsize)
        cbar.ax.tick_params(labelsize=max(10, cbar_fontsize - 2))

        # ------------------------------------------------------------------ #
        # Subnetwork labels on the left                                        #
        # ------------------------------------------------------------------ #
        left_trans = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData
        )
        max_subnet_chars = max(len(name) for name, _, _ in subnet_blocks)
        axes_width_est   = figsize * 0.82
        label_half_width = (subnet_fontsize / 72.0 * 0.6 * max_subnet_chars
                            / axes_width_est)
        subnet_label_x   = -(label_half_width + 0.005)
        for subnet_name, row_start, row_end in subnet_blocks:
            centre = (row_start + row_end) / 2.0
            ax.text(
                subnet_label_x, centre, subnet_name,
                transform=left_trans,
                ha='center', va='center',
                fontsize=subnet_fontsize, fontweight='bold',
                clip_on=False,
            )

        # ------------------------------------------------------------------ #
        # Separator lines                                                      #
        # ------------------------------------------------------------------ #
        subnet_boundary_set = set(subnet_boundaries)
        for b in area_boundaries:
            if b not in subnet_boundary_set:
                ax.axvline(x=b - 0.5, color='#C0C0C0', linewidth=2.)
                ax.axhline(y=b - 0.5, color='#C0C0C0', linewidth=2.)
        for b in subnet_boundaries:
            ax.axvline(x=b - 0.5, color='black', linewidth=3.)
            ax.axhline(y=b - 0.5, color='black', linewidth=3.)

        n, m, theta_band, beta_band = self._thetaBetaNM()
        ax.set_title(
            f'Cross-frequency LFP PLV  (n:m = {n}:{m})\n'
            f'rows: θ {theta_band[0]:g}–{theta_band[1]:g} Hz   '
            f'cols: β {beta_band[0]:g}–{beta_band[1]:g} Hz',
            fontsize=title_fontsize, pad=title_fontsize * 1.2)

        left_rect = max(0.02, -subnet_label_x - label_half_width + 0.03)
        fig.tight_layout(rect=[left_rect, 0.04, 0.96, 0.94], pad=3.0)
        fname = f'lfp_plv_fulllayer.{extension}'
        fig.savefig(os.path.join(self.plot_folder, fname), dpi=dpi,
                    bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    @timeit
    def plotAllLfpProxyPSD(self):
        """
        Plot power spectrum of the LFP proxy for each area, grouped by layer.
        Produces one figure per layer saved to the plot folder.
        """
        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    pass
            else:
                raise FileNotFoundError(
                    'plotAllLfpProxyPSD: neither lfp_proxy.pkl nor '
                    'lfp_proxy_kernel.pkl found in analysis folder.')

        tmin = self.ana_dict['plotAllLfpProxyPSD']['tmin']
        fmax = self.ana_dict['plotAllLfpProxyPSD']['fmax']
        num_col = self.ana_dict['plotAllLfpProxyPSD']['num_col']

        z = self.lfp.loc[:, self.lfp.columns >= tmin]
        t = z.columns.values
        fs = 1000.0 / (t[1] - t[0])  # ms -> Hz
        # None -> full-window Welch segment, the finest resolution the data
        # allows (see calc_power_spectrum); override to trade it for variance.
        nperseg = self.ana_dict['plotAllLfpProxyPSD'].get('nperseg')

        layers = z.index.get_level_values('layer').unique()

        for layer in layers:
            z_layer = z.xs(layer, level='layer')
            areas = z_layer.index
            num_row = math.ceil(len(areas) / num_col)

            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(num_row, num_col, figsize=(22, 22))
            axes = np.array(axes).reshape(-1)

            for num, area in enumerate(areas):
                ax = axes[num]
                freqs, power = calc_power_spectrum(
                    z_layer.loc[area].values, fs, nperseg=nperseg)
                mask = freqs <= fmax
                ax.semilogy(freqs[mask], power[mask], color='black', linewidth=1.5)
                ax.set_xlim(0, fmax)
                if num < (num_row - 1) * num_col:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Frequency (Hz)')
                if num % num_col != 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel('Power')
                ax.set_title(area, loc='left', fontsize=20, fontweight=0, color='black')

            for num in range(len(areas), len(axes)):
                axes[num].set_visible(False)

            fig.suptitle(f'LFP Proxy Power Spectrum — Layer {layer}', fontsize=22)
            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                f'lfp_proxy_psd_layer_{layer.replace("/", "_")}.png'
            ), dpi=self.ana_dict['dpi'])
            plt.clf()
            plt.close(fig)
            plt.style.use('default')

    @timeit
    def plotAreaLfpPsd(self):
        """
        Plot the power spectrum of the ROI (area-level) LFP proxy, one panel
        per area, in a single figure spanning [0, fmax] Hz.

        Reads lfp_proxy_area.pkl (computed by calcAreaLfp); computes it if absent.
        Saved as lfp_proxy_area_psd.<ext> in the plot folder.
        """
        pkl_path = os.path.join(self.ana_folder, 'lfp_proxy_area.pkl')
        try:
            roi = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            roi = self.calcAreaLfp()

        tmin      = self.ana_dict['plotAreaLfpPsd']['tmin']
        fmax      = self.ana_dict['plotAreaLfpPsd']['fmax']
        num_col   = self.ana_dict['plotAreaLfpPsd']['num_col']
        # None -> full-window Welch segment, the finest resolution the data
        # allows (see calc_power_spectrum); override to trade it for variance.
        nperseg   = self.ana_dict['plotAreaLfpPsd'].get('nperseg')
        extension = self.ana_dict['extension']

        z  = roi.loc[:, roi.columns >= tmin]
        t  = z.columns.values
        fs = 1000.0 / (t[1] - t[0])  # ms -> Hz

        areas   = z.index.tolist()
        num_row = math.ceil(len(areas) / num_col)

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(num_row, num_col,
                                 figsize=(num_col * 3.5, num_row * 2.5))
        axes = np.array(axes).reshape(-1)

        for num, area in enumerate(areas):
            ax = axes[num]
            freqs, power = calc_power_spectrum(
                z.loc[area].values, fs, nperseg=nperseg)
            mask = freqs <= fmax
            ax.semilogy(freqs[mask], power[mask], color='black', linewidth=1.2)
            ax.set_xlim(0, fmax)
            if num < (num_row - 1) * num_col:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Frequency (Hz)')
            if num % num_col != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel('Power')
            ax.set_title(area, loc='left', fontsize=12, color='black')

        for num in range(len(areas), len(axes)):
            axes[num].set_visible(False)

        fig.suptitle(f'ROI LFP Proxy Power Spectrum  (0–{fmax:g} Hz)', fontsize=18)
        fig.tight_layout()
        plt.savefig(os.path.join(self.plot_folder, f'lfp_proxy_area_psd.{extension}'),
                    dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    @timeit
    def plotAreaLfpCorrelation(self):
        """
        Plot the ROI x ROI LFP correlation matrix (functional connectivity) as
        a heatmap.  Areas are ordered by resting-state subnetwork; thick black
        lines separate subnetworks.  A diverging colormap is centred at zero.

        Reads lfp_area_correlation.pkl (computed by calcAreaLfpCorrelation;
        computed on demand if absent).  Saved as lfp_area_correlation.<ext>.
        """
        import matplotlib.transforms as mtransforms
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        pkl_path = os.path.join(self.ana_folder, 'lfp_area_correlation.pkl')
        try:
            corr = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            corr = self.calcAreaLfpCorrelation()

        extension = self.ana_dict['extension']
        dpi       = self.ana_dict['dpi']
        cfg       = self.ana_dict['plotAreaLfpCorrelation']
        vmax      = cfg['vmax']
        cmap_name = cfg.get('cmap', 'RdBu_r')
        cell_size = cfg.get('cell_size_inches', 0.4)

        # Areas are already subnet-ordered in the pkl; recompute boundaries.
        areas   = corr.index.tolist()
        n_total = len(areas)

        subnet_boundaries = []          # index where a new subnet begins
        subnet_blocks     = []          # (subnet_name, start, end)
        prev_subnet, start = None, 0
        for i, area in enumerate(areas):
            sn = left_ordering.get(area, 'other')
            if prev_subnet is not None and sn != prev_subnet:
                subnet_boundaries.append(i)
                subnet_blocks.append((prev_subnet, start, i))
                start = i
            prev_subnet = sn
        subnet_blocks.append((prev_subnet, start, n_total))

        figsize = max(10, n_total * cell_size)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color='white')
        im = ax.imshow(corr.values, aspect='equal', cmap=cmap,
                       vmin=-vmax, vmax=vmax, interpolation='nearest')

        tick_fontsize  = max(7, int(figsize / max(n_total, 1) * 72 * 0.30))
        title_fontsize = max(16, tick_fontsize + 8)

        ax.set_xticks(range(n_total))
        ax.set_xticklabels(areas, rotation=90, fontsize=tick_fontsize)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticks(range(n_total))
        ax.set_yticklabels(areas, fontsize=tick_fontsize)

        # Thin lines between every area, thick lines at subnet boundaries
        subnet_set = set(subnet_boundaries)
        for b in range(1, n_total):
            if b not in subnet_set:
                ax.axvline(x=b - 0.5, color='#D0D0D0', linewidth=0.6)
                ax.axhline(y=b - 0.5, color='#D0D0D0', linewidth=0.6)
        for b in subnet_boundaries:
            ax.axvline(x=b - 0.5, color='black', linewidth=2.5)
            ax.axhline(y=b - 0.5, color='black', linewidth=2.5)

        # Subnetwork labels on the left
        left_trans = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        for sname, s_start, s_end in subnet_blocks:
            ax.text(-0.08, (s_start + s_end) / 2.0, sname,
                    transform=left_trans, ha='right', va='center',
                    fontsize=max(9, tick_fontsize + 1), fontweight='bold',
                    clip_on=False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=0.6)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Pearson r', fontsize=max(12, tick_fontsize))

        ax.set_title('ROI LFP Correlation (functional connectivity)',
                     fontsize=title_fontsize, pad=title_fontsize)
        fig.tight_layout(rect=[0.06, 0.0, 0.96, 0.96])
        fig.savefig(os.path.join(self.plot_folder, f'lfp_area_correlation.{extension}'),
                    dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    @timeit
    def plotAreaLfpCoherence(self):
        """
        Plot the ROI x ROI LFP coherence matrix as a heatmap.  Areas are
        ordered by resting-state subnetwork; thick black lines separate
        subnetworks, thin grey lines separate individual areas.  A sequential
        colormap maps coherence in [0, vmax].

        Mirrors plotAreaLfpCorrelation (same area-level layout) but for the
        frequency-domain coherence rather than Pearson correlation, so the two
        ROI functional-connectivity views can be compared side by side.

        Reads lfp_area_coherence.pkl (computed by calcAreaLfpCoherence; computed
        on demand if absent).  Saved as lfp_area_coherence.<ext>.
        """
        import matplotlib.transforms as mtransforms
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        pkl_path = os.path.join(self.ana_folder, 'lfp_area_coherence.pkl')
        try:
            coh = pd.read_pickle(pkl_path)
        except FileNotFoundError:
            coh = self.calcAreaLfpCoherence()

        extension = self.ana_dict['extension']
        dpi       = self.ana_dict['dpi']
        cfg       = self.ana_dict['plotAreaLfpCoherence']
        vmax      = cfg['vmax']
        cmap_name = cfg.get('cmap', 'YlOrRd')
        cell_size = cfg.get('cell_size_inches', 0.4)

        # Areas are already subnet-ordered in the pkl; recompute boundaries.
        areas   = coh.index.tolist()
        n_total = len(areas)

        subnet_boundaries = []          # index where a new subnet begins
        subnet_blocks     = []          # (subnet_name, start, end)
        prev_subnet, start = None, 0
        for i, area in enumerate(areas):
            sn = left_ordering.get(area, 'other')
            if prev_subnet is not None and sn != prev_subnet:
                subnet_boundaries.append(i)
                subnet_blocks.append((prev_subnet, start, i))
                start = i
            prev_subnet = sn
        subnet_blocks.append((prev_subnet, start, n_total))

        figsize = max(10, n_total * cell_size)
        fig, ax = plt.subplots(figsize=(figsize, figsize))

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color='white')
        im = ax.imshow(coh.values, aspect='equal', cmap=cmap,
                       vmin=0., vmax=vmax, interpolation='nearest')

        tick_fontsize  = max(7, int(figsize / max(n_total, 1) * 72 * 0.30))
        title_fontsize = max(16, tick_fontsize + 8)

        ax.set_xticks(range(n_total))
        ax.set_xticklabels(areas, rotation=90, fontsize=tick_fontsize)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.set_yticks(range(n_total))
        ax.set_yticklabels(areas, fontsize=tick_fontsize)

        # Thin lines between every area, thick lines at subnet boundaries
        subnet_set = set(subnet_boundaries)
        for b in range(1, n_total):
            if b not in subnet_set:
                ax.axvline(x=b - 0.5, color='#D0D0D0', linewidth=0.6)
                ax.axhline(y=b - 0.5, color='#D0D0D0', linewidth=0.6)
        for b in subnet_boundaries:
            ax.axvline(x=b - 0.5, color='black', linewidth=2.5)
            ax.axhline(y=b - 0.5, color='black', linewidth=2.5)

        # Subnetwork labels on the left
        left_trans = mtransforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        for sname, s_start, s_end in subnet_blocks:
            ax.text(-0.08, (s_start + s_end) / 2.0, sname,
                    transform=left_trans, ha='right', va='center',
                    fontsize=max(9, tick_fontsize + 1), fontweight='bold',
                    clip_on=False)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=0.6)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_label('Coherence', fontsize=max(12, tick_fontsize))

        fmin = self.ana_dict['calcAreaLfpCoherence']['fmin']
        fmax = self.ana_dict['calcAreaLfpCoherence']['fmax']
        ax.set_title(f'ROI LFP Coherence  ({fmin:g}–{fmax:g} Hz)',
                     fontsize=title_fontsize, pad=title_fontsize)
        fig.tight_layout(rect=[0.06, 0.0, 0.96, 0.96])
        fig.savefig(os.path.join(self.plot_folder, f'lfp_area_coherence.{extension}'),
                    dpi=dpi, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    @timeit
    def plotLfpArea(self, area):
        """
        Plot z-scored LFP proxy time series for each layer in a given area.
        Layers are stacked vertically with a shared time axis.

        Parameters
        ----------
        area : str
        """
        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    pass
            else:
                raise FileNotFoundError(
                    'plotLfpArea: neither lfp_proxy.pkl nor '
                    'lfp_proxy_kernel.pkl found in analysis folder.')
        tmin = self.ana_dict['plotLfpArea']['tmin']
        extension = self.ana_dict['extension']

        z = self.lfp.loc[:, self.lfp.columns >= tmin]
        z_area = z.xs(area, level='area')
        layers = z_area.index.tolist()
        t = z_area.columns.values * 1e-3  # ms -> s

        n_layers = len(layers)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 2 * n_layers), sharex=True)
        if n_layers == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            ax.plot(t, z_area.loc[layer].values, color='black', linewidth=1.2)
            ax.set_ylabel(layer, fontsize=11)
            ax.set_xlim(t[0], t[-1])

        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'{area.title()} — LFP Proxy', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(
            self._category_plot_folder('lfp'),
            f'lfp_{area}.{extension}'
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    @timeit
    def plotLfpPsdArea(self, area):
        """
        Plot power spectrum of the LFP proxy for each layer in a given area.
        Layers are stacked vertically with a shared frequency axis.

        Parameters
        ----------
        area : str
        """
        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    pass
            else:
                raise FileNotFoundError(
                    'plotLfpPsdArea: neither lfp_proxy.pkl nor '
                    'lfp_proxy_kernel.pkl found in analysis folder.')
        tmin = self.ana_dict['plotLfpPsdArea']['tmin']
        fmax = self.ana_dict['plotLfpPsdArea']['fmax']
        extension = self.ana_dict['extension']

        z = self.lfp.loc[:, self.lfp.columns >= tmin]
        fs = 1000.0 / (z.columns.values[1] - z.columns.values[0])  # ms -> Hz
        # None -> full-window Welch segment, the finest resolution the data
        # allows (see calc_power_spectrum); override to trade it for variance.
        nperseg = self.ana_dict['plotLfpPsdArea'].get('nperseg')

        z_area = z.xs(area, level='area')
        layers = z_area.index.tolist()

        n_layers = len(layers)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 2 * n_layers), sharex=True)
        if n_layers == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            freqs, power = calc_power_spectrum(
                z_area.loc[layer].values, fs, nperseg=nperseg)
            mask = freqs <= fmax
            ax.semilogy(freqs[mask], power[mask], color='black', linewidth=1.2)
            ax.set_ylabel(layer, fontsize=11)
            ax.set_xlim(0, fmax)

        axes[-1].set_xlabel('Frequency (Hz)', fontsize=12)
        fig.suptitle(f'{area.title()} — LFP Proxy PSD', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(
            self._category_plot_folder('psd'),
            f'lfp_psd_{area}.{extension}'
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    @timeit
    def plotLfpSpectrogramArea(self, area):
        """
        Plot the LFP spectrogram (time-frequency representation) for each layer
        in a given area.  Layers are stacked vertically with a shared time axis.

        Power is shown in dB using a 2-D colour map.

        Configuration keys in ``ana_dict['plotLfpSpectrogramArea']``
        -------------------------------------------------------------
        tmin       : float - analysis start time in ms (default 500).
        fmin       : float - lower frequency bound in Hz (default 1.0).
        fmax       : float - upper frequency bound in Hz (default 100.0).
        nperseg    : int   - STFT window length in samples (default 256).
        noverlap   : int   - overlap between windows in samples
                             (default ``nperseg // 2``).
        db_range   : float - dynamic range in dB shown below the peak
                             (default 60.0; ``None`` uses full range).
        show_bands : bool  - draw dashed lines at canonical band boundaries
                             (default ``True``).

        Parameters
        ----------
        area : str
        """
        from scipy.signal import spectrogram as scipy_spectrogram

        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    pass
            else:
                raise FileNotFoundError(
                    'plotLfpSpectrogramArea: neither lfp_proxy.pkl nor '
                    'lfp_proxy_kernel.pkl found in analysis folder.')

        cfg        = self.ana_dict.get('plotLfpSpectrogramArea', {})
        tmin       = cfg.get('tmin', 500.)
        fmin       = cfg.get('fmin', 1.0)
        fmax       = cfg.get('fmax', 100.)
        nperseg    = cfg.get('nperseg', 256)
        noverlap   = cfg.get('noverlap', None)
        # Colour scheme for the spectrogram — swap in any of the alternatives:
        # 'inferno'   dark background, bright highlights          (default)
        # 'viridis'   sequential blue → green → yellow
        # 'plasma'    sequential purple → orange
        # 'magma'     sequential black → purple → white
        # 'hot'       classic heat-map (black → red → yellow → white)
        # 'bone'      greyscale + slight blue tint; publication-safe
        # 'turbo'     improved rainbow with better perceptual uniformity
        # 'RdYlBu_r'  diverging warm/cold; suited for signed residuals
        scheme     = 'viridis'
        db_range   = cfg.get('db_range', 60.0)
        show_bands = cfg.get('show_bands', True)
        extension  = self.ana_dict.get('extension', 'png')

        z     = self.lfp.loc[:, self.lfp.columns >= tmin]
        t_ms  = z.columns.values
        fs    = 1000.0 / (t_ms[1] - t_ms[0])  # ms → Hz

        n_samples = z.shape[1]

        # Ideal nperseg to resolve fmin, rounded up to next power of two.
        nperseg_ideal = int(2 ** np.ceil(np.log2(fs / fmin))) if fmin > 0 else nperseg
        # Hard cap: at most n_samples // 4 to guarantee ≥ 4 time segments.
        nperseg_max = max(n_samples // 4, 1)
        nperseg = min(max(nperseg, nperseg_ideal), nperseg_max)

        fmin_actual = fs / nperseg
        if fmin_actual > fmin and self.logger:
            self.logger.warning(
                'plotLfpSpectrogramArea: signal too short to resolve fmin=%.3f Hz; '
                'effective fmin=%.2f Hz (nperseg=%d, n_samples=%d). '
                'Increase t_sim to reach lower frequencies.',
                fmin, fmin_actual, nperseg, n_samples)

        if noverlap is None:
            noverlap = nperseg // 2
        else:
            noverlap = min(noverlap, nperseg - 1)

        # Canonical frequency bands: (f_lo Hz, f_hi Hz, label)
        _BANDS = [
            (0.5,  4,   'δ'),
            (4,    8,   'θ'),
            (8,    12,  'α'),
            (12,   30,  'β'),
            (30,   80,  'γ'),
            (80,   200, 'hγ'),
        ]

        z_area = z.xs(area, level='area')
        layers = z_area.index.tolist()

        n_layers = len(layers)
        fig, axes = plt.subplots(
            n_layers, 1,
            figsize=(10, 2.8 * n_layers),
            sharex=True)
        if n_layers == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            sig = z_area.loc[layer].values.astype(float)

            freqs, t_seg, Sxx = scipy_spectrogram(
                sig, fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann',
                scaling='density')

            # Convert to dB and restrict to [fmin, fmax]
            Sxx_db = 10.0 * np.log10(np.maximum(Sxx, 1e-30))
            f_mask      = (freqs >= fmin) & (freqs <= fmax)
            freqs_plot  = freqs[f_mask]
            Sxx_plot    = Sxx_db[f_mask, :]

            vmax = Sxx_plot.max()
            vmin = (vmax - db_range) if db_range is not None else Sxx_plot.min()

            # Absolute time axis in seconds
            t_abs_s = (t_ms[0] + t_seg * 1000.0) * 1e-3

            im = ax.pcolormesh(
                t_abs_s, freqs_plot, Sxx_plot,
                cmap=scheme,
                vmin=vmin, vmax=vmax,
                shading='gouraud',
                rasterized=True)

            if show_bands:
                for f_lo, f_hi, band_name in _BANDS:
                    if fmin < f_lo < fmax:
                        ax.axhline(
                            f_lo, color='white',
                            linewidth=0.6, linestyle='--', alpha=0.6)
                    label_lo = max(f_lo, fmin)
                    label_hi = min(f_hi, fmax)
                    if label_lo < label_hi:
                        f_mid = np.sqrt(label_lo * label_hi)
                        ax.text(
                            t_abs_s[-1], f_mid, f' {band_name}',
                            va='center', ha='left',
                            fontsize=7, color='white', alpha=0.8,
                            clip_on=False)

            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            ax.set_ylim(fmin, fmax)
            ax.set_title(layer, loc='left', fontsize=10)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label('Power (dB)', fontsize=8)
            cb.ax.tick_params(labelsize=7)

        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'{area.title()} — LFP Spectrogram', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(
            self._category_plot_folder('spectrogram'),
            f'lfp_spectrogram_{area}.{extension}'
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    @timeit
    def plotLfpWaveletArea(self, area):
        """
        Plot the LFP Morlet wavelet scalogram for each layer in a given area.
        Layers are stacked vertically with a shared time axis.

        Power is shown in dB using a 2-D colour map.  Compared to
        :meth:`plotLfpSpectrogramArea` (STFT), the frequency axis is
        logarithmically spaced, giving finer frequency resolution at low
        frequencies and finer time resolution at high frequencies.

        Configuration keys in ``ana_dict['plotLfpWaveletArea']``
        ---------------------------------------------------------
        tmin       : float - analysis start time in ms (default 500).
        fmin       : float - lower frequency bound in Hz (default 1.0).
        fmax       : float - upper frequency bound in Hz (default 100.0).
        n_scales   : int   - number of log-spaced frequency scales (default 64).
        w          : float - Morlet wavelet parameter; higher values increase
                             frequency resolution at the cost of time resolution
                             (default 6.0).
        db_range   : float - dynamic range in dB shown below the peak
                             (default 60.0; ``None`` uses full range).
        show_bands : bool  - draw dashed lines at canonical band boundaries
                             (default ``True``).

        Parameters
        ----------
        area : str
        """
        if not hasattr(self, 'lfp'):
            for pkl_name in ('lfp_proxy.pkl', 'lfp_proxy_kernel.pkl'):
                pkl_path = os.path.join(self.ana_folder, pkl_name)
                try:
                    self.lfp = pd.read_pickle(pkl_path)
                    break
                except FileNotFoundError:
                    pass
            else:
                raise FileNotFoundError(
                    'plotLfpWaveletArea: neither lfp_proxy.pkl nor '
                    'lfp_proxy_kernel.pkl found in analysis folder.')

        cfg        = self.ana_dict.get('plotLfpWaveletArea', {})
        tmin       = cfg.get('tmin', 500.)
        fmin       = cfg.get('fmin', 1.0)
        fmax       = cfg.get('fmax', 100.)
        n_scales   = cfg.get('n_scales', 64)
        w          = cfg.get('w', 6.0)
        db_range   = cfg.get('db_range', 60.0)
        show_bands = cfg.get('show_bands', True)
        extension  = self.ana_dict.get('extension', 'png')

        # Colour scheme for the scalogram — swap in any of the alternatives:
        # 'inferno'   dark background, bright highlights          (default)
        # 'viridis'   sequential blue → green → yellow
        # 'plasma'    sequential purple → orange
        # 'magma'     sequential black → purple → white
        # 'hot'       classic heat-map (black → red → yellow → white)
        # 'bone'      greyscale + slight blue tint; publication-safe
        # 'turbo'     improved rainbow with better perceptual uniformity
        # 'RdYlBu_r'  diverging warm/cold; suited for signed residuals
        scheme = 'inferno'

        # Canonical frequency bands: (f_lo Hz, f_hi Hz, label)
        _BANDS = [
            (0.5,  4,   'δ'),
            (4,    8,   'θ'),
            (8,    12,  'α'),
            (12,   30,  'β'),
            (30,   80,  'γ'),
            (80,   200, 'hγ'),
        ]

        z    = self.lfp.loc[:, self.lfp.columns >= tmin]
        t_ms = z.columns.values
        fs   = 1000.0 / (t_ms[1] - t_ms[0])  # ms → Hz

        z_area = z.xs(area, level='area')
        layers = z_area.index.tolist()

        n_layers = len(layers)
        fig, axes = plt.subplots(
            n_layers, 1,
            figsize=(10, 2.8 * n_layers),
            sharex=True)
        if n_layers == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            sig = z_area.loc[layer].values.astype(float)

            freqs, t_rel, power = calc_wavelet_scalogram(
                sig, fs, fmin=fmin, fmax=fmax, n_scales=n_scales, w=w)

            power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
            vmax = power_db.max()
            vmin = (vmax - db_range) if db_range is not None else power_db.min()

            t_abs_s = t_ms[0] * 1e-3 + t_rel

            im = ax.pcolormesh(
                t_abs_s, freqs, power_db,
                cmap=scheme,
                vmin=vmin, vmax=vmax,
                shading='gouraud',
                rasterized=True)

            ax.set_yscale('log')
            ax.set_ylim(fmin, fmax)

            if show_bands:
                for f_lo, f_hi, band_name in _BANDS:
                    if fmin < f_lo < fmax:
                        ax.axhline(
                            f_lo, color='white',
                            linewidth=0.6, linestyle='--', alpha=0.6)
                    label_lo = max(f_lo, fmin)
                    label_hi = min(f_hi, fmax)
                    if label_lo < label_hi:
                        f_mid = np.sqrt(label_lo * label_hi)
                        ax.text(
                            t_abs_s[-1], f_mid, f' {band_name}',
                            va='center', ha='left',
                            fontsize=7, color='white', alpha=0.8,
                            clip_on=False)

            ax.set_ylabel('Frequency (Hz)', fontsize=11)
            ax.set_title(layer, loc='left', fontsize=10)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label('Power (dB)', fontsize=8)
            cb.ax.tick_params(labelsize=7)

        axes[-1].set_xlabel('Time (s)', fontsize=12)
        fig.suptitle(f'{area.title()} — LFP Wavelet Scalogram', fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(
            self._category_plot_folder('wavelet'),
            f'lfp_wavelet_{area}.{extension}'
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    def _load_experimental_fc(self):
        """
        Load the experimental functional connectivity selected by
        ``ana_dict['experimental_fc']``.

        The sentinel value ``'rsDATA_7T_DKparcel.npy'`` (the default) selects
        the group-averaged Senden 7T resting-state data shipped in
        ``experimental_data/senden/rsData_7T_DKparcel/``; the per-participant
        correlation matrices are averaged over all participants. Any other
        value is taken as a path to a single subject's ``.npy`` file and is
        read through :meth:`personalFunctionalConnectivity`.

        Returns
        -------
        exp_fc : dict or None
            ``{'lh': DataFrame, 'rh': DataFrame}``, each 34x34, or None if the
            experimental data could not be found.
        areas : ndarray or None
            The 34 Desikan Killiany area names, or None alongside exp_fc.
        """
        data_dir = os.path.join(
            self.base_path, 'experimental_data/senden/rsData_7T_DKparcel/')
        roi_path = os.path.join(data_dir, 'ROIs.txt')

        selected = self.ana_dict['experimental_fc']
        group_average = (selected == 'rsDATA_7T_DKparcel.npy')
        source = os.path.join(data_dir, selected) if group_average else selected

        if not (os.path.exists(roi_path) and os.path.exists(source)):
            if self.logger:
                self.logger.warning(
                    'No experimental data found (%s). Only simulated data '
                    'will be plotted.', source)
            return None, None

        # Read in regions of interest.
        roi = pd.read_csv(
            roi_path, header=None, names=['roi'], dtype=str
        ).squeeze()
        # The rois are given in this manner: ctx-lh-bankssts
        # The name of the area is the last word after -
        roi = roi.apply(lambda x: x.split('-')[-1])
        # The cortical areas are in the range from 14 to 82
        roi = roi.drop(range(0, 14)).drop(range(82, 85))
        # Desikan Killiany area names from the left hemisphere, i.e. 0:34
        areas = roi.values[:34]

        if not group_average:
            return self.personalFunctionalConnectivity(source, areas), areas

        # Read in the bold signal. BOLD.shape = (600, 85, 19)
        # First dimension: timesteps
        # Second dimension: Desikan Killiany areas, left (0:34) and right
        # (34:68) hemisphere
        # Third dimension: Participants
        # orientation discrimination, numerosity
        BOLD = np.load(source)
        BOLD = BOLD[:, 14:82, :]

        # There are 600 timesteps in 1.5 second steps in the data
        resolution = 1.5
        timesteps = np.arange(600) * resolution

        # Extract the number of persons
        no_of_persons = BOLD.shape[2]

        # Extract BOLD series into a dictionary of DataFrames
        exp_fc = {'lh': np.zeros((34, 34)), 'rh': np.zeros((34, 34))}

        # Loop over all persons
        for person in range(no_of_persons):
            # Left hemisphere is 0:34, right hemisphere is 34:68
            lh_person = BOLD[:, 0:34, person]
            rh_person = BOLD[:, 34:68, person]

            # BOLD signal into DataFrames
            lh = pd.DataFrame(lh_person, index=timesteps, columns=areas)
            rh = pd.DataFrame(rh_person, index=timesteps, columns=areas)

            # Correlations of all columns, i.e. areas, with each other
            exp_fc['lh'] += lh.corr()
            exp_fc['rh'] += rh.corr()

        exp_fc['lh'] /= no_of_persons
        exp_fc['rh'] /= no_of_persons
        return exp_fc, areas

    @timeit
    def plot_functional_connectivity(self, save_fig=False, show_similarity_in_title=True):
        """
        Plot the functional connectivity of the network based on synaptic input
        currents and compare it with experimental BOLD data if available.

        The experimental side is selected automatically by
        ``ana_dict['experimental_fc']``: the default sentinel
        ``'rsDATA_7T_DKparcel.npy'`` uses the group-averaged Senden 7T data,
        any other value is read as a single subject's ``.npy`` file (see
        :meth:`_load_experimental_fc`). Both cases produce identically styled
        figures.

        Writes ``simulated_synaptic_currents_correlation.<ext>`` (experimental
        next to simulated), ``simulated_fc.<ext>`` and, when experimental data
        is available, ``experimental_fc.<ext>``.

        Parameters
        ----------
        save_fig : bool, optional
            If True, the figures will be saved to the plot folder. Default is
            False.
        show_similarity_in_title : bool, optional
            If True (and experimental data is available), add a suptitle to
            simulated_synaptic_currents_correlation.<ext> reporting the Pearson
            correlation and cosine similarity between the vectorized upper
            triangles of the experimental and simulated FC matrices (see
            _fc_similarity). Default is True.
        """
        exp_fc, _ = self._load_experimental_fc()
        exp_data_exists = exp_fc is not None

        # Load clustering information
        clustering = pd.Series(left_ordering)
        ordering = clustering.keys()
        tmp_lines = clustering.values
        lines_border = np.where(tmp_lines[:-1] != tmp_lines[1:])[0]
        lines = lines_border + 1
        extended_lines = np.append(lines_border, np.array([len(left_ordering) - 1]))
        tmp = np.append(np.array([0]), extended_lines)
        points = (tmp[1:] + tmp[:-1]) * .5 + 1
        texts = clustering.iloc[extended_lines].values

        def _draw(ax, matrix, title):
            im = ax.pcolormesh(matrix.loc[ordering][ordering],
                               vmin=-1, vmax=1, cmap='RdYlBu_r')
            ax.set_title(title)
            ax.set_xticks(points)
            ax.set_xticklabels(texts, rotation='vertical')
            ax.set_yticks(points)
            ax.set_yticklabels(texts)
            ax.axis('square')
            ax.invert_yaxis()
            ax.hlines(lines, *ax.get_xlim(), color='k')
            ax.vlines(lines, *ax.get_xlim(), color='k')
            return im

        # Figure settings
        plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
        nrows, ncols = (1, 2) if exp_data_exists else (1, 1)
        width, panel_wh_ratio = 5.63, 1.5
        height = width / panel_wh_ratio * float(nrows) / ncols
        # 并列图 subplot
        fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        # 单独图：模拟 FC 和实验 FC
        sim_fig, sim_axes = plt.subplots(1, 1, figsize=(3, 2))
        exp_fig, exp_axes = plt.subplots(1, 1, figsize=(3, 2))

        if exp_data_exists:
            # Plot experimental FC on both: axes[0] and exp_axes. Both figures
            # are saved, so each has to be drawn on directly.
            for ax in [axes[0], exp_axes]:
                im = _draw(ax, exp_fc['rh'], 'Experimental FC')
                cbar = ax.figure.colorbar(im, ax=ax, ticks=[-1, 0, 1])
                cbar.set_label('Correlation', rotation=270, labelpad=15)

        # Simulated FC computation
        curr_in = self.synapticInputCurrent()
        synaptic_currents = curr_in.T
        initial_transient = 500.
        tmax = self.sim_dict['t_sim']
        simulated_fc = synaptic_currents[
            (synaptic_currents.index >= initial_transient) & (synaptic_currents.index <= tmax)
        ].corr()

        # Plot simulated FC on both: axes[-1] and sim_axes
        for ax in [axes[-1], sim_axes]:
            im = _draw(ax, simulated_fc, 'Simulated FC')

        cbar = sim_fig.colorbar(im, ax=sim_axes, ticks=[-1, 0, 1])
        cbar.set_label('Correlation', rotation=270, labelpad=15)

        if show_similarity_in_title and exp_data_exists:
            pearson_r, cosine_sim = _fc_similarity(
                exp_fc['rh'].loc[ordering][ordering], simulated_fc.loc[ordering][ordering]
            )
            fig.suptitle(f'Sim vs Exp FC — Pearson r={pearson_r:.3f}, Cosine={cosine_sim:.3f}')

        # Save all three figures
        if save_fig:
            extension = self.ana_dict['extension']

            # 并列图
            fig.savefig(os.path.join(
                self.plot_folder,
                f'simulated_synaptic_currents_correlation.{extension}'
            ), dpi=self.ana_dict['dpi'])
            plt.close(fig)

            # 单独模拟图
            sim_fig.savefig(os.path.join(
                self.plot_folder,
                f'simulated_fc.{extension}'
            ), dpi=self.ana_dict['dpi'])
            plt.close(sim_fig)

            # 单独实验图
            if exp_data_exists:
                exp_fig.savefig(os.path.join(
                    self.plot_folder,
                    f'experimental_fc.{extension}'
                ), dpi=self.ana_dict['dpi'])
                plt.close(exp_fig)

        # Restore the ambient style: plt.style.use() mutates rcParams globally,
        # so without this the report style (and its dpi) leaks into every figure
        # drawn afterwards.
        plt.style.use('default')

    @staticmethod
    def personalFunctionalConnectivity(path, areas):
        """
        Load one subject's experimental functional connectivity.

        Two node layouts are recognised, and which one applies is decided by
        the node count rather than assumed:

        * 68 nodes - the cortical Desikan Killiany areas only, left hemisphere
          0:34 and right hemisphere 34:68, as in ``experimental_data/senden``
          and ``experimental_data/adni-adcn``;
        * 84 nodes - the full MRtrix ``fs_default`` DK-84 parcellation, where
          the subcortical block sits between the hemispheres, putting the
          cortex at 0:34 and 49:83.

        Either may be stored as a BOLD time series, ``(timesteps, nodes)`` or
        ``(timesteps, nodes, participants)``, from which the correlation
        matrix is computed (averaged over participants), or as an
        already-computed ``(nodes, nodes)`` connectivity matrix.

        Parameters
        ----------
        path : string
            Path to the ``.npy`` file.
        areas : sequence
            The 34 Desikan Killiany area names, used to label the result.

        Returns
        -------
        exp_fc : dict
            ``{'lh': DataFrame, 'rh': DataFrame}``, each 34x34.
        """
        data = np.load(path)
        n_nodes = data.shape[1]
        cortex_only = {'lh': slice(0, 34), 'rh': slice(34, 68)}
        full_dk84 = {'lh': slice(0, 34), 'rh': slice(49, 83)}
        if n_nodes == 68:
            hemispheres = cortex_only
        elif n_nodes == 84:
            hemispheres = full_dk84
        else:
            raise ValueError(
                'experimental_fc {} has {} nodes; expected 68 cortical '
                'Desikan Killiany areas or the full 84-node DK parcellation'
                .format(path, n_nodes))

        if data.ndim == 2 and data.shape[0] == data.shape[1]:
            return {
                hemi: pd.DataFrame(data[nodes, nodes],
                                   index=areas, columns=areas)
                for hemi, nodes in hemispheres.items()
            }

        # BOLD time series: add the participant axis if it is missing, then
        # average the per-participant correlation matrices.
        bold = data[:, :, np.newaxis] if data.ndim == 2 else data
        exp_fc = {}
        for hemi, nodes in hemispheres.items():
            correlations = [
                pd.DataFrame(bold[:, nodes, person], columns=areas).corr()
                for person in range(bold.shape[2])
            ]
            exp_fc[hemi] = sum(correlations) / len(correlations)
        return exp_fc

    @timeit
    def calculateBOLDConnectivity(self):
        """
        Calculates BOLD connectivities
        """
        try:
            BOLD_correlation = pd.read_pickle(
                    os.path.join(self.sim_folder, 'bold_correlation.pkl')
                    )
        except FileNotFoundError:
            if not hasattr(self, 'self.BOLD'):
                self.BOLD = self.computeBOLD()
            # Start and end time, cast to [s]
            t_min = self.ana_dict['plotBOLD']['tmin'] / 1000.
            t_max = self.sim_dict['t_sim'] / 1000.

            # The BOLD output was calculated by rk45, which does not give the
            # results on a specified grid. Thus we need to sample all signals
            # to the same grid.
            bold_timesteps = self.ana_dict['plotBOLD']['stepSize'] / 1000.
            t_new = np.arange(
                    t_min,
                    t_max,
                    bold_timesteps
                    )

            BOLD_RESAMPLED = self.BOLD.copy()
            for area, data in self.BOLD.groupby(level=0):
                t = data.loc[area, 't']
                b = data.loc[area, 'bold']
                # Last value might be out of range, extrapolate it
                bold_fun = interp1d(
                        t,
                        b,
                        bounds_error=False,
                        fill_value='extrapolate'
                        )
                b_new = bold_fun(t_new)
                BOLD_RESAMPLED.loc[area, 't'] = t_new
                BOLD_RESAMPLED.loc[area, 'bold'] = b_new

            # Calculate the correlations
            index = BOLD_RESAMPLED.loc[:, 'bold'].index
            BOLD_correlation = pd.DataFrame(
                    index=index,
                    columns=index,
                    dtype=np.float64
                    )
            for source in index:
                for target in index:
                    BOLD_correlation.loc[source, target] = np.corrcoef(
                            [
                                BOLD_RESAMPLED.loc[source, 'bold'],
                                BOLD_RESAMPLED.loc[target, 'bold']
                                ]
                            )[0,1]
            BOLD_correlation.to_pickle(
                    os.path.join(self.sim_folder, 'bold_correlation.pkl')
                    )
        return BOLD_correlation

    @timeit
    def calculateFuncionalConnectivityCorrelations(self):
        """
        Calculates functional connectivity correlations.
        """
        if not hasattr(self, 'self.BOLD_correlation'):
            self.BOLD_correlation = self.calculateBOLDConnectivity()
        if not hasattr(self, 'self.rate'):
            self.curr_in = self.synapticInputCurrent()
        # Correlations will be written into this file
        outfile = open(os.path.join(self.ana_folder, 'correlations.txt'), 'w')
        tmin = self.ana_dict['plotConnectivities']['tmin']
        exclude_diagonal = self.ana_dict['functconn_corr']['exclude_diagonal']

        # Read in synaptic currents
        synaptic_currents = self.curr_in.T

        # Correlate synaptic currents, yields a pandas dataframe of simulated
        # functional connectivity based on synaptic input currents
        # (df_sim_fc_syn)
        df_sim_fc_syn = synaptic_currents[synaptic_currents.index >= tmin].corr()
        if exclude_diagonal:
            np.fill_diagonal(df_sim_fc_syn.values, np.nan)

        # Read in simulated functional connectivity based on calculated BOLD
        # signal
        df_sim_fc_bold = self.BOLD_correlation
        if exclude_diagonal:
            np.fill_diagonal(df_sim_fc_bold.values, np.nan)

        # Sort
        df_sim_fc_syn = df_sim_fc_syn.sort_index(axis=0).sort_index(axis=1)
        df_sim_fc_bold = df_sim_fc_bold.sort_index(axis=0).sort_index(
                axis=1
                )

        # Derive numpy array containing the correlation values. Handier than
        # the underlying pandas dataframe
        sim_fc_bold = df_sim_fc_bold.values[
            np.tril_indices(df_sim_fc_bold.shape[0])
        ]
        sim_fc_bold = sim_fc_bold[~np.isnan(sim_fc_bold)]
        sim_fc_syn = df_sim_fc_syn.values[
            np.tril_indices(df_sim_fc_syn.shape[0])
        ]
        sim_fc_syn = sim_fc_syn[~np.isnan(sim_fc_syn)]

        # Correlate simulated functional connectivity based on synaptic
        # currents and calculated BOLD signal
        sim_fc_bold__sim_fc_syn__corr = np.corrcoef(
                [sim_fc_bold, sim_fc_syn]
                )[0,1]

        # This is PEP 8 conform, but is it really nice?
        outfile.write(
                ('correlation_simulated_bold_simulated_functional_connectivity'
                f': {sim_fc_bold__sim_fc_syn__corr}\n')
                )

        try:
            # =================================================================
            # ============= READ IN AND EXTRACT EXPERIMENTAL DATA =============
            # =================================================================
            # Set Path to experimental data
            data_dir = os.path.join(
                    self.base_path, 'experimental_data', 'senden', 'rsData_7T_DKparcel'
                    )

            # Read in regions of interest.
            roi = pd.read_csv(
                os.path.join(data_dir, 'ROIs.txt'),
                header=None, names=['roi'], dtype=str
            ).squeeze()
            # The rois are given in this manner: ctx-lh-bankssts
            # The name of the area is the last word after -
            roi = roi.apply(lambda x: x.split('-')[-1])
            # The cortical areas are in the range from 14 to 82
            roi = roi.drop(range(0, 14)).drop(range(82, 85))

            # Read in the bold signal. BOLD.shape = (600, 85, 19)
            # First dimension: timesteps
            # Second dimension: Desikan Killiany areas, left (0:34) and right
            # (34:68) hemisphere
            # Third dimension: Participants
            # orientation discrimination, numerosity
            BOLD = np.load(os.path.join(data_dir, 'rsDATA_7T_DKparcel.npy'))
            BOLD = BOLD[:, 14:82, :]

            # There are 600 timesteps in 1.5 second steps in the data
            resolution = 1.5
            timesteps = np.arange(600) * resolution

            # Extract the number of persons
            no_of_persons = BOLD.shape[2]

            # Extraction of Desikan Killiany area names from left hemisphere,
            # i.e. 0:34, and stripping the first 3 characters indicating the
            # hemisphere
            areas = roi.values[:34]

            # Extraction of BOLD series into a dictionary of Dataframes of form
            # exp_fc[person][hemisphere]
            # exp_fc contains the functional connectivities of 19 subjects
            exp_fc = {}

            # Loop over all persons
            for person in range(no_of_persons):
                # Left hemisphere is 0:34, right hemisphere is 34:68
                lh_person = BOLD[:, 0:34, person]
                rh_person = BOLD[:, 34:68, person]

                # BOLD signal into DataFrames
                lh = pd.DataFrame(lh_person, index=timesteps, columns=areas)
                rh = pd.DataFrame(rh_person, index=timesteps, columns=areas)

                # Correlations of all columns, i.e. areas, with each other
                lh_fc = lh.corr()
                rh_fc = rh.corr()

                # Correlation with itself is trivially 1, set those values to
                # nan
                if exclude_diagonal:
                    np.fill_diagonal(lh_fc.values, np.nan)
                    np.fill_diagonal(rh_fc.values, np.nan)

                # Sort and put into dictionary
                lh_fc = lh_fc.sort_index(axis=0).sort_index(axis=1)
                rh_fc = rh_fc.sort_index(axis=0).sort_index(axis=1)
                exp_fc[person] = {
                        'lh': lh_fc,
                        'rh': rh_fc
                        }

            # =================================================================
            # ======================= CALCULATIONS ============================
            # =================================================================
            # Calculate correlations of experimental functional connectivities
            # Convert all functional connectivities to numpy arrays. Correlate
            # these
            # Calculate correlations of all different experimental fcs with
            # simulated fcs
            exp_fc_array_lh = []
            exp_fc_array_rh = []
            exp_fc__sim_fc_syn__array_lh = []
            exp_fc__sim_fc_syn__array_rh = []
            exp_fc__sim_fc_bold__array_lh = []
            exp_fc__sim_fc_bold__array_rh = []
            for i in range(no_of_persons):
                tmp_lh = exp_fc[i]['lh'].values[
                    np.tril_indices(exp_fc[i]['lh'].shape[0])
                ]
                tmp_rh = exp_fc[i]['rh'].values[
                    np.tril_indices(exp_fc[i]['rh'].shape[0])
                ]
                tmp_lh = tmp_lh[~np.isnan(tmp_lh)]
                tmp_rh = tmp_rh[~np.isnan(tmp_rh)]
                exp_fc_array_lh.append(tmp_lh)
                exp_fc_array_rh.append(tmp_rh)

                exp_fc__sim_fc_syn__tmp_lh = np.corrcoef(sim_fc_syn, tmp_lh)[0, 1]
                exp_fc__sim_fc_syn__tmp_rh = np.corrcoef(sim_fc_syn, tmp_rh)[0, 1]
                exp_fc__sim_fc_syn__array_lh.append(exp_fc__sim_fc_syn__tmp_lh)
                exp_fc__sim_fc_syn__array_rh.append(exp_fc__sim_fc_syn__tmp_rh)

                exp_fc__sim_fc_bold__tmp_lh = np.corrcoef(sim_fc_bold, tmp_lh)[0, 1]
                exp_fc__sim_fc_bold__tmp_rh = np.corrcoef(sim_fc_bold, tmp_rh)[0, 1]
                exp_fc__sim_fc_bold__array_lh.append(exp_fc__sim_fc_bold__tmp_lh)
                exp_fc__sim_fc_bold__array_rh.append(exp_fc__sim_fc_bold__tmp_rh)

            exp__exp__corr_lh = np.corrcoef(exp_fc_array_lh)
            exp__exp__corr_rh = np.corrcoef(exp_fc_array_rh)

            # Calculate mean correlation between functional connectivities.
            # This gives us to what extent the functional connectivities of the
            # different subjects correspond to each other
            exp__exp_mean__corr_lh = np.sum(
                    np.tril(exp__exp__corr_lh, k=-1)
                    ).sum() / np.sum(range(no_of_persons))
            exp__exp_mean__corr_rh = np.sum(
                    np.tril(exp__exp__corr_rh, k=-1)
                    ).sum() / np.sum(range(no_of_persons))

            outfile.write(
                    ('exp__exp__corr_lh:\n'
                    f'{exp__exp__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__exp__corr_rh:\n'
                    f'{exp__exp__corr_rh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_syn__array_lh:\n'
                    f'{exp_fc__sim_fc_syn__array_lh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_syn__array_rh:\n'
                    f'{exp_fc__sim_fc_syn__array_rh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_bold__array_lh:\n'
                    f'{exp_fc__sim_fc_bold__array_lh}\n')
                    )
            outfile.write(
                    ('exp_fc__sim_fc_bold__array_rh:\n'
                    f'{exp_fc__sim_fc_bold__array_rh}\n')
                    )
            outfile.write(
                    ('exp__exp_mean__corr_lh: '
                    f'{exp__exp_mean__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__exp_mean__corr_rh: '
                    f'{exp__exp_mean__corr_rh}\n')
                    )

            # Calculate experimental mean functional connectivity
            exp_fc_mean_lh = np.sum(
                [exp_fc[i]['lh'].values[
                        np.tril_indices(exp_fc[i]['lh'].shape[0])
                    ] for i in range(no_of_persons)],
                axis=0
                ) / no_of_persons
            exp_fc_mean_rh = np.sum(
                [exp_fc[i]['rh'].values[
                        np.tril_indices(exp_fc[i]['rh'].shape[0])
                    ] for i in range(no_of_persons)],                  
                axis=0
                ) / no_of_persons

            # Diagonal elements gave nans (correlations with themeselves)
            exp_fc_mean_lh = exp_fc_mean_lh[~np.isnan(exp_fc_mean_lh)]
            exp_fc_mean_rh = exp_fc_mean_rh[~np.isnan(exp_fc_mean_rh)]

            # Correlation experimental functional connectivity with simulated
            # functional connectivity based on synaptic currents
            exp__sim_syn__corr_lh = np.corrcoef(
                    [exp_fc_mean_lh, sim_fc_syn]
                    )[0,1]
            exp__sim_syn__corr_rh = np.corrcoef(
                    [exp_fc_mean_rh, sim_fc_syn]
                    )[0,1]

            # Correlation experimental functional connectivity with simulated
            # functional connectivity based on calculated BOLD signal
            exp__sim_bold__corr_lh = np.corrcoef(
                    [exp_fc_mean_lh, sim_fc_bold]
                    )[0,1]
            exp__sim_bold__corr_rh = np.corrcoef(
                    [exp_fc_mean_rh, sim_fc_bold]
                    )[0,1]

            outfile.write(
                    ('exp__sim_syn__corr_lh: '
                    f'{exp__sim_syn__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__sim_syn__corr_rh: '
                    f'{exp__sim_syn__corr_rh}\n')
                    )
            outfile.write(
                    ('exp__sim_bold__corr_lh: '
                    f'{exp__sim_bold__corr_lh}\n')
                    )
            outfile.write(
                    ('exp__sim_bold__corr_rh: '
                    f'{exp__sim_bold__corr_rh}')
                    )
        except FileNotFoundError:
            if self.logger:
                self.logger.warning('Cannot load experimental FMRI data. '
                    'Either the datapath is wrong or the data is not available.')
        outfile.close()

    @timeit
    def calculateRateDistributionSimilarity(self):
        """
        Calculates the rate distribution similarity between simulated and
        experimental data.

        """
        if not hasattr(self, 'individual_rates'):
            self.individual_rates = self.individualFiringRate()
        rates_sim = self.individual_rates

        # KS distances will be written into this file
        outfile = os.path.join(self.ana_folder, 'rate_distr_similarity.txt')

        try:
            # =================================================================
            # ============= READ IN AND EXTRACT EXPERIMENTAL DATA =============
            # =================================================================
            # Set Path to experimental data
            data_dir = os.path.join(
                    self.base_path, 'experimental_data', 'rutishauser', 'spikes'
                    )
            # Load experimental spiking data
            filename = os.path.join(data_dir, 'mfc.mat')
            data = loadmat(filename, squeeze_me=True, mat_dtype=False,
                           chars_as_strings=True)
            data = data['data_mfc']
            # Calculate rates in spks/s
            rates_data = []
            for id in range(len(data)):
                # data structure described in README.m
                stim_on = data[id]['ts']['stim_on'][()]
                baseline = data[id]['ts']['baseline_stim_on'][()]
                for st in stim_on:
                    st = np.atleast_1d(st)
                    st = st[st < baseline]
                    rates_data.append(st.size / baseline)

            # =================================================================
            # ======================= CALCULATIONS ============================
            # =================================================================
            # Extract all rates of area
            rates_sim_area = rates_sim.apply(list).groupby('area').agg(sum)
            # Remove zero rate neurons
            if self.ana_dict['rate_distribution_similarity']['remove_zeros']:
                rates_data = [rate for rate in rates_data
                              if not np.isclose(rate, 0)]
                rates_sim_area = rates_sim_area.apply(
                    lambda rates: [rate for rate in rates
                                   if not np.isclose(rate, 0)])
            # Compute KS distance
            KS_stats = rates_sim_area.apply(ks_2samp, args=(rates_data,))
            # Split tuples and write to file
            KS_stats = pd.DataFrame(data=KS_stats.values.tolist(),
                                    index=KS_stats.index,
                                    columns=['KS_dist', 'p_value'])
            KS_stats.to_csv(outfile, sep='\t')
        except FileNotFoundError:
            if self.logger:
                self.logger.warning('Cannot load experimental spiking data. '
                    'Either the datapath is wrong or the data is not available.')

    def plotBOLDConnectivity(self):
        """
        Plots the BOLD connectivity matrix using a heatmap.If the BOLD 
        connectivity matrix has not been calculated yet, it will first 
        calculate it using the `calculateBOLDConnectivity` method.
        """
        if not hasattr(self, 'self.BOLD_correlation'):
            self.BOLD_correlation = self.calculateBOLDConnectivity()
        # Plot
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        sns.heatmap(
                self.BOLD_correlation,
                square=True,
                cmap="YlGnBu",
                cbar=True,
                cbar_kws={"shrink": .66},
                vmin=-1.,
                vmax=1.
                )
        plt.tight_layout()
        fig.savefig(os.path.join(
            self.plot_folder,
            'bold_correlation.{0}'.format(extension)
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    def plotRasterArea(self, area):
        """
        Generates a rasterplot of the spiking activity in a specified area.
        Parameters low and high are in ms.
        ----------
        area : string
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        extension = self.ana_dict['extension']
        fraction = self.ana_dict['plotRasterArea']['fraction']
        low = self.ana_dict['plotRasterArea']['low']
        high = self.ana_dict['plotRasterArea']['high']
        fig = plt.figure(1, figsize=(8, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        ind = []
        names = []
        gid_norm = 0
        ms_to_s = 0.001
        colors = {'E': '#1f77b4', 'I': '#ff7f0e'}
        for (layer, pop), sts in self.spikes.loc[area].items():
            # y label axis namin
            name = ' '.join([layer, pop])
            names.append(name)

            # Random shuffle spiketrains in place
            random.shuffle(sts)
            j = 0
            # Real population size, not all neurons spiked. Thus take the
            # fraction from this value.
            pop_size = self.popGids.loc[area, layer, pop].maxGID - \
                    self.popGids.loc[area, layer, pop].minGID + 1

            # Fraction of total number of neurons
            no_sts = int(fraction * pop_size)
            # Fraction of neurons that actually spiked
            frac_spiking = len(sts) * 1. / pop_size

            # where to put y label
            ind.append(- int(no_sts / 2) + gid_norm)
            # Loop as many times as we have spike trains
            for _ in range(no_sts):
                gid_norm = gid_norm - 1
                # Decide whether spiketrain contains spikes
                if random.random() < frac_spiking and j < len(sts):
                    st = sts[j]
                    j += 1
                    filtered_st = st[st > low]
                    filtered_st = filtered_st[filtered_st < high]
                    # TODO beautify plot
                    if len(filtered_st) > 0:
                        ax1.plot(
                            filtered_st * ms_to_s,
                            gid_norm * np.ones_like(filtered_st),
                            colors[pop],
                            marker='.',
                            markersize=2,
                            linestyle="None"
                        )
    
        ax1.axis([low * ms_to_s, high * ms_to_s, gid_norm, 0])
        ax1.set_xlabel('Time (s)', fontdict={"size": 12})
        ax1.tick_params(axis="x", labelsize=12)
        # ax1.set_ylabel("Layer-Population")
        ax1.set_yticks(ind)
        ax1.set_yticklabels(names, fontdict={"size": 12})
        ax1.set_title(area.title(), fontdict={"size": 16})
        fig.savefig(os.path.join(
            self._category_plot_folder('raster'),
            'raster_{0}.{1}'.format(area, extension)
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    def plotRasterAbetaArea(self, area):
        """
        Generates a rasterplot of the spiking activity in a specified area.
        Parameters low and high are in ms.
        ----------
        area : string
        """
        if not self._abeta_active():
            if self.logger:
                self.logger.info(
                    'No Abeta neurons (Abeta disabled or all ratios zero); '
                    'skipping Abeta raster for %s.', area)
            return
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, "AbetaGids"):
            self.AbetaGids = self._readAbetaPopGids()
        
        extension = self.ana_dict['extension']
        fraction = self.ana_dict['plotRasterArea']['fraction']
        low = self.ana_dict['plotRasterArea']['low']
        high = self.ana_dict['plotRasterArea']['high']
        fig = plt.figure(1, figsize=(8, 7))
        ax1 = fig.add_subplot(1, 1, 1)
        ind = []
        names = []
        gid_norm = 0
        ms_to_s = 0.001
        colors = {'Abeta_E': '#1f77b4', 'Abeta_I': '#ff7f0e', "E": "#a8dbff", "I": "#ffd6b3"}

        for (layer, pop), sts in self.spikes.loc[area].items():
            # y label axis naming
            name = ' '.join([layer, pop])
            names.append(name)

            # Random shuffle spiketrains in place  
            random.shuffle(sts)
            j = 0
            
            # Real population size, not all neurons spiked. Thus take the
            # fraction from this value.
            pop_size = self.popGids.loc[area, layer, pop].maxGID - \
                    self.popGids.loc[area, layer, pop].minGID + 1
            
            # 获取当前群体的GID范围
            min_gid = self.popGids.loc[area, layer, pop].minGID
            max_gid = self.popGids.loc[area, layer, pop].maxGID
            
            # 检查当前群体是否有对应的Abeta神经元
            has_abeta = False
            abeta_min_gid = None
            abeta_max_gid = None
            try:
                abeta_info = self.AbetaGids.loc[area, layer, pop]
                abeta_min_gid = abeta_info.minGID
                abeta_max_gid = abeta_info.maxGID
                has_abeta = True
            except KeyError:
                # 当前群体没有Abeta神经元
                has_abeta = False
            
            # Fraction of total number of neurons
            no_sts = int(fraction * pop_size)
            # Fraction of neurons that actually spiked
            frac_spiking = len(sts) * 1. / pop_size

            # where to put y label
            ind.append(- int(no_sts / 2) + gid_norm)
            
            # Loop as many times as we have spike trains
            for i in range(no_sts):
                gid_norm = gid_norm - 1
                # Decide whether spiketrain contains spikes
                if random.random() < frac_spiking and j < len(sts):
                    st = sts[j]
                    j += 1
                    filtered_st = st[st > low]
                    filtered_st = filtered_st[filtered_st < high]
                    
                    if len(filtered_st) > 0:
                        # 计算当前神经元的近似GID
                        # 基于在显示序列中的位置推算实际GID
                        neuron_gid = min_gid + int((i / no_sts) * pop_size)
                        
                        # 检查当前神经元是否为Abeta神经元
                        is_abeta = (has_abeta and 
                                abeta_min_gid <= neuron_gid <= abeta_max_gid)
                        
                        # 根据是否为Abeta神经元选择颜色
                        if is_abeta:
                            if pop == 'E':
                                color = colors['Abeta_E']
                            else:  # pop == 'I'
                                color = colors['Abeta_I']
                        else:
                            color = colors[pop]
                        
                        ax1.plot(
                            filtered_st * ms_to_s,
                            gid_norm * np.ones_like(filtered_st),
                            color,
                            marker='.',
                            markersize=2,
                            linestyle="None"
                        )

        ax1.axis([low * ms_to_s, high * ms_to_s, gid_norm, 0])
        ax1.set_xlabel('Time (s)', fontdict={"size": 12})
        ax1.tick_params(axis="x", labelsize=12)
        ax1.set_yticks(ind)
        ax1.set_yticklabels(names, fontdict={"size": 12})
        ax1.set_title(area, fontdict={"size": 16})
        fig.savefig(os.path.join(
            self._category_plot_folder('raster'),
            'raster_Abeta_{0}.{1}'.format(area, extension)
        ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)

    @timeit
    def plot_raster_statistics(self, save_fig=False, raster_areas=None):
        """
        Plots raster statistics including raster plots for specified areas and 
        boxplots for firing rates, CV ISI, and correlation coefficients.

        Parameters
        ----------
        save_fig : bool, optional
            If True, the figure will be saved to the plot folder. Default is False.
        raster_areas : list of str, optional
            List of areas to plot raster statistics for. Default is 
            ['caudalanteriorcingulate', 'pericalcarine', 'fusiform'].
        """
        if raster_areas is None:
            raster_areas = ['caudalanteriorcingulate', 'pericalcarine', 'fusiform']
        
        colors = {'E': '#4c72b0ff', 'I': '#c44e52ff'}
        raster_fraction = self.ana_dict['plotRasterArea']['fraction']
        raster_low = self.ana_dict['plotRasterArea']['low']
        raster_high = self.ana_dict['plotRasterArea']['high']
        roman_to_arabic_numerals = {
            'II/III': '2/3',
            'IV': '4',
            'V': '5',
            'VI': '6',
        }
        random.seed(1234)

        # Check if necessary attributes are already loaded or need to be calculated
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'spikes'):
            self.spikes = self._readSpikes()
        if not hasattr(self, 'rate'):
            self.rate = self.meanFiringRate()
        if not hasattr(self, 'pop_lv'):
            self.pop_lv = self.popLv()
        if not hasattr(self, 'pop_cv_isi'):
            self.pop_cv_isi = self.popCvIsi()
        if not hasattr(self, 'pop_cc'):
            self.pop_cc = self.popCorrCoeff()

        # Plotting
        plt.style.use('./misc/mplstyles/report_plots_master.mplstyle')
        fig = plt.figure(constrained_layout=True, figsize=(5.5, 3.5))
        label_prms = dict(fontsize=8, fontweight='bold', va='top', ha='right')
        gs = gridspec.GridSpec(3, 4, figure=fig)
        ax_raster1 = fig.add_subplot(gs[:, 0])
        ax_raster1.spines['top'].set_visible(False)
        ax_raster1.spines['right'].set_visible(False)
        ax_raster2 = fig.add_subplot(gs[:, 1])
        ax_raster2.spines['top'].set_visible(False)
        ax_raster2.spines['right'].set_visible(False)
        ax_raster3 = fig.add_subplot(gs[:, 2])
        ax_raster3.spines['top'].set_visible(False)
        ax_raster3.spines['right'].set_visible(False)
        ax_rates = fig.add_subplot(gs[0, 3])
        ax_rates.spines['top'].set_visible(False)
        ax_rates.spines['right'].set_visible(False)
        ax_cv = fig.add_subplot(gs[1, 3])
        ax_cv.spines['top'].set_visible(False)
        ax_cv.spines['right'].set_visible(False)
        ax_cc = fig.add_subplot(gs[2, 3])
        ax_cc.spines['top'].set_visible(False)
        ax_cc.spines['right'].set_visible(False)

        # Raster plots
        ms_to_s = 1e-3
        axs_raster = [ax_raster1, ax_raster2, ax_raster3]
        raster_labels = ['A', 'B', 'C']
        for ax, area, label in zip(axs_raster, raster_areas, raster_labels):
            ind = []
            names = []
            gid_norm = 0
            for (layer, pop), sts in self.spikes.loc[area].items():
                layer_roman = roman_to_arabic_numerals[layer]
                # Random shuffle spiketrains in place
                random.shuffle(sts)

                # Real population size, not all neurons spiked. Thus take the fraction from this value.
                popGid_alp = self.popGids.loc[area, layer, pop]
                pop_size = popGid_alp.maxGID - popGid_alp.minGID + 1
                # Fraction of total number of neurons
                no_sts = int(raster_fraction * pop_size)
                # Fraction of neurons that actually spiked
                frac_spiking = len(sts) / pop_size

                # y label position and name
                ind.append(- int(no_sts / 2) + gid_norm)
                name = ' '.join([layer_roman, pop])
                names.append(name)

                j = 0
                # Loop as many times as we have spike trains
                for _ in range(no_sts):
                    gid_norm = gid_norm - 1
                    # Decide whether spiketrain contains spikes
                    if random.random() < frac_spiking and j < len(sts):
                        st = sts[j]
                        j += 1
                        filtered_st = st[(st > raster_low) & (st < raster_high)]
                        if len(filtered_st) > 0:
                            ax.plot(
                                filtered_st * ms_to_s,
                                gid_norm * np.ones_like(filtered_st),
                                colors[pop],
                                marker='.',
                                markersize=1.5,
                                linestyle="None"
                            )

            ax.axis([raster_low * ms_to_s, raster_high * ms_to_s, gid_norm, 0])
            ax.set_xlabel('Time (s)')
            ax.set_yticks(ind)
            ax.set_yticklabels(names)
            ax.set_title(area)
            ax.text(s=label, transform=ax.transAxes, x=-0.2, y=1.06, **label_prms)

        # Boxplots
        axs_boxplots = [ax_rates, ax_cv, ax_cc]
        data_boxplots = [self.rate, self.pop_cv_isi, self.pop_cc]
        boxplots_labels = ['D', 'E', 'F']
        for ax, data, label in zip(axs_boxplots, data_boxplots, boxplots_labels):
            # reorder Series into DataFrame
            area = np.unique(data.index.get_level_values(0))
            layer = np.unique(data.index.get_level_values(1))
            pop_type = np.unique(data.index.get_level_values(2))
            multi_index = pd.MultiIndex.from_product([layer, pop_type])
            ind = [' '.join(i) for i in multi_index.tolist()]
            names = [' '.join((roman_to_arabic_numerals[l_], p_)) for l_, p_ in (i.split(' ') for i in ind)]
            data_lp = pd.DataFrame(data=np.nan, index=area, columns=ind)
            for (a, l, p), r in data.items():
                data_lp.loc[a, l+' '+p] = r

            boxplot = sns.boxplot(data=data_lp, orient='h', ax=ax, saturation=1,
                                  width=0.5, fliersize=2.5, color='k')
            col = [colors['E'], colors['I']]
            for i in range(len(ind)):
                mybox = boxplot.patches[i]
                mybox.set_facecolor(col[i % 2])
            ax.text(s=label, transform=ax.transAxes, x=-0.1, y=1.25, **label_prms)
            # Print the extension of the whiskers
            lower = []
            upper = []
            for name, x in data_lp.items():
                dat = x.dropna().values
                if len(dat) > 0:
                    upper_quartile = np.percentile(dat, 75)
                    lower_quartile = np.percentile(dat, 25)
                    iqr = upper_quartile - lower_quartile
                    upper_whisker = dat[dat <= upper_quartile + 1.5 * iqr].max()
                    lower_whisker = dat[dat >= lower_quartile - 1.5 * iqr].min()
                    lower.append(lower_whisker)
                    upper.append(upper_whisker)
            if self.logger:
                self.logger.debug('label: %s, lowest whisker: %s, highest whisker: %s',
                    label, round(min(lower), 1), round(max(upper), 1))
            ax.set_yticklabels(names)
        ax_rates.set_xlim(0)
        ax_rates.set_xlabel('Firing rate (spikes/s)')
        ax_cv.set_xlim(0)
        ax_cv.set_xlabel('CV interspike interval')
        ax_cc.set_xlabel('Correlation coefficient')

        # Save figure if save_fig is True
        if save_fig:
            extension = self.ana_dict['extension']
            fig.savefig(os.path.join(self.plot_folder, f'figure_spike_statistics.{extension}'),
                        dpi=self.ana_dict['dpi'])
            plt.close(fig)

        # Restore the ambient style (see plot_functional_connectivity).
        plt.style.use('default')

    def plot_all_binned_spike_rates_area(self):
        """
        Generates a summary plot giving an overview over the distribution of
        frequencies signal in the network.
        """
        if not hasattr(self, 'spikes_per_neuron_area_resolved'):
            self.spikes_per_neuron_population_resolved, self.spikes_per_neuron_area_resolved = self.binned_spikerates_per_neuron()
        signal = self.spikes_per_neuron_area_resolved

        # Determine max bold for y axis
        max_count = signal[:, 'hist'].apply(max).max()
        if self.ana_dict['plot_binned_spikerates_per_neuron']['max_rate']:
            max_rate = self.ana_dict['plot_binned_spikerates_per_neuron']['max_rate']
        else:
            max_rate = signal[:, 'bin_edges'].apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['binned_spikerates_per_neuron']['num_col']
        num_row = math.ceil(len(signal[:, 'bin_edges']) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, (tmp_count, tmp_rates)) in enumerate(signal.groupby(level=0)):
            mask = tmp_rates < max_rate
            rates = tmp_rates[mask]
            if max(tmp_rates) < max_rate:
                rates = rates[:-1]
            count = tmp_count[mask[:-1]]  # [:-1]
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.bar(
                    rates,
                    count,
                    width=rates[1] - rates[0],
                    edgecolor='black',
                    align='edge'
                    )

            # Same limits for everybody!
            ax.set_xlim(0, max_rate)
            ax.set_ylim(0, max_count)

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            'binned_spikerates_per_neuron_area_resolved.png'
            ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plotAllSynapticCurrentsSummary(self):
        """
        Generates a summary plot giving an overview over the synaptic currents
        in the network.
        """
        if not hasattr(self, 'curr_in'):
            self.curr_in = self.synapticInputCurrent()
        synaptic_currents = self.curr_in
        synaptic_currents = synaptic_currents.loc[:, synaptic_currents.columns>=500]

        # Determine max bold for y axis
        max_current = synaptic_currents.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllSynapticCurrentsSummary']['num_col']
        num_row = math.ceil(len(synaptic_currents) / num_col)

        for same_axis in [True, False]:
            # Initialize the figure
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(
                    num_row,
                    num_col,
                    figsize=(22, 22)
                    )

            # multiple line plot
            for num, (area, data_) in enumerate(synaptic_currents.groupby(level=0)):
                # Find the right spot on the plot
                ax = axes[num // num_col][num % num_col]

                times = data_.columns.values
                data = data_.values[0]

                # Plot the lineplot
                ax.plot(
                        times,
                        data,
                        marker='',
                        color='black',
                        linewidth=1.9,
                        alpha=0.9,
                        )

                # Same limits for everybody!
                ax.set_xlim(times[0], times[-1])
                if same_axis:
                    ax.set_ylim(0, max_current)

                # Not ticks everywhere
                if num < (num_row - 1)*num_col:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Time (s)')

                if num % num_col != 0:
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel('Synaptic current')

                # Add title
                ax.set_title(
                        area,
                        loc='left',
                        fontsize=20,
                        fontweight=0,
                        color='black'
                        )

            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                f'synaptic_currents_overview_same_axis_{same_axis}.png'
                ), dpi=self.ana_dict['dpi'])
            plt.clf()
            plt.close(fig)
            plt.style.use('default')

    def plotAllBOLDSignalSummary(self):
        """
        Generates a summary plot giving an overview over the BOLD signal
        in the network.
        """
        if not hasattr(self, 'BOLD'):
            self.BOLD = self.computeBOLD()

        # Determine max bold for y axis
        max_bold = self.BOLD[:, 'bold'].apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllBOLDSignalSummary']['num_col']
        num_row = math.ceil(len(self.BOLD[:, 'bold']) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, (data, times)) in enumerate(self.BOLD.groupby(level=0)):
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.plot(
                    times,
                    data,
                    marker='',
                    color='black',
                    linewidth=1.9,
                    alpha=0.9,
                    )

            # Same limits for everybody!
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(0, max_bold)

            # Not ticks everywhere
            if num < (num_row - 1)*num_col:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Time (s)')

            if num % num_col != 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel('BOLD')

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            'bold_overview.png'
            ), dpi=self.ana_dict['dpi'])
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plotAllFiringRatesSummary(self):
        """
        Generates a summary plot giving an overview over the spiking activity
        in the network. Each subplot is the time course of the spike rates
        averaged across all neurons in a given area.
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'rate_hist_areas'):
            self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()
        rate_hist_areas = self.rate_hist_areas * 1000.  # ms to s
        resolution = self.ana_dict['rate_histogram']['binsize']

        if self.ana_dict['rate_histogram']['t_start']:
            t_start = self.ana_dict['rate_histogram']['t_start']
        else:
            t_start = 0.

        if self.ana_dict['rate_histogram']['t_stop']:
            t_stop = self.ana_dict['rate_histogram']['t_stop']
        else:
            t_stop = self.sim_dict['t_sim']

        rate_hist_areas = rate_hist_areas.apply(lambda x: x[int(t_start/resolution):])

        times = np.arange(t_start, t_stop, resolution) / 1000.  # To seconds

        # Determine max rate for y axis
        max_rate = rate_hist_areas.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllFiringRatesSummary']['num_col']
        num_row = math.ceil(len(rate_hist_areas) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-v0_8-darkgrid')

        for same_y_axis in [False, True]:
            fig, axes = plt.subplots(
                    num_row,
                    num_col,
                    figsize=(22, 22)
                    )

            # multiple line plot
            for num, (area, data) in enumerate(rate_hist_areas.items()):
                # Find the right spot on the plot
                ax = axes[num // num_col][num % num_col]

                # Plot the lineplot
                ax.plot(
                        times,
                        data,
                        marker='',
                        color='black',
                        linewidth=1.9,
                        alpha=0.9,
                        )

                # Same limits for everybody!
                ax.set_xlim(times[0], times[-1])
                if same_y_axis:
                    ax.set_ylim(0, max_rate)
                else:
                    ax.set_ylim(0, max(data))

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Rate (Spikes / s)')

                # Add title
                ax.set_title(
                        area,
                        loc='left',
                        fontsize=20,
                        fontweight=0,
                        color='black'
                        )

            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                'spiking_overview_same_y_axis_{}.png'.format(same_y_axis)
                ), dpi=self.ana_dict['dpi'])
            plt.clf()
            plt.close(fig)
        plt.style.use('default')

    @timeit
    def _readSpikes(self):
        """
        Reads SpikeTrains of the simulation.

        Returns
        -------
        spikes : Series of lists of SpikeTrains
        """
        try:
            spikes = pd.read_pickle(
                os.path.join(self.sim_folder, 'spikes.pkl')
            )
        except FileNotFoundError:
            if self.logger:
                self.logger.info('Loading SpikeTrains from dat')
            spikes = self._readSpikesFromDAT()
            # Save spikes to pickle for faster read access
            spikes.to_pickle(os.path.join(self.sim_folder, 'spikes.pkl'))
        return spikes

    @timeit
    def _readPopGids(self):
        """
        Reads the min / max GID for each population.

        Returns
        -------
        popGids : DataFrame
            Columns ['minGID', 'maxGID']
        """
        try:
            popGids = pd.read_pickle(
                os.path.join(self.sim_folder, 'population_GIDs.pkl')
            )
        except FileNotFoundError:
            popGids = pd.read_csv(
                os.path.join(self.sim_folder, 'population_GIDs.dat'),
                sep=';', header=None, names=['minGID', 'maxGID']
            )
            popGids.index = pd.MultiIndex.from_tuples([
                literal_eval(s) for s in popGids.index
            ])
            popGids['pop_size'] = popGids['maxGID'] - popGids['minGID'] + 1
            popGids.index.names = ['area', 'layer', 'pop']
            # Save popGids to pickle for faster read access
            popGids.to_pickle(
                os.path.join(self.sim_folder, 'population_GIDs.pkl')
            )
        return popGids

    @timeit
    def _readAbetaPopGids(self):
        """
        Reads the min / max GID for each Abeta-affected population.

        Returns
        -------
        popGids : DataFrame
            Columns ['minGID', 'maxGID']
        """
        try:
            popGids = pd.read_pickle(
                os.path.join(self.sim_folder, 'Abeta_GID.pkl')
            )
        except FileNotFoundError:
            try:
                popGids = pd.read_csv(
                    os.path.join(self.sim_folder, 'Abeta_GIDs.dat'),
                    sep=';', header=None, names=['minGID', 'maxGID', "pop_size"]
                )
                popGids.index = pd.MultiIndex.from_tuples([
                    literal_eval(s) for s in popGids.index
                ])
                popGids.index.names = ['area', 'layer', 'pop']
                # Save popGids to pickle for faster read access
                popGids.to_pickle(
                    os.path.join(self.sim_folder, 'Abeta_GIDs.pkl')
                )
            except FileNotFoundError:
                a = np.random.default_rng(43)
                b = np.random.default_rng(a.choice(10000, 1))
        return popGids

    @timeit
    def _readIsyn(self):
        """
        Read post-synaptic currents (PSCs) of the simulation.

        Returns
        -------
        I_syn (pandas.DataFrame): Array of PSCs
        """
        try:
            I_syn = pd.read_pickle(os.path.join(self.sim_folder, "I_syn.pkl"))
        except:
            I_syn = self._readIsynFromDAT()
            I_syn.to_pickle(os.path.join(self.sim_folder, "I_syn.pkl"))
        return I_syn

    def _readSpikesFromDAT(self):
        """
        Reads spikes from dat output files using pandas.
        Stores all SpikeTrains for one population in a list wich in
        turn is contained in a Series.

        Returns
        -------
        spikes : Series of arrays of arrays containing spike timings.
        """
        python_sort = self.ana_dict['python_sort']
        if python_sort:
            # Read in population gids
            popGids = self.popGids
            # glob all spikes files
            dat_files = glob.glob(os.path.join(self.sim_folder, 'spikes', '*.dat'))
            # Read in all dat files into a big dataframe with two columns, gid and
            # t. Magically this seems to work in parallel
            spikes = pd.concat(
                    (
                        pd.read_csv(
                            f,
                            sep='\t',
                            skiprows=3,
                            index_col=False,
                            header=None,
                            names=['gid', 't']
                            ) for f in dat_files
                    ),
                    ignore_index=True
                    )
            # Sort spikes, first gid, then t
            spikes = spikes.sort_values(['gid', 't'])
            # Create a spiketrain (i.e. numpy arrays) for every gid and store them
            # in a cell
            spikes = spikes.groupby('gid').apply(
                    lambda group: group.t.values
                    ).reset_index()
            spikes.columns = ['gid', 't']

            no_spiking = []
            for row in popGids.itertuples():
                area = row.Index[0]
                layer = row.Index[1]
                pop = row.Index[2]
                first_gid = row.minGID
                last_gid = row.maxGID
                try:
                    first = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][0]
                    last = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][-1]
                    spikes.loc[first:last, 'area'] = area
                    spikes.loc[first:last, 'layer'] = layer
                    spikes.loc[first:last, 'pop'] = pop
                except IndexError:
                    no_spiking.append([area, layer, pop])
                    pass
            spikes = spikes.groupby(
                    ['area', 'layer', 'pop']
                    ).apply(lambda group: group.t.values)

            for (area, layer, pop) in no_spiking:
                spikes.loc[area, layer, pop] = np.array([])
            spikes = spikes.sort_index()
            return spikes
        else:
            # This routine should be used when the simulated time is long, eg
            # 100 seconds. It is not faster when analyzing 10 seconds of
            # biological time. But when we simulate long and thus generate a
            # lot of data we need to do a lot of sorting. And pandas is bad at
            # this. In fact the original routine won't even finish sorting the
            # data for 100 seconds of biological time in 24 h. So for analyzing
            # larger datasets we need a more efficient spike reading in
            # routine. This routine relies a lot on GNU coreutils. E.g. sort
            # can be parallelized and it implements mergesort.
            # Note: this routine implicitly assumes that every area has spiked

            # ==============================================================================
            #                                   Definitions
            # ==============================================================================

            sorted_fn = 'combined_and_sorted_spiketrains.txt'
            available_cores = multiprocessing.cpu_count()

            popGids = self.popGids.sort_values('minGID')

            # ==============================================================================
            #                                 Presorting data
            #
            # Here I presort all dat files. Presorted files can easily be merged without
            # memory constraints. Also mergesort is quite fast.
            # The reason I use GNU sort instead of sorting in python is that GNU sort also
            # works in parallel. GNU sort uses 8 cores per default, which seems be a good
            # value as a rule of thumb. Furthermore I sort as many files as possible
            # simultaneously. I can launch available_cores / 8 jobs at a given time.
            # But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 15 minutes
            # ==============================================================================

            file_ending = '*.dat'
            self.rec_folder = os.path.join(self.sim_folder, 'spikes')
            dat_files = glob.glob(os.path.join(self.rec_folder, file_ending))

            ts = time.time()
            pool = Pool(math.ceil(available_cores / 8))  # Gnu sort sorts in parallel with 8 threads
            _ = pool.map(shell_presort_dat, dat_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('presorting data took %s s', passed_time)

            # ==============================================================================
            #                                 mergesorting data
            #
            # In this step the data from before is mergesorted into a huge file. This file
            # contains all spikes, sorted by gid and time. GNU sort is used as it provides
            # a good mergesorting algorithm. This is way I used it instead of a pythonic
            # way. But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 22 minutes
            # The resulting file, for 100 seconds of time, is 120 GB
            # ==============================================================================

            ts = time.time()
            subprocess.check_output(
                    f'export LC_ALL=C; sort -k1,1n -k2,2n -m --parallel=8 {self.rec_folder}/*_sorted.txt > {self.rec_folder}/{sorted_fn}',
                    shell=True
                    )
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('mergesorting data took %s s', passed_time)

            # ==============================================================================
            #            Splitting sorted data into population resolved data files
            #
            # Now we have this gigantic sorted datafile (120GB) which we would like to
            # split into population specific files. Meaning: Every text file contains all
            # spikes of the given population in a sorted manner. Splitting such a big file
            # takes quite some time. The algorithm works in the following way:
            # 1) Find a population in the middle of the popGids Dataframe
            # 2) Split according to this population
            # 3) Now we have two dataframes, one is left and the other one is right. These
            #    Dataframes also need to be splitted. As they are independent, they can be
            #    split in parallel
            # 4) As the algorithm progresses, the chunks become more numerous (more
            #    parallelization) and smaller (faster).
            # 5) When all has been split, we are done :)
            #
            # I haven't benchmarked the GNU splitting routine against a pyhtonic approach
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 80 minutes
            # ==============================================================================

            all_tmp = [(popGids, sorted_fn, 0, self.rec_folder)]

            for iteration in range(math.ceil(math.log2(len(popGids)))):
                ts = time.time()
                cores = min(int(2**iteration), available_cores)
                pool = Pool(cores)
                tmp = pool.starmap(split_sorted_file, all_tmp)
                all_tmp = []
                if tmp:
                    for x in range(len(tmp)):
                        if tmp[x] and len(tmp[x]) > 0:
                            all_tmp.append((tmp[x][0], tmp[x][1], tmp[x][4]+1, self.rec_folder))
                            if len(tmp[x]) > 3:
                                all_tmp.append((tmp[x][2], tmp[x][3], tmp[x][4]+1, self.rec_folder))
                te = time.time()
                passed_time = round(te - ts, 3)
                if self.logger:
                    self.logger.debug('Splitting iteration %d took %s s', iteration, passed_time)

            # ==============================================================================
            # Rename files such that the filename contains exact information on population
            #
            # We rename all files such that the filename gives away which population we are
            # looking at.
            #
            # This step is fast, around 1 minute
            # ==============================================================================

            d_population_forward = {
                    'II/III': '23',
                    'IV': '4',
                    'V': '5',
                    'VI': '6'
                    }

            d_population_backward = {
                    '23': 'II/III',
                    '4': 'IV',
                    '5': 'V',
                    '6': 'VI'
                    }

            ts = time.time()
            file_ending = sorted_fn + '_*'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            for fn in dem_files:
                with open(fn) as f:
                    a = int(f.readline().split()[0])
                    exact_population_tmp = popGids[(popGids.minGID <= a) & (popGids.maxGID >= a)].index.tolist()[0]
                    exact_population = (exact_population_tmp[0], d_population_forward[exact_population_tmp[1]], exact_population_tmp[2])

                    # Assert that first and last spike are from the same population and
                    # that every sorting so far has done the correct thing.
                    check = int(subprocess.check_output(['tail', '-1', fn]).split()[0])
                    check_population = popGids[(popGids.minGID <= check) & (popGids.maxGID >= check)].index.tolist()[0]
                    assert exact_population_tmp == check_population

                    new_name = os.path.join(
                            self.rec_folder,
                            '_'.join(exact_population) + '.spiketrains'
                            )
                    os.rename(fn, new_name)
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('Assigning correct names to datafiles took %s s', passed_time)

            # ==============================================================================
            #          Pull spikefiles together, have one spiketrain per line, remove gids
            #
            # Before the text files look like this:
            #
            # 7 5.5
            # 7 8.6
            # 7 9.2
            # 8 5.7
            # 8 6.5
            #
            # We now make sure that every line contains a spiketrain and remove the gid
            # number as it is not important anymore. The result looks like this:
            #
            # 5.5 8.6 9.2
            # 5.7 6.5
            #
            # This step takes, for 100 seconds of bio time on 8 cores, 20 minutes. on 64
            # cores probably 3 minutes.
            # ==============================================================================

            ts = time.time()
            file_ending = '*.spiketrains'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            pool = Pool(available_cores)
            _ = pool.map(shell_spiketrainify, dem_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('spiketrainify data took %s s', passed_time)

            # ==============================================================================
            #                        Read files into pandas Dataframes
            #
            # Reading in 100 seconds of bio time takes 20 minutes. Saving takes 1.5 minutes.
            # Note: This probably can be optimized
            # ==============================================================================

            ts = time.time()
            file_ending = '*.cut'
            spikes_txt = glob.glob(os.path.join(self.rec_folder, file_ending))
            all_spikes = pd.Series(index=popGids.index, dtype=object)
            for spike_txt_file in spikes_txt:
                area, layer, pop, *_ = spike_txt_file.split('/')[-1].replace('.','_').split('_')
                layer = d_population_backward[layer]
                tmp = []
                with open(spike_txt_file) as f:
                    lines=f.readlines()
                    for line in lines:
                        tmp.append(np.fromstring(line, dtype=float, sep=' '))
                all_spikes[(area, layer, pop)] = np.array(tmp, dtype=object)
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('putting data into Series took %s s', passed_time)

            ts = time.time()
            all_spikes.to_pickle(os.path.join(self.ana_folder, 'spikes.pkl'))
            te = time.time()
            passed_time = round(te - ts, 3)
            if self.logger:
                self.logger.info('saving Series took %s s', passed_time)
            return all_spikes

    def _readIsynFromDAT(self):
        """
        Read post-synaptic current from dat and save it as pickle.
        
        Returns
        ---
        I_syn: 

        """
        python_sort = self.ana_dict.get("I_syn_python_sort", False)
        # --------------------cd----------------------------------------------
        # Small / moderate datasets: keep the original pandas-only path
        # ------------------------------------------------------------------
        if python_sort:
            if self.logger:
                self.logger.info('I_syn sorting: pandas')
            Isyn_files = glob.glob(os.path.join(self.sim_folder, "I_syn", "*dat"))
            I_syn = pd.concat(
                (pd.read_csv(
                    f,
                    sep="\t",
                    skiprows=3,
                    index_col=False,
                    header=None,
                    names=["gid", "t", "I_syn_ex", "I_syn_in"]
                    ) for f in Isyn_files),
                ignore_index=True
                )
            I_syn = I_syn.sort_values(["gid", "t"])
            I_syn = I_syn.groupby("gid").agg({"I_syn_ex": list, "I_syn_in": list}).reset_index()
            I_syn["I_syn_in"] = I_syn["I_syn_in"].apply(np.array)
            I_syn["I_syn_ex"] = I_syn["I_syn_ex"].apply(np.array)
            
            for row in self.popGids.itertuples():
                area = row.Index[0]
                layer = row.Index[1]
                pop = row.Index[2]
                first_gid = row.minGID
                last_gid = row.maxGID
                try:
                    first = np.where(
                                (I_syn.gid <= last_gid) & (I_syn.gid >= first_gid)
                                )[-1][0]
                    last = np.where(
                                (I_syn.gid <= last_gid) & (I_syn.gid >= first_gid)
                                )[-1][-1]
                    I_syn.loc[first:last, 'area'] = area
                    I_syn.loc[first:last, 'layer'] = layer
                    I_syn.loc[first:last, 'pop'] = pop
                except IndexError:
                    pass
            I_syn = I_syn.groupby(["area", "layer", "pop"]).agg({"I_syn_ex": list, "I_syn_in": list}).reset_index()
            I_syn["I_syn_ex"] = I_syn["I_syn_ex"].apply(np.array)
            I_syn["I_syn_in"] = I_syn["I_syn_in"].apply(np.array)

        else:
            # ------------------------------------------------------------------
            # Large datasets (e.g., >100 GB): external sort + recursive split
            # ------------------------------------------------------------------
            if self.logger:
                self.logger.info('I_syn sorting: shell')
            sorted_fn = 'combined_and_sorted_Isyn.txt'
            available_cores = multiprocessing.cpu_count()
            popGids = self.popGids.sort_values('minGID')
            self.rec_folder = os.path.join(self.sim_folder, 'I_syn')

            # 1) presort each dat file in parallel (GNU sort is fast and O(1) memory)
            file_ending = '*.dat'
            dat_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            ts = time.time()
            pool = Pool(max(1, math.ceil(available_cores / 8)))  # GNU sort uses 8 threads by default
            _ = pool.map(shell_presort_dat, dat_files)
            pool.close()
            pool.join()
            if self.logger:
                self.logger.info('presorting I_syn data took %s s', round(time.time() - ts, 3))

            # 2) mergesort all presorted chunks into one globally sorted file
            ts = time.time()
            subprocess.check_output(
                    f'export LC_ALL=C; sort -k1,1n -k2,2n -m --parallel=8 {self.rec_folder}/*_sorted.txt > {self.rec_folder}/{sorted_fn}',
                    shell=True
                    )
            if self.logger:
                self.logger.debug('mergesorting I_syn data took %s s', round(time.time() - ts, 3))

            # 3) recursively split the huge file into population-specific files
            all_tmp = [(popGids, sorted_fn, 0, self.rec_folder)]
            for iteration in range(math.ceil(math.log2(len(popGids)))):
                ts = time.time()
                cores = min(int(2**iteration), available_cores)
                pool = Pool(cores)
                tmp = pool.starmap(split_sorted_file, all_tmp)
                pool.close()
                pool.join()
                all_tmp = []
                if tmp:
                    for x in range(len(tmp)):
                        if tmp[x] and len(tmp[x]) > 0:
                            all_tmp.append((tmp[x][0], tmp[x][1], tmp[x][4]+1, self.rec_folder))
                            if len(tmp[x]) > 3:
                                all_tmp.append((tmp[x][2], tmp[x][3], tmp[x][4]+1, self.rec_folder))
                if self.logger:
                    self.logger.debug('Splitting iteration %d took %s s', iteration, round(time.time() - ts, 3))

            # 4) rename split files to include population info (area_layer_pop.isyn)
            d_population_forward = {
                    'II/III': '23',
                    'IV': '4',
                    'V': '5',
                    'VI': '6'
                    }

            d_population_backward = {
                    '23': 'II/III',
                    '4': 'IV',
                    '5': 'V',
                    '6': 'VI'
                    }

            ts = time.time()
            file_ending = sorted_fn + '_*'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            for fn in dem_files:
                with open(fn) as f:
                    line = f.readline().split()
                    if not line:
                        continue
                    a = int(line[0])
                exact_population_tmp = popGids[(popGids.minGID <= a) & (popGids.maxGID >= a)].index.tolist()[0]
                exact_population = (exact_population_tmp[0], d_population_forward[exact_population_tmp[1]], exact_population_tmp[2])

                # Sanity check: first/last gid must belong to same population
                check = int(subprocess.check_output(['tail', '-1', fn]).split()[0])
                check_population = popGids[(popGids.minGID <= check) & (popGids.maxGID >= check)].index.tolist()[0]
                assert exact_population_tmp == check_population

                new_name = os.path.join(
                        self.rec_folder,
                        '_'.join(exact_population) + '.isyn'
                        )
                os.rename(fn, new_name)
            if self.logger:
                self.logger.debug('Assigning population names to I_syn files took %s s', round(time.time() - ts, 3))

            # 5) stream each population file once; group by gid on the fly
            rows = []
            dem_files = glob.glob(os.path.join(self.rec_folder, '*.isyn'))
            for isyn_file in dem_files:
                area, layer_code, pop, *_ = os.path.basename(isyn_file).replace('.', '_').split('_')
                layer = d_population_backward[layer_code]

                list_ex, list_in = [], []
                current_gid = None
                tmp_ex, tmp_in = [], []

                with open(isyn_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 4:
                            continue  # skip headers/empty lines
                        try:
                            gid = int(parts[0])
                            ex_val = float(parts[2])
                            in_val = float(parts[3])
                        except ValueError:
                            continue  # skip malformed lines

                        if current_gid is None:
                            current_gid = gid
                        if gid != current_gid:
                            list_ex.append(np.array(tmp_ex, dtype=float))
                            list_in.append(np.array(tmp_in, dtype=float))
                            tmp_ex, tmp_in = [], []
                            current_gid = gid

                        tmp_ex.append(ex_val)
                        tmp_in.append(in_val)

                if current_gid is not None:
                    list_ex.append(np.array(tmp_ex, dtype=float))
                    list_in.append(np.array(tmp_in, dtype=float))

                rows.append({
                    'area': area,
                    'layer': layer,
                    'pop': pop,
                    'I_syn_ex': list_ex,
                    'I_syn_in': list_in
                })
            I_syn = pd.DataFrame(rows)

        return I_syn

    def getHash(self):
        """
        Creates a hash from analysis parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        hash_ = dicthash.generate_hash_from_dict(self.ana_dict)
        return hash_

    def dump(self, base_folder):
        """
        Exports the full analysis specification. Creates a subdirectory of
        base_folder from the analysis hash where it puts all files.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        hash = self.getHash()
        out_folder = os.path.join(base_folder, hash)
        try:
            os.mkdir(out_folder)
        except OSError:
            pass

        # output simple data as yaml
        fn = os.path.join(out_folder, 'ana.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.ana_dict, outfile, default_flow_style=False)


def calc_power_spectrum(signal, fs, nperseg=None):
    """
    Compute one-sided power spectral density of a signal using Welch's
    method (Hann window, density scaling).

    Parameters
    ----------
    signal : array-like, shape (N,)
        Input time series (should be zero-mean).
    fs : float
        Sampling frequency in Hz.
    nperseg : int or None
        Segment length.  Defaults to the full signal length N, i.e. a
        single Hann-tapered segment spanning the whole window.  This
        maximises frequency resolution: Δf = fs / nperseg is then fs / N,
        the finest attainable from the data, with N // 2 + 1 points spanning
        DC up to and including the Nyquist frequency (fs/2) - as dense a
        frequency axis as the recording allows. It comes at the cost of
        Welch's usual variance reduction from averaging multiple segments;
        pass a smaller value (e.g. N // 4) to trade resolution for a
        smoother estimate.

    Returns
    -------
    freqs : np.ndarray
        Frequencies in Hz, from 0 to fs/2, spaced by fs/nperseg.
    power : np.ndarray
        Power spectral density in V²/Hz (scipy 'density' normalisation).
    """
    signal = np.asarray(signal, dtype=float)
    if nperseg is None:
        nperseg = len(signal)
    nperseg = min(nperseg, len(signal))
    freqs, power = welch(signal, fs=fs, nperseg=nperseg,
                         window='hann', scaling='density')
    return freqs, power


def _delay_signal(arr, dt, delay_ms):
    """
    Shift a time series forward in time by `delay_ms`, i.e. return x(t - delay_ms)
    sampled on the same time grid. Operates along the last axis; samples before
    the shift have no recorded history and are zero-filled.

    dt : float
        Sampling interval of `arr` along its last axis, in ms.
    """
    shift = int(round(delay_ms / dt))
    if shift <= 0:
        return arr
    shifted = np.zeros_like(arr)
    shifted[..., shift:] = arr[..., :-shift]
    return shifted


def _fc_similarity(mat_a, mat_b):
    """
    Pearson correlation and cosine similarity between two square FC matrices
    of matching shape and row/column ordering, computed on the vectorized
    upper triangle (off-diagonal, k=1) so the diagonal (always 1) and the
    redundant lower triangle don't inflate the similarity.

    Returns
    -------
    pearson_r : float
    cosine_sim : float
    """
    a = np.asarray(mat_a)
    b = np.asarray(mat_b)
    iu = np.triu_indices_from(a, k=1)
    va, vb = a[iu], b[iu]
    pearson_r = np.corrcoef(va, vb)[0, 1]
    cosine_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
    return pearson_r, cosine_sim


def calc_wavelet_scalogram(signal, fs, fmin=1.0, fmax=100.0, n_scales=64, w=6.0):
    """
    Compute a Morlet continuous wavelet transform scalogram.

    Uses ``scipy.signal.morlet2`` with logarithmically spaced frequencies,
    giving finer time resolution at high frequencies and finer frequency
    resolution at low frequencies — unlike the fixed-resolution STFT.

    Parameters
    ----------
    signal : array-like, shape (N,)
        Input time series.
    fs : float
        Sampling frequency in Hz.
    fmin : float
        Lowest frequency of interest in Hz.
    fmax : float
        Highest frequency of interest in Hz.
    n_scales : int
        Number of frequency scales (log-spaced between fmin and fmax).
    w : float
        Morlet wavelet parameter controlling the trade-off between time and
        frequency resolution.  Higher values increase frequency resolution at
        the cost of time resolution.  Default 6.0 is standard for LFP.

    Returns
    -------
    freqs : np.ndarray, shape (n_scales,)
        Centre frequencies in Hz, log-spaced from fmin to fmax.
    t : np.ndarray, shape (N,)
        Time axis in seconds (same length as input signal).
    power : np.ndarray, shape (n_scales, N)
        Instantaneous power (squared modulus of CWT coefficients).
        Units are signal-units squared (not dB).
    """
    from scipy.signal import fftconvolve

    def _morlet2(M, s, w):
        """Morlet wavelet, matching the removed scipy.signal.morlet2."""
        x = (np.arange(M) - (M - 1) / 2.0) / s
        return (np.pi ** -0.25) * np.sqrt(1.0 / s) * np.exp(1j * w * x) * np.exp(-0.5 * x ** 2)

    signal = np.asarray(signal, dtype=float)
    n      = len(signal)
    freqs  = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    scales = w * fs / (2.0 * np.pi * freqs)

    coef = np.empty((n_scales, n), dtype=complex)
    for i, scale in enumerate(scales):
        M = min(int(10 * scale), n)
        wavelet_data = np.conj(_morlet2(M, s=scale, w=w)[::-1])
        coef[i] = fftconvolve(signal, wavelet_data, mode='same')

    power = np.abs(coef) ** 2
    t     = np.arange(n) / fs
    return freqs, t, power


def hilbert_envelope(signal):
    """
    Compute the instantaneous amplitude (envelope) and instantaneous phase
    of a real signal via the Hilbert transform.

    Parameters
    ----------
    signal : array-like, shape (N,)
        Input time series (should be zero-mean / band-passed for meaningful results).

    Returns
    -------
    envelope : np.ndarray
        Instantaneous amplitude of the analytic signal.
    phase : np.ndarray
        Instantaneous phase in radians.
    """
    analytic = hilbert(np.asarray(signal, dtype=float))
    return np.abs(analytic), np.angle(analytic)


def cvIsi(sts, t_start=None, t_stop=None, CV_min_spikes=2, take_mean=True):
    """
    Calculates the cv isi for a subset of neurons for a list of spiketrains.
    The spiketrains contained in the list of spiketrains have to contain at
    least 2 spikes for calculating the interspike interval isi. If this
    condition is not met this function returns 0.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    cv : float
        cv_isi
    """
    # ensure that there are only spiketrains of len > 1. This is
    # for np.diff which needs to output at least one value in order
    # for np.std and np.mean to return something which is not
    # np.nan. Otherwise return 0.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) >= CV_min_spikes]
    if len(sts) > 0:
        isi = [np.diff(x, 1) for x in sts]
        cv = [np.std(x) / np.mean(x) for x in isi]
        if take_mean:
            return np.mean(cv)
        else:
            return cv
    return 0.

def calculate_lv(isi, t_ref):
    """
    Calculates the local variation lv for a interspike interval distribution.

    Parameters
    ----------
    isi : list
        list of interspike intervals isi of a single neuron

    Returns
    -------
    lv : float
        lv
    """
    # NOTE Elephant and mam use different functions. Elephant uses the normal
    # local variation whereas mam uses the revised local variation. LV depends
    # on firing rate fluctuations which are caused by the refractory period.
    # This can be compensated for by subtracting the refractoriness constant,
    # t_ref, from the ISIs.
    # Here we take the revised local variation.
    # Multi area model function
    val = np.sum(
            (1. - 4 * isi[:-1] * isi[1:] / (isi[:-1] + isi[1:]) ** 2) \
                    * (1 + 4 * t_ref / (isi[:-1] + isi[1:]))
                    ) * 3 / (isi.size - 1.)
    # Elephant function
    # val = 3. * np.mean(np.power(np.diff(isi) / (isi[:-1] + isi[1:]), 2))
    return val

def LV(sts, t_ref, t_start=None, t_stop=None, LV_min_spikes=3, take_mean=True):
    """
    Calculates the local variation lv for a list of spiketrains sts. First we
    filter for spiktrains of length > 2 because otherwise the calculation
    fails. In this case we return 0. At the end we divide by the number of
    spiketrains, opposed to dividing by the number of neurons in a population.
    This way we take only neurons that have actually spiked into account.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    lv : float
        sum of single lvs, needs to normalized (=divided by neuron numbers)
    """
    # ensure that there are only spiketrains of len > 2.
    # So every spiketrain st in sts has len(st) > 1.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) >= LV_min_spikes]
    if len(sts) > 0:
        isi = [np.diff(x, 1) for x in sts]
        lvr = [calculate_lv(x, t_ref) for x in isi]
        if take_mean:
            return np.mean(lvr)
        else:
            return lvr
    return 0.

def calc_rates(sts, sim_dict, ana_dict):
    """
    Calculates the histogram of rates of a list of spiketrains sts.
    NOTE: The units of the returned rates are in spikes / ms

    Returns
    -------
    rate : ndarray
        array of binned rates
    """
    resolution = ana_dict['rate_histogram']['binsize']
    t_min = 0.
    t_max = ana_dict['rate_histogram']['t_stop']
    if t_min is None:
        t_min = 0.
    if t_max is None:
        t_max = sim_dict['t_sim']
    num_bins = int((t_max - t_min) / resolution)
    if len(sts) > 0:
        # Gives same output as:
        # elstat.time_histogram(
        #     sts, binsize=resolution*pq.ms, output='counts'
        # )
        counts, _ = np.histogram(
                np.concatenate(sts).ravel(),
                bins=num_bins,
                range=(t_min, t_max)  # or range=(t_min + resolution / 2., t_max + resolution / 2.)
                )
        rate = counts * 1. / resolution
        return rate
    return np.zeros(num_bins)

def correlation(sts, ana_dict, sim_dict):
    """
    Calculates the correlation coefficients for a subset of neurons for a
    list of spiketrains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    cc : float
        Correlation coefficient
    """
    subsample = ana_dict['correlation_coefficient']['subsample']
    _, hist = instantaneous_spike_count(sts, ana_dict, sim_dict)
    rates = strip_binned_spiketrains(hist)[:subsample]
    # Need at least 2 spiketrains
    if len(rates) > 1:
        cc = np.corrcoef(rates)
        cc = np.extract(1-np.eye(cc[0].size), cc)
        cc[np.where(np.isnan(cc))] = 0.
        return np.mean(cc)
    return 0.

def instantaneous_spike_count(data, ana_dict, sim_dict):
    '''
    Create a histogram of spike trains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    bins : np.array
        Bins
    hist : np.array
        Histogram
    '''
    tbin = ana_dict['correlation_coefficient']['tbin']
    tmin = ana_dict['correlation_coefficient']['tmin']
    tmax = ana_dict['correlation_coefficient']['tmax']
    if tmin is None:
        tmin = 0.
    if tmax is None:
        tmax = sim_dict['t_sim']
    assert(tmin < tmax)
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist

def strip_binned_spiketrains(sp):
    '''
    Removes binned spiketrains which do not contain a single spike
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    sp : np.array
        Array containing histogram.

    Returns
    -------
    sp_stripped : np.array
        Binned spiketrains with empty spiketrains removed.
    '''
    sp_stripped = np.array(
            [x for x in sp if abs(np.max(x) - np.min(x)) > 1e-16]
            )
    return sp_stripped

def shell_presort_dat(fn):
    """
    Presort a single .dat file using GNU sort.
    Header lines (non-numeric first character) are filtered automatically,
    so this works for both spike and Isyn .dat files regardless of header count.

    Parameters
    ----------
    fn : str
        Path to the .dat file to sort.
    """
    subprocess.check_output(
            f"export LC_ALL=C; f={fn}; grep -E '^[0-9]' ${{f}} | sort -k1,1n -k2,2n --parallel=8 > ${{f%.dat}}_sorted.txt",
            shell=True
            )

def shell_spiketrainify(fn):
    """
    This function is used to sort and organize the spiking data into
    spiketrains per neuron.

    Parameters
    ----------
    fn : str
        The filename to be processed.
    """
    lol2 = '''
    awk '
    {
      if($1==k)
      printf("%s"," ")
      else {
          if(NR!=1)
          print ""
        printf("%s\t",$1)
      }
      for(i=2;i<NF;i++)
        printf("%s ",$i)
      printf("%s",$NF)
      k=$1
    }
    END{
    print ""
    }' '''
    lol3 = f'{fn} > {fn}.sorted'
    lol4 = f'''; cut -d$'\t' -f2 {fn}.sorted > {fn}.sorted.cut'''
    # lol5 = f'; rm {fn} {fn}.sorted'
    lol = lol2 + lol3 + lol4  # + lol5

    subprocess.check_output(
            lol,
            shell=True
            )

def split_sorted_file(df, fn, iteration, rec_folder):
    """
    Split a sorted text file into two halves based on GID range.

    Uses awk to locate the split line and csplit to divide the file.
    Intermediate files (whose names contain ``_left_`` or ``_right_``) are
    deleted after splitting; the original merged file is preserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Population GID table for the current chunk (must have maxGID column).
    fn : str
        Basename of the file to split (relative to rec_folder).
    iteration : int
        Current recursion depth, embedded in output filenames.
    rec_folder : str
        Directory that contains fn and receives the two output files.

    Returns
    -------
    tuple of (df_left, fn_left, df_right, fn_right, iteration) or None
    """
    if len(df) <= 1:
        return None

    where_to_split = math.floor(len(df) / 2)
    df_left = df.iloc[:where_to_split]
    df_right = df.iloc[where_to_split:]
    maxGID = df_left.iloc[-1].maxGID

    fn_left = fn + f"_left_{iteration}"
    fn_right = fn + f"_right_{iteration}"

    full_fn = os.path.join(rec_folder, fn)
    prefix = os.path.join(rec_folder, f"part.{fn}.")

    a = subprocess.check_output(
        f"awk '$1>{maxGID}{{print NR; exit}}' {full_fn}",
        shell=True,
        text=True
    ).strip()

    if not a:
        return None

    subprocess.check_output(
        f"csplit -sf {prefix} {full_fn} {a}",
        shell=True
    )

    part0 = os.path.join(rec_folder, f"part.{fn}.00")
    part1 = os.path.join(rec_folder, f"part.{fn}.01")

    if not (os.path.exists(part0) and os.path.exists(part1)):
        return None

    if os.path.getsize(part0) == 0 or os.path.getsize(part1) == 0:
        if os.path.exists(part0):
            os.remove(part0)
        if os.path.exists(part1):
            os.remove(part1)
        return None

    os.rename(part0, os.path.join(rec_folder, fn_left))
    os.rename(part1, os.path.join(rec_folder, fn_right))

    if '_left_' in fn or '_right_' in fn:
        os.remove(full_fn)

    return df_left, fn_left, df_right, fn_right, iteration

def kernel_for_psc(tau_s, dt):
    """
    Calculates the kernel used in the calculation of postsynaptic currents.

    Parameters
    ----------
    tau_s : float
        Synaptic time constant
    dt : float
        Simulation resolution

    Returns
    -------
    kernel : np.array
    """
    # Calculate exponential kernel for PSCs
    t_ker = np.arange(-10*tau_s, 10*tau_s, dt)
    kernel = np.exp(- t_ker / tau_s) / tau_s
    kernel[t_ker < -dt/2] = 0  # Make filter causal
    kernel /= kernel.sum()  # Normalize filter
    return kernel

def analysisDictFromDump(dump_folder):
    """
    Creates a analysis dict from the files created by Analysis.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    ana_dict : dict
        Full analysis dictionary
    """
    fn = os.path.join(dump_folder, 'ana.yaml')
    with open(fn, 'r') as ana_file:
        ana_dict = yaml.load(ana_file)
    return ana_dict
