# -*- coding: utf-8 -*-
import nest
import numpy as np
import os
from datetime import datetime
import yaml
from dicthash import dicthash

from helpers.lognormal import mu_sigma_lognorm
from helpers.neuron_types import BASE_TYPE, POPULATIONS, type_fractions

class Simulation():
    """
    Handles the setup of the network parameters and
    provides functions to connect the network and devices.

    Parameters
    ----------
    sim_dict : dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: default_sim_params.py)
    net_dict : dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)
    """

    def __init__(self, sim_dict, net_dict, main_logger=None, rank_logger=None):
        """
        Parameters
        ----------
        sim_dict : dict
            Simulation parameters
        net_dict : dict
            Network parameters
        main_logger : logging.Logger or None
            Logger for rank 0 output. None for non-rank-0 processes.
        rank_logger : logging.Logger or None
            Logger for rank-specific output.
        """
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.logger = main_logger
        self.rank_logger = rank_logger


    def set_data_path(self, data_path):
        """
        Sets the path for the output files.

        Parameters
        ----------
        data_path : string
        """
        self.data_path = data_path
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            self.spike_path = os.path.join(self.data_path, 'spikes')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            self.volt_path = os.path.join(self.data_path, 'voltages')
        if "I_syn" in self.sim_dict["rec_dev"]:
            self.I_syn_path = os.path.join(self.data_path, "I_syn")

        if self.logger:
            if os.path.isdir(data_path):
                self.logger.info('data directory already exists')
            else:
                os.mkdir(data_path)
                self.logger.info('data directory created')
            self.logger.info('Data will be written to %s', self.data_path)

            if 'spike_recorder' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.spike_path):
                    os.mkdir(self.spike_path)
            if 'voltmeter' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.volt_path):
                    os.mkdir(self.volt_path)
            if "I_syn" in self.sim_dict["rec_dev"]:
                if not os.path.isdir(self.I_syn_path):
                    os.mkdir(self.I_syn_path)

    def setup_nest(self, num_threads):
        """
        Hands parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        Parameters
        ----------
        num_threads : int
            Local number of threads (per MPI process).
        """
        nest.ResetKernel()
        master_seed = self.sim_dict['master_seed']
        if self.logger:
            self.logger.info('Master seed: %i', master_seed)
        nest.SetKernelStatus(
            {'local_num_threads': num_threads}
            )
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        if self.logger:
            self.logger.info('Number of total processes: %i', N_tp)

        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'rng_seed': master_seed,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time'],
            }
        nest.SetKernelStatus(kernel_dict)

    def _neuron_type_spec(self, name, pop_type):
        """
        Returns model, parameters and parameter distribution of a neuron type.

        Parameters
        ----------
        name : str
            Name of a type in ``net_dict['neuron_types']`` or ``BASE_TYPE``.
        pop_type : str
            ``'E'`` or ``'I'``.

        Returns
        -------
        neuron_model : str
        neuron_params : dict
        param_dist : dict
        """
        if pop_type not in POPULATIONS:
            msg = 'Populations have to be E or I'
            self.rank_logger.error(msg)
            raise NotImplementedError(msg)
        if name == BASE_TYPE:
            spec = self.net_dict
        else:
            spec = self.net_dict['neuron_types'][name]
        return (spec['neuron_model_' + pop_type],
                spec['neuron_params_' + pop_type],
                spec['neuron_param_dist_' + pop_type])

    def _split_population(self, pop, nn):
        """
        Splits a population into its neuron types.

        Parameters
        ----------
        pop : tuple
            ``(area, layer, population)`` index.
        nn : int
            Number of neurons in the population.

        Returns
        -------
        list of (str, int)
            ``(name, number)`` for every custom type in declaration order,
            followed by the remaining ``BASE_TYPE`` neurons.
        """
        nn = int(nn)
        parts = []
        assigned = 0
        fractions = type_fractions(
            self.net_dict.get('neuron_types', {}), pop[0], pop[1], pop[-1])
        for name, frac in fractions:
            n = min(int(np.round(nn * frac)), nn - assigned)
            parts.append((name, n))
            assigned += n
        parts.append((BASE_TYPE, nn - assigned))
        return parts

    def _type_dc(self, pop, name):
        """
        DC input of the neurons of one type in a population.

        Returns
        -------
        float or None
            The entry of ``net_dict['dc_drive_types']``, or None for the base
            neurons and when the network has no per-type DC (no downscaling),
            in which case the population's ``dc_drive`` applies.
        """
        dc_types = self.net_dict.get('dc_drive_types')
        if name == BASE_TYPE or dc_types is None or name not in dc_types:
            return None
        return float(dc_types.loc[pop, name])

    def _set_param_dist(self, nodes, neuron_params, param_dist):
        """
        Draws neuron parameters from their distributions.

        Parameters
        ----------
        nodes : NodeCollection
            Neurons to set the parameters of.
        neuron_params : dict
            Mean parameters of the neurons.
        param_dist : dict
            ``{param: {'distribution': str, 'rel_sd': float}}``
        """
        for prm, dist_dict in param_dist.items():
            if dist_dict['rel_sd'] > 0:
                param_dist = dist_dict['distribution']
                if param_dist == 'lognormal':
                    mean_prm = neuron_params[prm]
                    offset_prm = 0.
                    if prm == 'V_th':
                        mean_prm -= neuron_params['E_L']
                        offset_prm += neuron_params['E_L']
                    assert mean_prm > 0
                    mu_param, sigma_param = mu_sigma_lognorm(
                        mean=mean_prm,
                        rel_sd=dist_dict['rel_sd']
                    )
                    nodes.set({prm:
                        offset_prm + nest.random.lognormal(
                        mean=mu_param,
                        std=sigma_param
                    )})
                else:
                    err_msg = "Parameter distribution "
                    err_msg += f"{param_dist} not implemented."
                    self.rank_logger.error(err_msg)
                    raise NotImplementedError(err_msg)

    def create_populations(self):
        """
        Creates the neuronal populations.

        Every population is split into the neuron types given in
        ``net_dict['neuron_types']`` (in declaration order) and the remaining
        base neurons. The parts are created one after another, so each
        population occupies one contiguous GID range, and receive the
        parameters of their type. DC input and the initial membrane potential,
        drawn from a normal distribution, are assigned to the whole population.

        Writes ``population_GIDs.dat`` (``pop;minGID;maxGID``) and
        ``neuron_type_GIDs.dat`` (``pop;type;minGID;maxGID;count``, one row per
        non-empty type of every population, base neurons included).
        """
        self.rank_logger.info('Memory on rank %d before creating populations: %.2fMB', nest.Rank(), self._getMemoryMB())
        self.pops = {}
        type_totals = {}
        pop_rows = []
        type_rows = []
        for pop, nn in self.net_dict['neuron_numbers'].items():
            if nn > 0:
                # Create the neuron types one after another
                parts = []
                for name, n in self._split_population(pop, nn):
                    if n > 0:
                        neuron_model, neuron_params, param_dist = \
                            self._neuron_type_spec(name, pop[-1])
                        nodes = nest.Create(neuron_model, n)
                        nodes.set(neuron_params)
                        parts.append((name, nodes, neuron_params, param_dist))
                        type_totals[name] = type_totals.get(name, 0) + n
                        type_rows.append((pop, name, nodes[0].global_id,
                                          nodes[-1].global_id, n))
                population = parts[0][1]
                for _, nodes, _, _ in parts[1:]:
                    population = population + nodes

                # Assign DC input; neuron types with their own synaptic time
                # constants have their own downscaling compensation
                nest.SetStatus(
                    population, {
                        'I_e': self.net_dict['dc_drive'].loc[pop]
                    }
                )
                for name, nodes, _, _ in parts:
                    dc = self._type_dc(pop, name)
                    if dc is not None:
                        nodes.set({'I_e': dc})

                population.V_m = nest.random.normal(
                    mean=self.sim_dict['V0_mean'],
                    std=self.sim_dict['V0_sd']
                )
                # Neuron parameters
                for _, nodes, neuron_params, param_dist in parts:
                    self._set_param_dist(nodes, neuron_params, param_dist)

                self.rank_logger.debug(
                    '[%s] total=%d | %s', pop, nn,
                    ' | '.join('{}={}'.format(name, len(nodes))
                               for name, nodes, _, _ in parts))

                self.pops[pop] = population
                pop_rows.append((pop, population[0].global_id,
                                 population[-1].global_id))

        if nest.Rank() == 0:
            self._write_gid_files(pop_rows, type_rows)
        if self.logger:
            self.logger.info(
                'Created %d populations: %s', len(self.pops),
                ', '.join('{} {}'.format(n, name)
                          for name, n in type_totals.items()))
        self.rank_logger.info('Memory on rank %d after creating populations: %.2fMB', nest.Rank(), self._getMemoryMB())

    def _write_gid_files(self, pop_rows, type_rows):
        """
        Writes the GID ranges of populations and neuron types.

        Parameters
        ----------
        pop_rows : list of tuple
            ``(pop, minGID, maxGID)`` per population.
        type_rows : list of tuple
            ``(pop, type, minGID, maxGID, count)`` per neuron type and
            population.
        """
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        with open(pop_file_name, 'w') as pop_file:
            for row in pop_rows:
                pop_file.write('{};{};{}\n'.format(*row))
        type_file_name = os.path.join(self.data_path, 'neuron_type_GIDs.dat')
        with open(type_file_name, 'w') as type_file:
            for row in type_rows:
                type_file.write('{};{};{};{};{}\n'.format(*row))

    def create_populations_pos(self):
        """
        Creates spatially-positioned populations for topographic mapping.

        Every population is created as a grid of base neurons. For each neuron
        type a random, disjoint subset of the grid then receives the
        parameters of that type; NEST cannot change the model of existing
        nodes, so a type's model has to equal the base model here.

        TODO: Topographic mapping is out of scope for now. This method is not
        exercised by the default pipeline (``topographic_mapping`` defaults to
        False) and needs revisiting before it can be relied upon.
        """
        self.rank_logger.info('Memory on rank %d before creating populations: %.2fMB',
            nest.Rank(), self._getMemoryMB())

        self.pops = {}
        pop_rows = []
        type_rows = []
        random_seed_generator = np.random.default_rng(seed=self.sim_dict["master_seed"])

        for pop, nn in self.net_dict['neuron_numbers'].items():
            if nn > 0:
                neuron_model_pop, neuron_params_pop, nrn_prm_dist_pop = \
                    self._neuron_type_spec(BASE_TYPE, pop[-1])

                shape = self.generate_shape(pop, nn)
                center = [0., 0., self.net_dict["thickness"].loc[pop[:-1]] / 2]
                extent = [1., 1., 2.]

                pop_grid = nest.Create(neuron_model_pop, positions=nest.spatial.grid(shape=shape, center=center, extent=extent))
                pop_grid.set(neuron_params_pop)
                head = pop_grid[0].global_id
                tail = pop_grid[-1].global_id
                nn_grid = shape[0] * shape[1] * shape[2]

                nest.SetStatus(
                    pop_grid, {
                        'I_e': self.net_dict['dc_drive'].loc[pop]
                    }
                )
                pop_grid.V_m = nest.random.normal(
                    mean=self.sim_dict['V0_mean'],
                    std=self.sim_dict['V0_sd']
                )

                # Randomly assign disjoint subsets of the grid to the types
                random_id_generator = np.random.default_rng(seed=int(random_seed_generator.uniform(0, 1) * 1000))
                order = head + random_id_generator.permutation(nn_grid)
                start = 0
                is_base = np.ones(nn_grid, dtype=bool)
                for name, n in self._split_population(pop, nn_grid):
                    if name == BASE_TYPE or n == 0:
                        continue
                    neuron_model, neuron_params, param_dist = \
                        self._neuron_type_spec(name, pop[-1])
                    if neuron_model != neuron_model_pop:
                        msg = ('Topographic mapping requires neuron type {} to '
                               'use the base model {}.'.format(name, neuron_model_pop))
                        self.rank_logger.error(msg)
                        raise NotImplementedError(msg)
                    gids = np.sort(order[start:start + n])
                    start += n
                    is_base[gids - head] = False
                    nodes = nest.NodeCollection(gids.tolist())
                    nodes.set(neuron_params)
                    dc = self._type_dc(pop, name)
                    if dc is not None:
                        nodes.set({'I_e': dc})
                    self._set_param_dist(nodes, neuron_params, param_dist)
                    type_rows.append((pop, name, gids.tolist()))
                base_gids = head + np.flatnonzero(is_base)
                if base_gids.size:
                    self._set_param_dist(
                        nest.NodeCollection(base_gids.tolist()),
                        neuron_params_pop, nrn_prm_dist_pop)

                self.pops[pop] = pop_grid
                pop_rows.append((pop, head, tail))

        if nest.Rank() == 0:
            pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
            with open(pop_file_name, 'w') as pop_file:
                for row in pop_rows:
                    pop_file.write('{};{};{}\n'.format(*row))
            # Types are scattered over the grid, so their GIDs are listed
            # explicitly instead of as ranges.
            type_file_name = os.path.join(self.data_path, 'neuron_type_GIDs_pos.dat')
            with open(type_file_name, 'w') as type_file:
                for row in type_rows:
                    type_file.write('{};{};{}\n'.format(*row))
        if self.logger:
            self.logger.info('Created %d spatial populations', len(self.pops))
        self.rank_logger.info('Memory on rank %d after creating populations: %.2fMB',
            nest.Rank(), self._getMemoryMB())

    def generate_shape(self, pop_name, nn_pop):
        """
        Create and save positions of population.

        TODO: Topographic mapping is out of scope for now. Helper for
        ``create_populations_pos``; revisit alongside it.
        """
        grid = [0, 0, 0]
        # print(self.net_dict["thickness"])
        # print(pop_name[:-1])
        h = self.net_dict["thickness"].loc[pop_name[:-1]]
        rho_N = (nn_pop / h) ** (1/3)
        grid[0] = int(np.round(rho_N))
        grid[1] = int(np.round(rho_N))
        grid[2] = int(np.round(rho_N * h))
        return grid

    def save_position(self, pop, file):
        """
        Save position and GID in an area.

        TODO: Topographic mapping is out of scope for now. Unimplemented stub;
        revisit alongside ``create_populations_pos``.
        - To save GID-position pairs for populations.
        - To save file for every region.
        """
        pass

    def create_devices(self):
        """
        Creates the recording devices.

        Only devices which are given in sim_dict['rec_dev'] are created.
        """
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            recdict = {
                'record_to': 'ascii',
                'label': os.path.join(self.spike_path, 'spike_recorder')
            }
            self.spike_recorder = nest.Create('spike_recorder', params=recdict)
        if 'voltmeter' in self.sim_dict['rec_dev']:
            recdictmem = {
                'record_to': 'ascii',
                'label': os.path.join(self.volt_path, 'voltmeter'),
                'record_from': ['V_m'],
            }
            self.voltmeter = nest.Create('multimeter', params=recdictmem)
        if "I_syn" in self.sim_dict["rec_dev"]:
            recdictmem = {
                "record_to": "ascii",
                "label": os.path.join(self.I_syn_path, "I_syn"),
                "record_from": ["I_syn_ex", "I_syn_in"]
            }
            self.I_syn_recorder = nest.Create("multimeter", params=recdictmem)

        if 'spike_recorder' in self.sim_dict['rec_dev']:
            if self.logger:
                self.logger.info('Spike detectors created')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            if self.logger:
                self.logger.info('Voltmeters created')
        if "I_syn" in self.sim_dict["rec_dev"]:
            if self.logger:
                self.logger.info("Synaptic current recoreders created")

    def create_poisson(self):
        """
        Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.
        """
        if self.logger:
            self.logger.info('Poisson background input created')
        self.poisson = {}
        for pop, nn in self.net_dict['neuron_numbers'].items():
            if nn > 0.:
                sn_ext = self.net_dict['synapses_external'].loc[pop]
                K_ext = sn_ext / nn
                rate_ext = self.net_dict['rate_ext'].loc[pop] * K_ext
                poiss = nest.Create('poisson_generator')
                poiss.set({'rate': rate_ext})
                self.poisson[pop] = poiss

    def create_single_spike(self):
        """
        Creates the single spike generator.
        """
        if self.logger:
            self.logger.info('Single spike input created')
        self.single_spike = {}
        for pop, spike_time in self.net_dict['spike_time'].items():
            if spike_time >= 0.:
                spike = nest.Create(
                        'spike_generator',
                        params={'spike_times': [spike_time]}
                        )
                self.single_spike[pop] = spike

    def create_stimuli(self):
        """
        Creates stimulus input.
        """
        if self.logger:
            self.logger.info("Stimuli input created")
        self.stimuli = {}
        for pop, stimulus in self.sim_dict["stimuli"].items():
            if stimulus["type"] == "ac":
                stim = nest.Create("ac_generator", params=stimulus["params"])
                # TODO complete other input types.
                self.stimuli[pop] = stim

    def connect_neurons(self):
        """
        Connects the neuronal populations.
        """
        conn_rule = self.sim_dict['connection_rule']
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            for (area_j, layer_j, pop_j), source_pop in self.pops.items():
                synapse_nr = self.net_dict['synapses_internal'].loc[
                    (area_i, layer_i, pop_i),
                    (area_j, layer_j, pop_j)
                ]

                if synapse_nr > 0.:
                    if area_i == area_j:
                        min_delay = self.sim_dict['sim_resolution']
                        if pop_j == 'E':
                            mean_delay = self.net_dict['delay_e']
                            std_delay = self.net_dict['delay_e_sd']
                        else:
                            mean_delay = self.net_dict['delay_i']
                            std_delay = self.net_dict['delay_i_sd']
                    else:
                        min_delay = max(self.sim_dict['delay_cc_min'],
                                        self.sim_dict['sim_resolution'])
                        mean_delay = self.net_dict['delay_cc'].loc[
                            area_i, area_j
                        ]
                        std_delay = self.net_dict['delay_cc_sd'].loc[
                            area_i, area_j
                        ]
                    delay_distr = self.net_dict['delay_distribution']
                    if delay_distr == 'normal_clipped':
                        mu_delay = mean_delay
                        sigma_delay = std_delay
                    elif delay_distr == 'lognormal_clipped':
                        mu_delay, sigma_delay = mu_sigma_lognorm(
                            mean=mean_delay, rel_sd=std_delay/mean_delay
                        )
                    else:
                        err_msg = f"Delay distribution {delay_distr} not implemented."
                        self.rank_logger.error(err_msg)
                        raise NotImplementedError(err_msg)
                    weight = self.net_dict['weights'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    w_sd = self.net_dict['weights_sd'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    if conn_rule == 'fixed_total_number':
                        conn_dict_rec = {
                            'rule': conn_rule, 'N': synapse_nr
                        }
                    elif conn_rule == 'fixed_indegree':
                        neuron_nr = self.net_dict['neuron_numbers'].loc[
                            (area_i, layer_i, pop_i)
                        ]
                        indegree = int(np.round(synapse_nr / neuron_nr))
                        conn_dict_rec = {
                            'rule': conn_rule, 'indegree': indegree
                        }
                    elif conn_rule == "pairwise_bernoulli":
                        try:
                            p_center = synapse_nr / (self.net_dict["neuron_numbers"].loc[(area_i, layer_i, pop_i)] * self.net_dict["neuron_numbers"].loc[(area_j, layer_j, pop_j)])
                            sigma_x = self.sim_dict["sigma_x"] / 1000.
                            sigma_y = self.sim_dict["sigma_y"] / 1000.
                            conn_dict_rec = {
                                "rule": conn_rule,
                                'p': p_center * nest.spatial_distributions.gaussian2D(
                         nest.spatial.distance.x, nest.spatial.distance.y, std_x=sigma_x, std_y=sigma_y),
                            }
                        except KeyError:
                            self.rank_logger.warning('Spatial connection parameter required but not found')

                    else:
                        msg = f'Unknown connection rule {conn_rule}.'
                        self.rank_logger.error(msg)
                        raise NotImplementedError(msg)
                    if np.isclose(self.net_dict['p_transmit'], 1):
                        syn_dict = {'synapse_model': 'static_synapse'}
                    else:
                        syn_dict = {'synapse_model': 'bernoulli_synapse',
                                    'p_transmit': self.net_dict['p_transmit']}
                        
                    if delay_distr == 'normal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.normal(mean=mu_delay, std=sigma_delay), min_delay)
                    elif delay_distr == 'lognormal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.lognormal(mean=mu_delay, std=sigma_delay), min_delay)

                    # Skipping connection with weight == 0.0
                    if weight == 0.0:
                        self.rank_logger.warning(
                            'Found weight == 0.0 between %s and %s. Skipping connection.',
                            (area_i, layer_i, pop_i), (area_j, layer_j, pop_j)
                        )
                        continue
                    
                    if weight < 0:
                        syn_dict['weight'] = nest.math.min(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    else:
                        syn_dict['weight'] = nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict
                    )
            self.rank_logger.debug(
                'Connected all of area %s, layer %s and population %s on rank %d. Memory: %.2f MB.',
                area_i, layer_i, pop_i, nest.Rank(), self._getMemoryMB()
            )
        self.rank_logger.info('Memory on rank %d after creating connections: %.2fMB', nest.Rank(), self._getMemoryMB())
        
        if self.logger:
            self.logger.info('Connections are established')

    def connect_poisson(self):
        """
        Connects the Poisson generators to the microcircuit.
        """
        if self.logger:
            self.logger.info('Poisson background input is connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            conn_dict_poisson = {'rule': 'all_to_all'}
            weight = self.net_dict['weights_ext'].loc[
                (area_i, layer_i, pop_i)
            ]
            w_sd = self.net_dict['weights_ext_sd'].loc[
                (area_i, layer_i, pop_i)
            ]
            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0),
                'delay': self.sim_dict['sim_resolution']
            }
            nest.Connect(
                self.poisson[(area_i, layer_i, pop_i)], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
            )

    def connect_single_spike(self):
        """
        Connects a spike generator emitting a single spike.
        """
        if self.logger:
            self.logger.info('Single spike generator input is connected')
        # Loop over all existing
        for pop, spike in self.single_spike.items():
            # Calculate indegree K for scaling weight
            nn = self.net_dict['neuron_numbers'].loc[pop]
            sn_ext = self.net_dict['synapses_external'].loc[pop]
            K_ext = sn_ext / nn
            # Choose first item in list of target gids
            target_pop = [self.pops[pop][0]]
            weight = self.net_dict['weights_ext'].loc[pop]
            # Scale weight
            weight *= 1e3*K_ext
            syn_dict_single_spike = {
                'synapse_model': 'static_synapse',
                'weight': weight,
                }
            nest.Connect(
                spike,
                target_pop,
                syn_spec=syn_dict_single_spike
            )

    def connect_stimuli(self):
        """
        Connect stimulus input with population.
        """
        if self.logger:
            self.logger.info("Stimulus input connected")
        for target_pop, stim in self.stimuli.items():
            nest.Connect(stim, self.pops[target_pop])

    def connect_devices(self):
        """ Connects the recording devices to the microcircuit."""
        # if nest.Rank() == 0:
        #     if ('spike_recorder' in self.sim_dict['rec_dev'] and
        #             'voltmeter' not in self.sim_dict['rec_dev']):
        #         print('Spike detector connected')
        #     elif ('spike_recorder' not in self.sim_dict['rec_dev'] and
        #             'voltmeter' in self.sim_dict['rec_dev']):
        #         print('Voltmeter connected')
        #     elif ('spike_recorder' in self.sim_dict['rec_dev'] and
        #             'voltmeter' in self.sim_dict['rec_dev']):
        #         print('Spike detector and voltmeter connected')
        #     else:
        #         print('no recording devices connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            if 'voltmeter' in self.sim_dict['rec_dev']:
                nest.Connect(self.voltmeter, target_pop)
            if 'spike_recorder' in self.sim_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_recorder)
            if "I_syn" in self.sim_dict["rec_dev"]:
                nest.Connect(self.I_syn_recorder, target_pop)

        if self.logger:
            connected = ', '.join(self.sim_dict['rec_dev'])
            self.logger.info('Devices connected to %d populations: %s',
                             len(self.pops), connected)     

    def setup(self, data_path, num_threads):
        """ Execute subfunctions of the network.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        Parameters
        ----------
        data_path : string
        num_threads : int
        """
        self.set_data_path(data_path)
        self.setup_nest(num_threads)
        
        # TODO: Topographic mapping is out of scope for now; create_populations_pos
        # needs revisiting before this branch can be relied upon.
        if self.net_dict['topographic_mapping']:
            self.create_populations_pos()
        else:
            self.create_populations()
        self.create_devices()
        self.create_poisson()
        self.create_single_spike()
        self.create_stimuli()
        self.connect_neurons()
        self.connect_poisson()
        self.connect_stimuli()
        self.connect_single_spike()
        self.connect_devices()

    def simulate(self):
        """ Simulates the model."""
        if self.logger:
            self.logger.info('Start simulating')
        nest.Simulate(self.sim_dict['t_sim'])
        if self.logger:
            self.logger.info('Simulation finished')

    def getHash(self):
        """
        Creates a hash from simulation parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        stim = {}
        for key, val in self.sim_dict["stimuli"].items():
            if isinstance(key, tuple):
                key = "_".join(str(k) for k in key)
                stim[key] = val
        if stim:
            self.sim_dict["stimuli"] = stim
        hash = dicthash.generate_hash_from_dict(self.sim_dict)
        
        stim = {}
        for key, val in self.sim_dict["stimuli"].items():
            if isinstance(key, str):
                key = tuple(key.split("_"))
                stim[key] = val
        if stim:
            self.sim_dict["stimuli"] = stim
        return hash

    def _getMemoryMB(self):
        """
        Return the currently occupied memory for the job in MB.

        Returns
        -------
        currMem : float
            Curently occupied memory in MB
        """
        try:
            currMem = nest.ll_api.sli_func('memory_thisjob')/1024.
        except AttributeError:
            currMem = nest.sli_func('memory_thisjob')/1024.
        return currMem

    def dump(self, base_folder):
        """
        Exports the full simulation specification. Creates a subdirectory of
        base_folder from the simulation hash where it puts all files.

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
        fn = os.path.join(out_folder, 'sim.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.sim_dict, outfile, default_flow_style=False)

def simulationDictFromDump(dump_folder):
    """
    Creates a simulation dict from the files created by Simulation.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    sim_dict : dict
        Full simulation dictionary
    """
    # Read sim.yaml
    fn = os.path.join(dump_folder, 'sim.yaml')
    with open(fn, 'r') as sim_file:
        sim_dict = yaml.load(sim_file, Loader=yaml.Loader)
    return sim_dict
