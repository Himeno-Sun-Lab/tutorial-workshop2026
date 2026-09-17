# -*- coding: utf-8 -*-
import nest
import numpy as np
import os
from datetime import datetime
import yaml
from dicthash import dicthash

from helpers.lognormal import mu_sigma_lognorm
from helpers.abeta import normalize_abeta_ratio, resolve_abeta_ratio

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

    def create_populations(self):
        """
        Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from a
        normal distribution.
        """
        # Create cortical populations.
        self.rank_logger.info('Memory on rank %d before creating populations: %.2fMB', nest.Rank(), self._getMemoryMB())
        self.pops = {}
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        local_num_threads = nest.GetKernelStatus('local_num_threads')
        with open(pop_file_name, 'w+') as pop_file:
            for pop, nn in self.net_dict['neuron_numbers'].items():
                if nn > 0:
                    if pop[-1] == 'E':
                        neuron_model_pop = self.net_dict['neuron_model_E']
                        neuron_params_pop = self.net_dict['neuron_params_E']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                    elif pop[-1] == 'I':
                        neuron_model_pop = self.net_dict['neuron_model_I']
                        neuron_params_pop = self.net_dict['neuron_params_I']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']
                    else:
                        msg = 'Populations have to be E or I'
                        self.rank_logger.error(msg)
                        raise NotImplementedError(msg)
                    population = nest.Create(neuron_model_pop, nn)
                    population.set(neuron_params_pop)

                    # Assign DC input
                    nest.SetStatus(
                        population, {
                            'I_e': self.net_dict['dc_drive'].loc[pop]
                        }
                    )

                    # population.set(neuron_params_pop)
                    population.V_m = nest.random.normal(
                        mean=self.sim_dict['V0_mean'],
                        std=self.sim_dict['V0_sd']
                    )
                    # Neuron parameters
                    for prm, dist_dict in nrn_prm_dist_pop.items():
                        if dist_dict['rel_sd'] > 0:
                            param_dist = dist_dict['distribution']
                            if param_dist == 'lognormal':
                                mean_prm = neuron_params_pop[prm]
                                offset_prm = 0.
                                if prm == 'V_th':
                                    mean_prm -= neuron_params_pop['E_L']
                                    offset_prm += neuron_params_pop['E_L']
                                assert mean_prm > 0
                                mu_param, sigma_param = mu_sigma_lognorm(
                                    mean=mean_prm,
                                    rel_sd=dist_dict['rel_sd']
                                )
                                population.set({prm:
                                    offset_prm + nest.random.lognormal(
                                    mean=mu_param,
                                    std=sigma_param
                                )})
                            else:
                                err_msg = "Parameter distribution "
                                err_msg += f"{param_dist} not implemented."
                                self.rank_logger.error(err_msg)
                                raise NotImplementedError(err_msg)

                    self.pops[pop] = population
                    pop_file.write('{};{};{}\n'.format(
                        pop, population[0].get()['global_id'], population[-1].get()['global_id']
                    ))
        self.rank_logger.info('Memory on rank %d after creating populations: %.2fMB', nest.Rank(), self._getMemoryMB())

    def _validate_abeta_ratios(self):
        """
        Normalizes and validates ``Abeta_ratio_E`` / ``Abeta_ratio_I``.

        Each may be a scalar fraction, a (possibly nested) dict keyed by area
        and optionally layer, or a path string to a CSV (see :mod:`helpers.abeta`).
        Any CSV path is loaded and the value is replaced in ``net_dict`` with the
        canonical scalar/dict form. Raises and logs on an unsupported type or an
        out-of-range fraction.
        """
        area_list = self.net_dict.get('area_list')
        layer_list = self.net_dict.get('layer_list')
        for param_name in ('Abeta_ratio_E', 'Abeta_ratio_I'):
            self.net_dict[param_name] = normalize_abeta_ratio(
                self.net_dict[param_name], param_name,
                area_list=area_list, layer_list=layer_list,
                logger=self.rank_logger,
            )

    def _abeta_ratio_for_pop(self, pop):
        """
        Resolves the effective amyloid fraction for a single population.

        Parameters
        ----------
        pop : tuple
            ``(area, layer, population)`` index, where ``population`` is
            ``'E'`` or ``'I'``.

        Returns
        -------
        float
            The fraction of amyloid-affected neurons for this population.
        """
        area, layer = pop[0], pop[1]
        if pop[-1] == 'E':
            ratio_cfg = self.net_dict['Abeta_ratio_E']
        elif pop[-1] == 'I':
            ratio_cfg = self.net_dict['Abeta_ratio_I']
        else:
            msg = 'Populations have to be E or I'
            self.rank_logger.error(msg)
            raise NotImplementedError(msg)
        return resolve_abeta_ratio(ratio_cfg, area, layer)

    def create_populations_Abeta(self):
        """
        Creates neuronal populations with amyloid-affected neurons
        """

        # ==================== 内存监控和初始化 ====================
        self.rank_logger.info('Memory on rank %d before creating populations: %.2fMB', nest.Rank(), self._getMemoryMB())

        # 校验Abeta_ratio配置（标量或按脑区/层的字典），不合法则抛出异常并记录日志
        self._validate_abeta_ratios()

        # 初始化群体字典和统计变量
        self.pops = {}
        processed_populations = 0
        total_abeta_neurons = 0
        
        # ==================== 文件设置 ====================
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        abeta_file_name = os.path.join(self.data_path, 'Abeta_GIDs.dat')
        
        # 计算总群体数量用于进度监控
        total_populations = len([pop for pop, nn in self.net_dict['neuron_numbers'].items() if nn > 0])
        
        # ==================== 优化的双文件写入主循环 ====================
        # 使用行缓冲模式，每行立即写入磁盘
        with open(pop_file_name, 'w+', buffering=1) as pop_file, \
            open(abeta_file_name, 'w+', buffering=1) as abeta_file:
            # if nest.Rank() == 0:
            #     # 写入Aβ文件头部信息
            #     abeta_file.write('# Abeta-affected neurons Global IDs for Alzheimer\'s Disease Simulation\n')
            #     abeta_file.write('# Generated by NEST simulator on SLURM cluster\n')
            #     abeta_file.write('# Format: Population_Name;Start_GID;End_GID;Neuron_Count\n')
            
            # 定义刷新间隔（适合集群环境）
            FLUSH_INTERVAL = max(1, total_populations // 20)  # 每5%进度刷新一次
            MEMORY_CHECK_INTERVAL = max(1, total_populations // 10)  # 每10%检查内存
            
            # 遍历所有神经元群体
            for pop, nn in self.net_dict['neuron_numbers'].items():
                if nn > 0:  # 只处理有神经元的群体
                    
                    # ==================== 神经元类型识别和参数加载 ====================
                    if pop[-1] == 'E':  # 兴奋性神经元
                        # 正常兴奋性神经元参数
                        neuron_model_pop = self.net_dict['neuron_model_E']
                        neuron_params_pop = self.net_dict['neuron_params_E']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                        
                        # Aβ影响的兴奋性神经元参数
                        neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_E"]
                        neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_E']
                        nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_E']

                    elif pop[-1] == 'I':  # 抑制性神经元
                        # 正常抑制性神经元参数
                        neuron_model_pop = self.net_dict['neuron_model_I']
                        neuron_params_pop = self.net_dict['neuron_params_I']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']

                        # Aβ影响的抑制性神经元参数
                        neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_I"]
                        neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_I']
                        nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_I']
                        
                    else:
                        msg = 'Populations have to be E or I'
                        self.rank_logger.error(msg)
                        raise NotImplementedError(msg)

                    # ==================== 神经元数量计算 ====================
                    # 按脑区/层解析当前群体的Aβ比例（支持标量或嵌套字典配置）
                    Abeta_ratio = self._abeta_ratio_for_pop(pop)
                    nn_Abeta = int(np.round(nn * Abeta_ratio))  # 受Aβ影响的神经元数量
                    nn_normal = int(nn - nn_Abeta)             # 正常神经元数量
                    total_abeta_neurons += nn_Abeta
                    
                    # ==================== 分别创建两类神经元群体 ====================
                    created_populations = []  # 存储创建的子群体

                    # 创建受Aβ影响的神经元（如果数量大于0）
                    if nn_Abeta > 0:
                        pop_Abeta = nest.Create(neuron_model_pop_Abeta, nn_Abeta)
                        pop_Abeta.set(neuron_params_pop_Abeta)
                        created_populations.append(pop_Abeta)

                        # 立即记录Aβ影响神经元的GID信息（行缓冲会立即写入）
                        abeta_file.write('{};{};{};{}\n'.format(
                            pop,                                    # 群体名称
                            pop_Abeta[0].get()['global_id'],       # 起始GID
                            pop_Abeta[-1].get()['global_id'],      # 结束GID
                            nn_Abeta                               # 神经元数量
                        ))

                    # 创建正常神经元（如果数量大于0）
                    if nn_normal > 0:
                        pop_normal = nest.Create(neuron_model_pop, nn_normal)
                        pop_normal.set(neuron_params_pop)
                        created_populations.append(pop_normal)

                    # 每个群体一行 DEBUG：GID 范围在创建后一并记录
                    self.rank_logger.debug(
                        '[%s] total=%d | Abeta=%d (GID %s-%s) | normal=%d (GID %s-%s) | ratio=%.2f',
                        pop, nn,
                        nn_Abeta,
                        pop_Abeta[0].get()['global_id']  if nn_Abeta  > 0 else '-',
                        pop_Abeta[-1].get()['global_id'] if nn_Abeta  > 0 else '-',
                        nn_normal,
                        pop_normal[0].get()['global_id']  if nn_normal > 0 else '-',
                        pop_normal[-1].get()['global_id'] if nn_normal > 0 else '-',
                        Abeta_ratio,
                    )
                    
                    # ==================== 合并群体 ====================
                    if created_populations:
                        if len(created_populations) == 2:
                            population = created_populations[0] + created_populations[1]  # Aβ + 正常
                        else:
                            population = created_populations[0]  # 只有一种类型
                    else:
                        continue  # 如果没有创建任何神经元，跳过此群体
                    
                    # ==================== 直流驱动设置 ====================
                    nest.SetStatus(
                        population, {
                            'I_e': self.net_dict['dc_drive'].loc[pop]
                        }
                    )
                    
                    # ==================== 膜电位初始化 ====================
                    population.V_m = nest.random.normal(
                        mean=self.sim_dict['V0_mean'],
                        std=self.sim_dict['V0_sd']
                    )
                    
                    # ==================== 参数分布设置 ====================
                    # 1. 为Aβ影响的神经元设置参数分布
                    if nn_Abeta > 0:
                        for prm, dist_dict in nrn_prm_dist_pop_Abeta.items():
                            if dist_dict['rel_sd'] > 0:
                                param_dist = dist_dict['distribution']
                                if param_dist == 'lognormal':
                                    mean_prm = neuron_params_pop_Abeta[prm]
                                    offset_prm = 0.
                                    if prm == 'V_th':
                                        mean_prm -= neuron_params_pop_Abeta['E_L']
                                        offset_prm += neuron_params_pop_Abeta['E_L']
                                    assert mean_prm > 0
                                    mu_param, sigma_param = mu_sigma_lognorm(
                                        mean=mean_prm,
                                        rel_sd=dist_dict['rel_sd']
                                    )
                                    pop_Abeta.set({prm:
                                        offset_prm + nest.random.lognormal(
                                        mean=mu_param,
                                        std=sigma_param
                                    )})
                                else:
                                    self.rank_logger.error(
                                        'Parameter distribution %s not implemented for Abeta neurons.', param_dist)
                                    raise NotImplementedError(
                                        'Parameter distribution %s not implemented for Abeta neurons.' % param_dist)

                    # 2. 为正常神经元设置参数分布
                    if nn_normal > 0:
                        for prm, dist_dict in nrn_prm_dist_pop.items():
                            if dist_dict['rel_sd'] > 0:
                                param_dist = dist_dict['distribution']
                                if param_dist == 'lognormal':
                                    mean_prm = neuron_params_pop[prm]
                                    offset_prm = 0.
                                    if prm == 'V_th':
                                        mean_prm -= neuron_params_pop['E_L']
                                        offset_prm += neuron_params_pop['E_L']
                                    assert mean_prm > 0
                                    mu_param, sigma_param = mu_sigma_lognorm(
                                        mean=mean_prm,
                                        rel_sd=dist_dict['rel_sd']
                                    )
                                    pop_normal.set({prm:
                                        offset_prm + nest.random.lognormal(
                                        mean=mu_param,
                                        std=sigma_param
                                    )})
                                else:
                                    self.rank_logger.error(
                                        'Parameter distribution %s not implemented for normal neurons.', param_dist)
                                    raise NotImplementedError(
                                        'Parameter distribution %s not implemented for normal neurons.' % param_dist)

                    # ==================== 群体存储和记录 ====================
                    self.pops[pop] = population
                    if nest.Rank() == 0:
                        # 立即记录整个群体的信息（行缓冲会立即写入）
                        pop_file.write('{};{};{}\n'.format(
                            pop,                                        # 群体名称
                            population[0].get()['global_id'],          # 群体起始GID
                            population[-1].get()['global_id']          # 群体结束GID
                        ))
                    
                    # ==================== 进度监控和缓冲区管理 ====================
                    processed_populations += 1
                    
                    # 定期强制刷新缓冲区（适合长时间运行的SLURM作业）
                    if processed_populations % FLUSH_INTERVAL == 0:
                        pop_file.flush()
                        abeta_file.flush()
                        progress_pct = 100 * processed_populations / total_populations
                        self.rank_logger.info(
                            'Network build progress: %d/%d (%.1f%%), Abeta neurons so far: %d',
                            processed_populations, total_populations, progress_pct, total_abeta_neurons)
                    
                    # 定期内存检查（防止内存溢出导致SLURM作业被杀）
                    # if processed_populations % MEMORY_CHECK_INTERVAL == 0:
                    #     current_memory = self._getMemoryMB()
                    #     print(f'[内存监控] Rank {nest.Rank()}: {current_memory:.2f}MB, '
                    #         f'平均每个群体: {current_memory/processed_populations:.2f}MB')
                        
                    #     # 如果内存使用过高，发出警告
                    #     if current_memory > 50000:  # 50GB警告阈值
                    #         print(f'警告: 内存使用较高 ({current_memory:.2f}MB), 请监控SLURM作业状态')
            
            # 最终刷新确保所有数据写入磁盘
            pop_file.flush()
            abeta_file.flush()
        
        # ==================== 最终统计和验证 ====================
        final_memory = self._getMemoryMB()
        total_neurons = sum(len(pop) for pop in self.pops.values())
        # if self.logger:
        #     self.logger.info(
        #         "AD network ready: %d pops | %d neurons total "
        #         "(Abeta %.1f%% = %d, normal %.1f%% = %d)",
        #         len(self.pops), total_neurons,
        #         100 * total_abeta_neurons / total_neurons, total_abeta_neurons,
        #         100 * (total_neurons - total_abeta_neurons) / total_neurons,
        #         total_neurons - total_abeta_neurons,
        #     )
            # self.logger.info('GID files: %s | %s', pop_file_name, abeta_file_name)
        self.rank_logger.debug('Memory on rank %d after creating populations: %.2fMB',
            nest.Rank(), final_memory)

        # 验证文件是否正确写入
        try:
            with open(pop_file_name, 'r') as f:
                pop_lines = len(f.readlines())
            with open(abeta_file_name, 'r') as f:
                abeta_lines = len([l for l in f.readlines() if not l.startswith('#')])
            if self.logger:
                self.logger.debug('File check: population_GIDs.dat has %d rows, Abeta_GIDs.dat has %d records',
                    pop_lines, abeta_lines)
        except Exception as e:
            if self.logger:
                self.logger.error('File check failed: %s', e)

    def create_populations_pos(self):
        """
        Creates spatially-positioned populations for topographic mapping.

        TODO: Topographic mapping is out of scope for now. This method is not
        exercised by the default pipeline (``topographic_mapping`` defaults to
        False) and needs revisiting before it can be relied upon.
        """
        self.rank_logger.info('Memory on rank %d before creating populations: %.2fMB',
            nest.Rank(), self._getMemoryMB())

        # 校验Abeta_ratio配置（标量或按脑区/层的字典），不合法则抛出异常并记录日志
        self._validate_abeta_ratios()

        self.pops = {}
        processed_populations = 0
        total_abeta_neurons = 0
        
        random_seed_generator = np.random.default_rng(seed=self.sim_dict["master_seed"])

        # ==================== 文件设置 ====================
        pop_gid_fn = os.path.join(self.data_path, 'population_GIDs.dat')
        abeta_gid_fn = os.path.join(self.data_path, 'Abeta_GIDs.dat')
        total_populations = len([pop for pop, nn in self.net_dict['neuron_numbers'].items() if nn > 0])
        
        # ==================== 优化的双文件写入主循环 ====================
        # 使用行缓冲模式，每行立即写入磁盘
        with open(pop_gid_fn, 'w+', buffering=1) as pop_file, \
            open(abeta_gid_fn, 'w+', buffering=1) as abeta_file:
            
            # 定义刷新间隔（适合集群环境）
            FLUSH_INTERVAL = max(1, total_populations // 20)  # 每5%进度刷新一次
            MEMORY_CHECK_INTERVAL = max(1, total_populations // 10)  # 每10%检查内存
            
            # 遍历所有神经元群体
            for pop, nn in self.net_dict['neuron_numbers'].items():
                pos_fn = os.path.join(self.data_path, pop[0] + ".dat") 
                with open(pos_fn, "w+") as pos_f:
                    if nn > 0:  # 只处理有神经元的群体
                        if pop[-1] == 'E':  # 兴奋性神经元
                            # 正常兴奋性神经元参数
                            neuron_model_pop = self.net_dict['neuron_model_E']
                            neuron_params_pop = self.net_dict['neuron_params_E']
                            nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                            
                            # Aβ影响的兴奋性神经元参数
                            neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_E"]
                            neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_E']
                            nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_E']

                        elif pop[-1] == 'I':  # 抑制性神经元
                            # 正常抑制性神经元参数
                            neuron_model_pop = self.net_dict['neuron_model_I']
                            neuron_params_pop = self.net_dict['neuron_params_I']
                            nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']

                            # Aβ影响的抑制性神经元参数
                            neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_I"]
                            neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_I']
                            nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_I']
                        
                        else:
                            msg = 'Populations have to be E or I'
                            self.rank_logger.error(msg)
                            raise NotImplementedError(msg)

                        # ==================== 神经元数量计算 ====================
                        shape = self.generate_shape(pop, nn)
                        center = [0., 0., self.net_dict["thickness"].loc[pop[:-1]] / 2]
                        extent = [1., 1., 2.]
                        
                        pop_grid = nest.Create(neuron_model_pop, positions=nest.spatial.grid(shape=shape, center=center, extent=extent))
                        head = pop_grid[0].get()["global_id"]
                        tail = pop_grid[-1].get()["global_id"]
                        nn_grid = shape[0] * shape[1] * shape[2]

                        # 按脑区/层解析当前群体的Aβ比例（支持标量或嵌套字典配置）
                        Abeta_ratio = self._abeta_ratio_for_pop(pop)
                        nn_Abeta_grid = int(np.round(nn_grid * Abeta_ratio))
                        random_id_generator = np.random.default_rng(seed=int(random_seed_generator.uniform(0, 1) * 1000))
                        Abeta_GIDs = head + random_id_generator.choice(nn_grid, size=nn_Abeta_grid, replace=False)
                        Abeta_GIDs.sort()
                        self.rank_logger.debug('Abeta_GIDs: %s', Abeta_GIDs)
                        pop_Abeta = nest.NodeCollection(Abeta_GIDs.tolist())
                        pop_Abeta.set(neuron_model_pop_Abeta)
                        if nn_Abeta_grid > 0:
                            abeta_file.write('{};{};{};{}\n'.format(
                                pop,                                                                    # 群体名称
                                Abeta_GIDs                                                              # 神经元GID列表
                            ))
                        
                        # ==================== 直流驱动设置 ====================
                        nest.SetStatus(
                            pop_grid, {
                                'I_e': self.net_dict['dc_drive'].loc[pop]
                            }
                        )
                        
                        # ==================== 膜电位初始化 ====================
                        pop_grid.V_m = nest.random.normal(
                            mean=self.sim_dict['V0_mean'],
                            std=self.sim_dict['V0_sd']
                        )
                        
                        # ==================== 参数分布设置 ====================
                        if nn_grid > 0:
                            for prm, dist_dict in nrn_prm_dist_pop.items():
                                if dist_dict['rel_sd'] > 0:
                                    param_dist = dist_dict['distribution']
                                    if param_dist == 'lognormal':
                                        mean_prm = neuron_params_pop[prm]
                                        offset_prm = 0.
                                        if prm == 'V_th':
                                            mean_prm -= neuron_params_pop['E_L']
                                            offset_prm += neuron_params_pop['E_L']
                                        assert mean_prm > 0
                                        mu_param, sigma_param = mu_sigma_lognorm(
                                            mean=mean_prm,
                                            rel_sd=dist_dict['rel_sd']
                                        )
                                        pop_grid.set({prm:
                                            offset_prm + nest.random.lognormal(
                                            mean=mu_param,
                                            std=sigma_param
                                        )})
                                    else:
                                        err_msg = f"Parameter distribution {param_dist} not implemented for normal neurons."
                                        self.rank_logger.error(err_msg)
                                        raise NotImplementedError(err_msg)

                        if nn_Abeta_grid > 0:
                            for prm, dist_dict in nrn_prm_dist_pop_Abeta.items():
                                if dist_dict['rel_sd'] > 0:
                                    param_dist = dist_dict['distribution']
                                    if param_dist == 'lognormal':
                                        mean_prm = neuron_params_pop_Abeta[prm]
                                        offset_prm = 0.
                                        if prm == 'V_th':
                                            mean_prm -= neuron_params_pop_Abeta['E_L']
                                            offset_prm += neuron_params_pop_Abeta['E_L']
                                        assert mean_prm > 0
                                        mu_param, sigma_param = mu_sigma_lognorm(
                                            mean=mean_prm,
                                            rel_sd=dist_dict['rel_sd']
                                        )
                                        pop_Abeta.set({prm:
                                            offset_prm + nest.random.lognormal(
                                            mean=mu_param,
                                            std=sigma_param
                                        )})
                                    else:
                                        err_msg = f"Parameter distribution {param_dist} not implemented for Abeta neurons."
                                        self.rank_logger.error(err_msg)
                                        raise NotImplementedError(err_msg)

                        # ==================== 群体存储和记录 ====================
                        self.pops[pop] = pop_grid
                        if nest.Rank() == 0:
                            # 立即记录整个群体的信息（行缓冲会立即写入）
                            pop_file.write('{};{};{}\n'.format(
                                pop,
                                head,
                                tail,
                                nn_grid
                            ))
                        
                        # ==================== 进度监控和缓冲区管理 ====================
                        processed_populations += 1
                        
                        # 定期强制刷新缓冲区（适合长时间运行的SLURM作业）
                        if processed_populations % FLUSH_INTERVAL == 0:
                            pop_file.flush()
                            abeta_file.flush()
                            progress_pct = 100 * processed_populations / total_populations
                            print(f'[SLURM进度监控] 阿尔茨海默病网络构建进度: {processed_populations}/{total_populations} '
                                f'({progress_pct:.1f}%), 累积Aβ神经元: {total_abeta_neurons}')
            
            # 最终刷新确保所有数据写入磁盘
            pop_file.flush()
            abeta_file.flush()
        
        # ==================== 最终统计和验证 ====================
        final_memory = self._getMemoryMB()
        total_neurons = sum(len(pop) for pop in self.pops.values())
        if self.logger:
            self.logger.info("Alzheimer's Disease Brain Network Created")
            self.logger.info('Created %d populations', len(self.pops))
            self.logger.info('Neurons in total: %d', total_neurons)
            self.logger.info('Amyloid-affected neurons: %d (%.2f%%)',
                total_abeta_neurons, 100 * total_abeta_neurons / total_neurons)
            self.logger.info('Normal neurons: %d (%.2f%%)',
                total_neurons - total_abeta_neurons, 100 * (total_neurons - total_abeta_neurons) / total_neurons)
            self.logger.info('Memory at Rank %d: %.2fMB', nest.Rank(), final_memory)
            self.logger.info('Population GIDs: %s', pop_gid_fn)
            self.logger.info('Amyloid GIDs: %s', abeta_gid_fn)

        # 验证文件是否正确写入
        try:
            with open(pop_gid_fn, 'r') as f:
                pop_lines = len(f.readlines())
            with open(abeta_gid_fn, 'r') as f:
                abeta_lines = len([l for l in f.readlines() if not l.startswith('#')])
            if self.logger:
                self.logger.info('File check: population_GIDs.dat has %d rows, Abeta_GIDs.dat has %d records',
                    pop_lines, abeta_lines)
        except Exception as e:
            if self.logger:
                self.logger.warning('File check failed: %s', e)

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
        elif self.net_dict["Abeta"]:
            self.create_populations_Abeta()
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
