"""
Some helper functions for snakemake.
"""
import subprocess
import logging
import colorlog
import sys
import os
import nest

CLIFMT = colorlog.ColoredFormatter(
    '%(asctime)s.%(msecs)03d %(log_color)s[%(levelname)s]%(reset)s %(message)s',
    datefmt='%Z-%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red'
    }
)
LOGFMT = logging.Formatter(
    '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Z-%Y-%m-%d %H:%M:%S'
)


def get_git_revision_hash():
    git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
            ).decode('ascii').strip()
    return git_hash


def nested_dict_update(default_dict, specific_dict, logger=None):
    """
    Updates the values of default_dict by the values in specific_dict
    recursively.
    """
    for key in specific_dict:
        # Fully replace neuron_params_X or neuron_param_dist_X dictionaries
        if isinstance(key, str) and key.startswith('neuron_param'):
            if logger:
                logger.debug('Specified %s dict to %s', key, specific_dict[key])
            default_dict[key] = specific_dict[key]

        elif isinstance(specific_dict[key], dict):
            if key not in default_dict:
                # 直接添加，不递归
                default_dict[key] = specific_dict[key]
            elif isinstance(default_dict[key], dict):
                # 递归更新
                nested_dict_update(default_dict[key], specific_dict[key], logger)
        # elif isinstance(specific_dict[key], dict):
        #     nested_dict_update(default_dict[key], specific_dict[key])
        else:
            if key in default_dict.keys():
                if isinstance(specific_dict[key], type(default_dict[key])) or default_dict[key] is None:
                    default_dict[key] = specific_dict[key]
                else:
                    raise TypeError(
                        'Type of specific_dict[{0}]={1} and '
                        'default_dict[{0}]={2} dont match.'.format(
                            key,
                            specific_dict[key],
                            default_dict[key]
                        )
                    )
            else:
                if logger:
                    logger.debug('Adding entry default_dict[%s]=%s', key, specific_dict[key])
                default_dict[key] = specific_dict[key]


def build_loggers(log_path, mode='DEBUG'):
    """
    Initialize main logger (rank 0 only) and rank logger (all ranks).

    Parameters
    ----------
    log_path : str
        Path to the main log file (e.g., 'out/log/experiment.log')
    mode : str
        Logging level name, one of 'DEBUG', 'INFO', 'WARNING', 'ERROR',
        'CRITICAL'. Controls the minimum severity of messages written to
        both the stream and file handlers. Defaults to 'DEBUG'.

    Returns
    -------
    tuple
        (main_logger, rank_logger) where main_logger is None for non-rank-0
    """
    level = getattr(logging, str(mode).upper(), logging.DEBUG)

    # Initialize main logger (rank 0 only)
    main_logger = None
    if nest.Rank() == 0:
        main_logger = logging.getLogger("main")
        if not main_logger.handlers:
            main_logger.setLevel(level)
            if sys.stderr.isatty():
                cli = colorlog.StreamHandler()
                cli.setFormatter(CLIFMT)
            else:
                cli = logging.StreamHandler()
                cli.setFormatter(LOGFMT)
            cli.setLevel(level)
            log = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            log.setFormatter(LOGFMT)
            log.setLevel(level)
            main_logger.addHandler(cli)
            main_logger.addHandler(log)
            main_logger.info("Main logger initialized (level=%s).", mode.upper())

    # Initialize rank logger (all ranks)
    rank_path = os.path.splitext(log_path)[0]
    os.makedirs(rank_path, exist_ok=True)
    rank_logger = logging.getLogger(str(nest.Rank()))
    if not rank_logger.handlers:
        rank_logger.setLevel(level)
        rank_log = logging.FileHandler(
            os.path.join(rank_path, f"rank{nest.Rank()}.log"),
            mode="w",
            encoding="utf-8"
        )
        rank_log.setFormatter(LOGFMT)
        rank_log.setLevel(level)
        rank_logger.addHandler(rank_log)
        rank_logger.info("Logger at rank %d initialized (level=%s).", nest.Rank(), mode.upper())

    return main_logger, rank_logger
