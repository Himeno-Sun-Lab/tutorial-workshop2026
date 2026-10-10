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


def _is_number(value):
    """True for int and float, excluding bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def nested_dict_update(default_dict, specific_dict, logger=None):
    """
    Updates the values of default_dict by the values in specific_dict
    recursively.

    ``neuron_params_X`` / ``neuron_param_dist_X`` dicts replace the default as
    a whole. Every other dict is merged into the default dict of the same key.
    A value must have the type of the default it replaces; ``int`` and
    ``float`` are interchangeable as long as no information is lost (the value
    takes the default's type). A key the defaults do not have, or whose
    default is None, is taken as it is.

    Raises
    ------
    TypeError
        If a value's type does not match its default, including a dict given
        for a default that is not a dict.
    """
    def _mismatch(key, value, default):
        return TypeError(
            'Type of specific_dict[{0}]={1!r} and default_dict[{0}]={2!r} '
            "don't match.".format(key, value, default))

    for key in specific_dict:
        value = specific_dict[key]
        # Fully replace neuron_params_X or neuron_param_dist_X dictionaries
        if isinstance(key, str) and key.startswith('neuron_param'):
            if logger:
                logger.debug('Specified %s dict to %s', key, value)
            default_dict[key] = value
            continue

        if key not in default_dict or default_dict[key] is None:
            if logger:
                logger.debug('Adding entry default_dict[%s]=%s', key, value)
            default_dict[key] = value
            continue

        default = default_dict[key]
        if isinstance(value, dict):
            if not isinstance(default, dict):
                raise _mismatch(key, value, default)
            nested_dict_update(default, value, logger)
        elif isinstance(value, type(default)):
            default_dict[key] = value
        elif _is_number(value) and _is_number(default):
            if isinstance(default, float):
                default_dict[key] = float(value)
            elif float(value).is_integer():
                default_dict[key] = int(value)
            else:
                raise _mismatch(key, value, default)
        else:
            raise _mismatch(key, value, default)


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
