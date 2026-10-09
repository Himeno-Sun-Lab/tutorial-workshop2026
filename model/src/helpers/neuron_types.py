# -*- coding: utf-8 -*-
"""
Helpers for the ``neuron_types`` network parameter.

``neuron_types`` declares additional, custom-named neuron types that replace a
fraction of the neurons of every population. Each population is split into the
declared types (in declaration order) plus the remaining ``base`` neurons,
which use ``neuron_model_X`` / ``neuron_params_X`` / ``neuron_param_dist_X``::

    params['neuron_types'] = {
        'Abeta': {
            'ratio_E': 'path/to/abeta_ratio.csv',
            'ratio_I': 0.0,
            'neuron_params_E': {'E_L': -65.0, 'V_th': -50.0},
        },
        'Tau': {
            'ratio_E': {'entorhinal': 0.3, 'default': 0.0},
        },
    }

Recognised keys per type (all optional, ``X`` is ``E`` or ``I``):

- ``ratio_X``: fraction of the ``X`` neurons of a population that are of this
  type (see *Ratio forms* below). Defaults to ``0``.
- ``neuron_model_X``: NEST model. Defaults to the base ``neuron_model_X``.
- ``neuron_params_X``: model parameters. When the model equals the base model
  the given entries override the base ``neuron_params_X``, so only the changed
  parameters need to be listed; otherwise the dict is used as-is.
- ``neuron_param_dist_X``: parameter distribution, same rule as above.

Parameters of the side (``E``/``I``) whose ratio is zero everywhere are not
needed. The type name ``base`` is reserved for the remaining neurons. For every
``(area, layer, X)`` the ratios of all types must sum to at most 1.

Ratio forms
-----------
Each ``ratio_X`` may be specified in one of four forms:

1. A scalar (``int``/``float``) in ``[0, 1]`` - the same fraction is applied
   uniformly to every area and every layer.
2. A flat ``dict`` keyed by area name, ``{area: fraction}`` - each listed area
   gets its own uniform fraction across all of its layers.
3. A nested ``dict``, ``{area: {layer: fraction}}`` - a per-layer fraction can
   be given for individual layers of an area. Scalar and nested area entries
   may be mixed within the same dict.
4. A ``str`` path to a CSV file describing the distribution (see below). The
   path is loaded and converted into one of the dict forms above; this lets the
   distribution be controlled hierarchically from an external file.

In the dict forms an optional ``"default"`` key provides the fallback fraction
for unlisted areas (top level) or unlisted layers (nested). When no
``"default"`` is given, unlisted areas/layers receive a fraction of ``0.0``.

CSV format
----------
The CSV is effectively a ``(area, layer)`` MultiIndex mapping to a fraction.
It must contain an ``area`` column and a value column (named ``ratio``,
``fraction`` or ``value``); a ``layer`` column is optional:

    area,layer,ratio
    default,,0.02
    precuneus,II/III,0.22
    precuneus,IV,0.12
    precuneus,V,0.20
    precuneus,VI,0.15
    posteriorcingulate,,0.18

Semantics:

- An empty ``layer`` cell means the fraction applies to the whole area.
- The special area name ``default`` (with an empty layer) sets the global
  fallback for unlisted areas.
- Rows that give a specific ``layer`` set that layer's fraction for the area;
  an area may mix a layer-less (whole-area default) row with per-layer rows.

After loading, areas that carry only a whole-area value collapse to a scalar,
yielding exactly the same structure as the hand-written dict forms above.
"""

import os
from copy import deepcopy

import pandas as pd


# Name of the neurons that are not of any custom type.
BASE_TYPE = 'base'

# Neuron classes a type can define parameters for.
POPULATIONS = ('E', 'I')

# Keys a neuron type may define, per population X.
_TYPE_KEYS = ('ratio', 'neuron_model', 'neuron_params', 'neuron_param_dist')

# Internal sentinel used as the per-area fallback layer key.
_DEFAULT_KEY = 'default'

# Accepted (lower-cased) names for the value column in a CSV.
_VALUE_COLUMNS = ('ratio', 'fraction', 'value')


def _is_number(value):
    """True for real numbers, explicitly excluding ``bool``."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _raise(msg, logger=None, exc=ValueError):
    if logger is not None:
        logger.error(msg)
    raise exc(msg)


def load_ratio_csv(path, param_name='ratio', logger=None):
    """
    Load a (possibly MultiIndexed) ratio CSV into the nested-dict form.

    Parameters
    ----------
    path : str
        Path to the CSV file (see module docstring for the expected format).
    param_name : str
        Name of the parameter, used for error messages.
    logger : logging.Logger, optional
        Logger used to report the error before the exception is raised.

    Returns
    -------
    dict
        A ``{area: fraction}`` / ``{area: {layer: fraction}}`` mapping, with an
        optional top-level ``"default"`` key, ready for :func:`resolve_ratio`.

    Raises
    ------
    TypeError
        If ``path`` is not a string.
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the CSV is malformed (missing columns, empty area, non-numeric or
        missing fraction).
    """
    def _fail(msg, exc=ValueError):
        _raise('{}: {}'.format(param_name, msg), logger, exc)

    if not isinstance(path, str):
        _fail('expected a path string, got {}'.format(type(path).__name__), TypeError)
    if not os.path.isfile(path):
        _fail('CSV file not found: {}'.format(path), FileNotFoundError)

    try:
        # Read everything as strings so empty layer cells stay empty (and are
        # not coerced to NaN floats), then convert fractions explicitly.
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:  # pragma: no cover - pandas raises many subtypes
        _fail('failed to read CSV {!r}: {}'.format(path, e))

    colmap = {str(c).strip().lower(): c for c in df.columns}
    if 'area' not in colmap:
        _fail("CSV must contain an 'area' column; found {}".format(list(df.columns)))
    value_col = next((colmap[c] for c in _VALUE_COLUMNS if c in colmap), None)
    if value_col is None:
        _fail("CSV must contain a value column named one of {}; found {}".format(
            list(_VALUE_COLUMNS), list(df.columns)))
    area_col = colmap['area']
    layer_col = colmap.get('layer')

    nested = {}  # area -> {layer_or_default: fraction}
    for i, row in df.iterrows():
        # +2 maps the 0-based row index to a 1-based spreadsheet line (header + 1).
        line = i + 2
        area = str(row[area_col]).strip()
        if area == '':
            _fail('empty area in CSV row {}'.format(line))
        layer = str(row[layer_col]).strip() if layer_col is not None else ''
        raw = str(row[value_col]).strip()
        where = "area '{}'{}".format(area, "/'" + layer + "'" if layer else '')
        if raw == '':
            _fail('missing ratio for {} (CSV row {})'.format(where, line))
        try:
            frac = float(raw)
        except ValueError:
            _fail('ratio {!r} for {} is not a number (CSV row {})'.format(
                raw, where, line))
        key = layer if layer else _DEFAULT_KEY
        nested.setdefault(area, {})[key] = frac

    # Collapse areas that only carry a whole-area value into a plain scalar,
    # matching the hand-written dict form.
    result = {}
    for area, layer_map in nested.items():
        if set(layer_map) == {_DEFAULT_KEY}:
            result[area] = layer_map[_DEFAULT_KEY]
        else:
            result[area] = layer_map
    return result


def normalize_ratio(ratio, param_name, area_list=None,
                    layer_list=None, logger=None):
    """
    Resolve any accepted ratio form to a scalar/dict, then validate.

    A string is interpreted as a path to a CSV file and loaded via
    :func:`load_ratio_csv`; scalars and dicts are passed through. The result is
    validated with :func:`validate_ratio` before being returned, so callers can
    store the canonical form once (at network-build time) and downstream code
    never needs to touch the filesystem again.

    Parameters
    ----------
    ratio : int, float, dict or str
        The configured ratio (see module docstring for accepted forms).
    param_name : str
        Name of the parameter, used for error messages.
    area_list, layer_list : iterable of str, optional
        Valid area/layer names, used for key validation.
    logger : logging.Logger, optional
        Logger used to report errors before exceptions are raised.

    Returns
    -------
    int, float or dict
        The normalized, validated ratio.
    """
    if isinstance(ratio, str):
        ratio = load_ratio_csv(ratio, param_name=param_name, logger=logger)
    validate_ratio(ratio, param_name, area_list=area_list,
                   layer_list=layer_list, logger=logger)
    return ratio


def validate_ratio(ratio, param_name, area_list=None,
                   layer_list=None, logger=None):
    """
    Validate the structure and numeric range of a ratio parameter.

    Expects an already-normalized value (scalar or dict); use
    :func:`normalize_ratio` first to handle CSV path strings.

    Parameters
    ----------
    ratio : int, float or dict
        The configured ratio (see module docstring for accepted forms).
    param_name : str
        Name of the parameter, used for error messages.
    area_list : iterable of str, optional
        Valid area names. When given, dict keys are checked against it.
    layer_list : iterable of str, optional
        Valid layer names. When given, nested dict keys are checked against it.
    logger : logging.Logger, optional
        Logger used to report the error before the exception is raised.

    Raises
    ------
    TypeError
        If ``ratio`` (or any nested value/key) has an unsupported type.
    ValueError
        If any fraction lies outside ``[0, 1]`` or a key names an unknown
        area/layer.
    """
    areas = set(area_list) if area_list is not None else None
    layers = set(layer_list) if layer_list is not None else None

    def _fail(msg, exc=ValueError):
        _raise('{}: {}'.format(param_name, msg), logger, exc)

    def _check_fraction(value, context):
        if not _is_number(value):
            _fail('{} must be a number, got {}'.format(
                context, type(value).__name__), TypeError)
        if not (0.0 <= float(value) <= 1.0):
            _fail('{} must be within [0, 1], got {}'.format(context, value))

    # Form 1: scalar.
    if _is_number(ratio):
        _check_fraction(ratio, 'value')
        return

    # Forms 2 & 3: dict.
    if not isinstance(ratio, dict):
        _fail('must be a number, dict or CSV path string, got {}'.format(
            type(ratio).__name__), TypeError)

    for area_key, area_val in ratio.items():
        if not isinstance(area_key, str):
            _fail('area key {!r} must be a string'.format(area_key), TypeError)
        if area_key != _DEFAULT_KEY and areas is not None and area_key not in areas:
            _fail('unknown area {!r}; not in area_list'.format(area_key))

        if isinstance(area_val, dict):
            # Nested per-layer mapping.
            for layer_key, layer_val in area_val.items():
                if not isinstance(layer_key, str):
                    _fail('layer key {!r} under area {!r} must be a string'.format(
                        layer_key, area_key), TypeError)
                if (layer_key != _DEFAULT_KEY and layers is not None
                        and layer_key not in layers):
                    _fail('unknown layer {!r} under area {!r}'.format(
                        layer_key, area_key))
                _check_fraction(layer_val, "ratio for '{}/{}'".format(
                    area_key, layer_key))
        else:
            _check_fraction(area_val, "ratio for area '{}'".format(area_key))


def ratio_is_zero(ratio):
    """
    Return True if a (normalized) ratio is zero everywhere.

    Handles scalars and (nested) dicts; the optional ``"default"`` entry is just
    another value and is checked like any other. A path string is treated as
    *not* zero (it has not been loaded yet), since its contents are unknown.

    Parameters
    ----------
    ratio : int, float, dict or str
        The ratio to inspect (see module docstring for accepted forms).

    Returns
    -------
    bool
    """
    if _is_number(ratio):
        return float(ratio) == 0.0
    if isinstance(ratio, dict):
        for value in ratio.values():
            if isinstance(value, dict):
                if any(_is_number(v) and float(v) != 0.0 for v in value.values()):
                    return False
            elif _is_number(value) and float(value) != 0.0:
                return False
        return True
    # Unknown form (e.g. an unresolved path string): assume it may be non-zero.
    return False


def resolve_ratio(ratio, area, layer):
    """
    Resolve the effective fraction for a single ``(area, layer)``.

    Assumes ``ratio`` has already passed :func:`normalize_ratio`
    (i.e. CSV paths have been loaded and the value validated).

    Parameters
    ----------
    ratio : int, float or dict
        The normalized ratio (see module docstring for accepted forms).
    area : str
        Area name of the population.
    layer : str
        Layer name of the population.

    Returns
    -------
    float
        The fraction in ``[0, 1]``. Areas/layers not covered by a dict resolve
        to ``0.0`` unless a ``"default"`` entry is present.
    """
    if isinstance(ratio, str):
        raise TypeError(
            'Ratio is still a path string ({!r}); call normalize_ratio to '
            'load it before resolving.'.format(ratio))

    if _is_number(ratio):
        return float(ratio)

    if area in ratio:
        area_val = ratio[area]
    elif _DEFAULT_KEY in ratio:
        area_val = ratio[_DEFAULT_KEY]
    else:
        return 0.0

    if _is_number(area_val):
        return float(area_val)

    # Nested per-layer mapping.
    if layer in area_val:
        return float(area_val[layer])
    if _DEFAULT_KEY in area_val:
        return float(area_val[_DEFAULT_KEY])
    return 0.0


def normalize_neuron_types(neuron_types, base, area_list, layer_list,
                           logger=None):
    """
    Resolve the ``neuron_types`` parameter into its canonical form.

    Every type is completed to define all keys for both ``E`` and ``I``: ratios
    are loaded/validated with :func:`normalize_ratio`, missing ratios become
    ``0``, and missing or partial models/parameters are completed from the base
    neuron (see module docstring). Finally the ratios of all types are checked
    to sum to at most 1 for every ``(area, layer, X)``.

    Parameters
    ----------
    neuron_types : dict or None
        ``{name: type_dict}`` as configured.
    base : dict
        Base neuron definition holding ``neuron_model_X``, ``neuron_params_X``
        and ``neuron_param_dist_X`` for ``X`` in ``E``/``I``.
    area_list, layer_list : iterable of str
        Area and layer names of the network.
    logger : logging.Logger, optional
        Logger used to report errors before exceptions are raised.

    Returns
    -------
    dict
        ``{name: type_dict}`` in declaration order, every ``type_dict`` holding
        ``ratio_X``, ``neuron_model_X``, ``neuron_params_X`` and
        ``neuron_param_dist_X`` for both ``X``.
    """
    if neuron_types is None:
        return {}
    if not isinstance(neuron_types, dict):
        _raise('neuron_types must be a dict, got {}'.format(
            type(neuron_types).__name__), logger, TypeError)

    allowed = {'{}_{}'.format(k, X) for k in _TYPE_KEYS for X in POPULATIONS}
    resolved = {}
    for name, spec in neuron_types.items():
        if not isinstance(name, str) or not name:
            _raise('neuron_types: type name {!r} must be a non-empty '
                   'string'.format(name), logger, TypeError)
        if name == BASE_TYPE:
            _raise("neuron_types: the name '{}' is reserved for the remaining "
                   "neurons".format(BASE_TYPE), logger)
        if not isinstance(spec, dict):
            _raise("neuron_types['{}'] must be a dict, got {}".format(
                name, type(spec).__name__), logger, TypeError)
        unknown = sorted(set(spec) - allowed)
        if unknown:
            _raise("neuron_types['{}']: unknown keys {}; allowed are {}".format(
                name, unknown, sorted(allowed)), logger)

        out = {}
        for X in POPULATIONS:
            ratio = normalize_ratio(
                spec.get('ratio_' + X, 0.0),
                "neuron_types['{}']['ratio_{}']".format(name, X),
                area_list=area_list, layer_list=layer_list, logger=logger)
            model = spec.get('neuron_model_' + X, base['neuron_model_' + X])
            same_model = model == base['neuron_model_' + X]
            out['ratio_' + X] = ratio
            out['neuron_model_' + X] = model
            for key in ('neuron_params_' + X, 'neuron_param_dist_' + X):
                given = spec.get(key)
                if same_model:
                    merged = deepcopy(base[key])
                    merged.update(deepcopy(given or {}))
                    out[key] = merged
                elif given is not None:
                    out[key] = deepcopy(given)
                elif key.startswith('neuron_param_dist') or ratio_is_zero(ratio):
                    out[key] = {}
                else:
                    _raise("neuron_types['{}']: '{}' is required because "
                           "'neuron_model_{}' ({}) differs from the base "
                           "model".format(name, key, X, model), logger)
        resolved[name] = out

    # The types share each population, so their fractions must fit into it.
    for X in POPULATIONS:
        for area in area_list:
            for layer in layer_list:
                total = sum(resolve_ratio(t['ratio_' + X], area, layer)
                            for t in resolved.values())
                if total > 1.0 + 1e-9:
                    _raise('neuron_types: ratios of {} sum to {:.4f} > 1 in '
                           '{}/{}/{}'.format(list(resolved), total, area,
                                             layer, X), logger)
    return resolved


def type_fractions(neuron_types, area, layer, pop):
    """
    Fractions of a population ``(area, layer, pop)`` taken by each type.

    Parameters
    ----------
    neuron_types : dict
        Normalized ``neuron_types`` (see :func:`normalize_neuron_types`).
    area, layer, pop : str
        Population index; ``pop`` is ``'E'`` or ``'I'``.

    Returns
    -------
    list of (str, float)
        ``(name, fraction)`` for every custom type, in declaration order. The
        ``base`` neurons take the remaining ``1 - sum(fractions)``.
    """
    return [(name, resolve_ratio(t['ratio_' + pop], area, layer))
            for name, t in neuron_types.items()]


def active_neuron_types(neuron_types):
    """
    Names of the types that have a non-zero ratio somewhere.

    Parameters
    ----------
    neuron_types : dict or None
        Normalized ``neuron_types``.

    Returns
    -------
    list of str
    """
    return [name for name, t in (neuron_types or {}).items()
            if not all(ratio_is_zero(t['ratio_' + X]) for X in POPULATIONS)]
