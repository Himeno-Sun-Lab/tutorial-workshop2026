import pandas as pd


def ratio_exc_to_inh(path):
    """
    Provides the ratio of excitatory to inhibitory cells.

    Returns
    -------
    ratio : Series
        Layer-resolved fraction of excitatory cells.
    """
    return pd.read_csv(path, index_col=0)['ratio']
