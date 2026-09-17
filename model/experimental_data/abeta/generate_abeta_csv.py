# -*- coding: utf-8 -*-
"""
Generate Abeta_ratio CSV files from ADNI amyloid-PET (AV45 SUVR) data.

For each amyloid-positive ADNI subject this reads the per-region SUVR table
(``amyloid_output/suvr_by_region.csv``), maps the FreeSurfer Desikan-Killiany
regions onto the model's area names, averages the two hemispheres (the model is
single-hemisphere), and converts each region's amyloid burden into a fraction
of amyloid-affected neurons.

SUVR -> affected-neuron fraction
--------------------------------
AV45 SUVR ~ 1.0 reflects no specific amyloid binding; a SUVR of ~1.11
(cerebellar-grey reference) is the conventional amyloid-positivity threshold.
We map SUVR linearly above that threshold and clip into ``[0, RATIO_MAX]``:

    ratio = clip((mean_suvr - SUVR_THRESHOLD) * SLOPE, 0, RATIO_MAX)

so sub-threshold regions get 0.0 (covered by the CSV ``default`` row) and the
most heavily affected cortex (SUVR ~ 2.0) saturates at ``RATIO_MAX``. The
constants are deliberately conservative so that even severe burden replaces only
a minority of neurons with the Abeta neuron model.

The output CSV uses the (area, layer, ratio) format read by
``src/helpers/abeta.py`` (an empty layer means the value applies to the whole
area; amyloid PET has no laminar resolution, so all rows are area-level).
"""

import os

import pandas as pd

# --- SUVR -> fraction mapping constants (see module docstring) ---
SUVR_THRESHOLD = 1.11   # AV45 amyloid-positivity threshold (cerebellar reference)
SLOPE = 0.33            # affected fraction gained per SUVR unit above threshold
RATIO_MAX = 0.30        # cap on the affected-neuron fraction

# Amyloid-positive subjects (composite SUVR positive; 002_S_1155 is negative).
POSITIVE_SUBJECTS = ['002_S_0729', '002_S_5018', '305_S_6810']

# FreeSurfer subcortical structures have no counterpart in the cortical model.
SUBCORTICAL = {
    'accumbens', 'amygdala', 'caudate', 'hippocampus',
    'pallidum', 'putamen', 'thalamus',
}

HERE = os.path.dirname(os.path.abspath(__file__))
ADNI_DIR = os.path.join(HERE, '..', 'ADNI')


def _model_area_names():
    """Valid model area names (Desikan-Killiany full names)."""
    from data_loader.dk_fullnames_to_shortnames import dk_full_to_short
    return set(dk_full_to_short.keys())


def _fs_base_to_area(region):
    """'rh_rostral_middle_frontal' -> 'rostralmiddlefrontal' (or None)."""
    if not region.startswith(('lh_', 'rh_')):
        return None
    base = region[3:]
    if base in SUBCORTICAL:
        return None
    return base.replace('_', '')


def suvr_to_ratio(mean_suvr):
    """Map a regional mean SUVR to an affected-neuron fraction in [0, RATIO_MAX]."""
    return round(min(max((mean_suvr - SUVR_THRESHOLD) * SLOPE, 0.0), RATIO_MAX), 4)


def build_subject_csv(subject, valid_areas):
    """Build the (area, ratio) table for one subject."""
    src = os.path.join(ADNI_DIR, subject, 'amyloid_output', 'suvr_by_region.csv')
    df = pd.read_csv(src)
    df['area'] = df['region'].map(_fs_base_to_area)
    df = df.dropna(subset=['area'])

    unknown = sorted(set(df['area']) - valid_areas)
    if unknown:
        raise ValueError('{}: regions not in model area list: {}'.format(
            subject, unknown))

    # Average hemispheres, then map to a fraction.
    per_area = df.groupby('area')['mean_suvr'].mean()
    rows = [('default', '', 0.0)]
    for area, suvr in per_area.sort_index().items():
        rows.append((area, '', suvr_to_ratio(suvr)))
    return pd.DataFrame(rows, columns=['area', 'layer', 'ratio'])


def main():
    valid_areas = _model_area_names()
    for subject in POSITIVE_SUBJECTS:
        out = build_subject_csv(subject, valid_areas)
        out_fn = os.path.join(HERE, '{}_abeta_ratio.csv'.format(subject))
        out.to_csv(out_fn, index=False)
        print('wrote {} ({} areas)'.format(out_fn, len(out) - 1))


if __name__ == '__main__':
    main()
