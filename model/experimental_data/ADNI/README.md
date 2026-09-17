# ADNI participant 023_S_6661

One complete individual dataset: **structural connectivity, functional connectivity and amyloid
distribution for the same person**, all parcellated into the 34 Desikan-Killiany areas per
hemisphere the model uses. It is what makes patient-specific simulation possible — the same
participant's anatomy, the same participant's pathology, and the same participant's measured
dynamics to compare against.

| File | What it is | Shape | Feeds |
|---|---|---|---|
| `023_S_6661_sc.mat` | **Structural connectivity** from diffusion MRI tractography | `C_L`, `C_R`, `Dist` each 34x34, plus `DK_Names` | `net_params["predictive_connectomic_params"]["con_path"]` |
| `023_S_6661_bold.npy` | **Resting-state BOLD** time series | `(835, 68, 1)` = frames x DK areas (lh 0:34, rh 34:68) x participants | `ana_params["experimental_fc"]` |
| `023_S_6661_abeta_ratio.csv` | **Amyloid burden** per area, from amyloid-PET | 34 areas + a `default` row; `ratio` in [0, 1] | `net_params["Abeta_ratio_E"]` |

The `.mat` has exactly the same variable layout as the group-average HCP connectome in
`../hcp_dti/`, so it is a drop-in replacement: change `con_path` and the network is rebuilt on this
person's wiring instead of the group average.

## This participant

- Amyloid burden: mean **0.267** of excitatory neurons affected, range 0.014–0.300, heaviest in
  bankssts, caudal middle frontal, cuneus, inferior parietal and lateral occipital cortex.
- Resting-state scan: 835 frames — long by ADNI standards, which makes the functional connectivity
  reference far less noisy than the 5 s of simulated BOLD it will be compared against.
- Structural connectome: 964 of 1156 possible intra-hemispheric edges are non-zero.

`../../../experiments/tutorial_subject.py` wires all three together.

## Provenance and terms

Data from the **Alzheimer's Disease Neuroimaging Initiative** (ADNI, adni.loni.usc.edu),
preprocessed into DK-parcellated connectomes, BOLD time series and amyloid ratios. ADNI's Data Use
Agreement applies wherever this repository is shared: it covers redistribution and requires the
standard ADNI acknowledgement in any publication that uses it.

The amyloid ratio file is derived, not raw: regional amyloid-PET SUVR converted to a fraction of
affected excitatory neurons. `../abeta/generate_abeta_csv.py` shows the conversion.
