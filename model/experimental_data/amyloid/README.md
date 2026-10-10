# Amyloid data

Everything the tutorial needs for its amyloid experiments, from participants of the **Alzheimer's
Disease Neuroimaging Initiative**, all parcellated into the 34 Desikan-Killiany areas per hemisphere
the model uses.

| File | What it is | Shape | Feeds | Used by |
|---|---|---|---|---|
| `amyloid_ratio.csv` | **Amyloid burden** per area, from amyloid-PET | 33 areas + a `default` row; `ratio` in [0, 1] | `net_params["neuron_types"]["Abeta"]["ratio_E"]` | `experiments/tutorial_amyloid.py` (Appendix B), B.8 |
| `amyloid_SC.mat` | **Structural connectivity** from diffusion MRI tractography | `C_L`, `C_R`, `Dist` each 34x34, plus `DK_Names` | `net_params["predictive_connectomic_params"]["con_path"]` | 5.1, B.8 |
| `amyloid_bold.npy` | **Resting-state BOLD** time series | `(835, 68, 1)` = frames x DK areas (lh 0:34, rh 34:68) x participants | `ana_params["experimental_fc"]` | 5.1, B.8 |

`amyloid_SC.mat` and `amyloid_bold.npy` belong to **one participant**: that person's anatomy and
measured dynamics to compare against. `amyloid_ratio.csv` comes from **another** ADNI participant;
for a fully individual model, replace it with the first person's own map in the same format
(`area,layer,ratio`).

## The amyloid map

- Mean **0.158** of excitatory neurons affected, range 0.035-0.296, heaviest in rostral middle
  frontal cortex, frontal pole, precuneus and posterior cingulate. Areas not listed take the
  `default` row.

## The connectome and BOLD participant

- Resting-state scan: 835 frames - long by ADNI standards, which makes the functional connectivity
  reference far less noisy than the 5 s of simulated BOLD it will be compared against.
- Structural connectome: 964 of 1156 possible intra-hemispheric edges are non-zero.

The `.mat` has exactly the same variable layout as the group-average HCP connectome in
`../hcp_dti/`, so it is a drop-in replacement: change `con_path` and the network is rebuilt on this
person's wiring instead of the group average.

## Provenance and terms

Data from the **Alzheimer's Disease Neuroimaging Initiative** (ADNI, adni.loni.usc.edu),
preprocessed into DK-parcellated connectomes, BOLD time series and amyloid ratios. ADNI's Data Use
Agreement applies wherever this repository is shared: it covers redistribution and requires the
standard ADNI acknowledgement in any publication that uses it.

The amyloid ratio file is derived, not raw: regional amyloid-PET SUVR converted to a fraction of
affected excitatory neurons.
