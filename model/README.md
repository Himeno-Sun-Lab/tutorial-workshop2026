# The model

This directory is a **self-contained snapshot** of the multi-area cortical model the tutorial runs:
its source and every dataset it reads. Nothing else needs to be downloaded or cloned.

```
src/                     model code
  network.py                 builds neuron numbers, connectivity, weights, delays, downscaling
  simulation.py              instantiates and runs the network in NEST
  analysis.py                spike statistics, LFP proxy, coherence, PLV, BOLD, figures
  default_*_params.py        every parameter and its default, with comments
  snakemake_*.py             the three pipeline stages the tutorial calls
  data_loader/               readers for each dataset below
  data_preprocessing/        cytoarchitecture and predictive connectomics
  theory/rates.py            NNMT mean-field solver
experimental_data/       the measurements the model is built from
  voneconomokoskinas/        layer-resolved neuron densities
  fraction_EI/               excitatory / inhibitory ratios per layer
  hcp_dti/                   diffusion-tractography connectome and distances
  senden/                    7T resting-state fMRI, used as the reference for functional connectivity
  amyloid/                   ADNI amyloid-PET map, and one participant's connectome and BOLD
simulated_data/          precomputed full-scale mean-field rates (needed for downscaling)
misc/mplstyles/          figure style used by the analysis stage
```

The tutorial never writes here: all output goes to `../results/`.

**Upstream.** The model extends the Human Multi-Area Model (HuMAM):

> Pronold, J., van Meegen, A., Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C.,
> Bakker, R., & van Albada, S. J. (2024). Multi-scale spiking network model of human cerebral
> cortex. *Cerebral Cortex*. https://doi.org/10.1093/cercor/bhae409

Licence: see `LICENSE`. Each dataset directory carries its own README with the original source and
citation. The amyloid distribution is derived from ADNI data; ADNI's data use agreement applies to
it.
