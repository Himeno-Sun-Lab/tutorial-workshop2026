# Simulating a Disease-Conditioned Human Cortex

A self-contained tutorial for a **multi-area spiking network model of one human cerebral
hemisphere** — built on the Human Multi-Area Model (HuMAM) and the NEST simulator, and extended with
amyloid-beta pathology.

**Everything is in this one repository.** The model's source and every dataset it reads are bundled
in `model/` (8.5 MB). Clone this, install NEST, and run — there is nothing else to fetch.

The tutorial teaches **the model**: what it represents, what its parameters mean physically, how to
run it, and how to read what comes out. It is deliberately *not* a walkthrough of the source code —
the model is treated the way you would treat any scientific instrument.

At full scale it has 3.47 M neurons and 42.8 G synapses. The tutorial runs the same model,
downscaled by a factor of 2000, on a laptop, then shows how to send the full-scale version to a
cluster.

## The interface you learn

```
experiments/my_experiment.py         you write this: three dictionaries
        |
        v
  network  ->  simulate  ->  analyse        three commands
        |
        v
results/<hash>/<hash>/<hash>/plots/   ~200 figures and a set of result files
```

## Contents
|File|Detail|
|---|---|
| `Cortical_Model_Tutorial.ipynb` | The tutorial. 56 cells, English |
| `environment.yml` | The conda environment, pinned — `mamba env create -f environment.yml` |
| `experiments/tutorial_CN.py` | Healthy cortex — a fully commented experiment file, the model's control panel |
| `experiments/tutorial_AD.py` | The same network with an amyloid-PET-derived pathology distribution |
| `experiments/tutorial_subject.py` | One ADNI participant, individualised end to end: their connectome, their amyloid map, their fMRI as the comparison target |
| `tools/eta_ext_search.py` | Calibrates the external drive for a downscaled network (see Part 3) |
| `model/` | The model: source and all the data it reads. A snapshot — see `model/README.md` |
| `model/experimental_data/ADNI/` | One complete participant: structural connectivity, resting-state BOLD and amyloid-PET — see its README |
| `cluster/submit_model.sh` | Self-contained SLURM job: clones this repository on the cluster, writes full-scale experiment files, runs the three stages |
| `results/` | Created on the first run. Safe to delete |

Nothing writes into `model/`: every run puts its output in `results/`, which is gitignored.

## Requirements
### OS
| Platform | Status | How |
|---|---|---|
| **macOS** 12+ (Intel or Apple Silicon) | supported | native conda/mamba |
| **Linux** (Ubuntu 20.04+, RHEL 8+) | supported | native conda/mamba |
| **Windows 10/11** | supported **through WSL2** | NEST has **no native Windows build** |

**NEST has no native Windows build.** On Windows, run everything inside WSL2 (or the official NEST
Docker image) and attach VS Code with the WSL extension.

### Specs

| Specs | **Minimum** | **Recommended** | Why |
|---|---|---|---|
| **CPU** | 4 physical cores | 8+ cores | NEST parallelises over OpenMP threads |
| **RAM** | **8 GB** | **16 GB** | The default tutorial network needs 3.1 GB measured; the `tiny` variant needs 1.3 GB |
| **Free disk** | 8 GB | 20 GB | ~4 GB environment, ~1.5 GB repository and its data, ~120 MB per completed run |
| **GPU** | not used | not used | NEST 3.8 is a CPU simulator |

**Laptop notes.** Plug in - both macOS and Windows down-clock hard on battery. Expect fans: a 5 s
simulation is ~19 s of full-throttle multicore compute. If a cell seems to hang for minutes, you are
probably swapping; stop it and use a smaller network (Part 2).

Compute time is about **8 minutes**, in three marked blocks: two full pipeline runs (~3 min each)
and an optional seed check (~80 s).


## Quick start

```bash
# 1. get the tutorial (once) -- the model is bundled inside it
git clone <TUTORIAL_REPOSITORY_URL> ~/Codes/Cortical-Model-Tutorial

# 2. create the environment (once) -- every version pinned
cd ~/Codes/Cortical-Model-Tutorial
mamba env create -f environment.yml
mamba activate nest38
python -m ipykernel install --user --name nest38 --display-name "Python (NEST 3.8)"

# 3. verify NEST before anything else
python -c "import nest; print(nest.__version__)"    # expect 3.8.0

# 4. launch
cd ~/Codes/Cortical-Model-Tutorial
jupyter lab Cortical_Model_Tutorial.ipynb
```

The notebook is saved against a kernel named `nest38`, which is why the command above registers that
name; if yours differs, pick it from Jupyter's kernel selector. NEST is the only thing the tutorial
cannot bundle.

### Running a smaller or larger model

Edit `N_scaling` in `experiments/tutorial_CN.py` (and `tutorial_AD.py`):

| `N_scaling` | Neurons | Synapses | Peak RAM | Simulate 5 s |
|---|---|---|---|---|
| 0.002 | 6.9 k | 8.6 M | 1.3 GB | ~7 s |
| **0.005** (default) | 17 k | 21 M | 3.1 GB | ~19 s |
| 0.02 | 69 k | 86 M | ~10.6 GB | ~75 s |

Hold `K_scaling` at 0.1 — it is the knob that changes the dynamics, and 0.1 is the value the project
validates. See Part 3 of the notebook.

## What the tutorial covers

1. Requirements, installation, platform caveats
2. **What the model represents** — three scales, the data behind every number, the neuron and
   synapse equations, and what each simulated signal maps onto experimentally
3. **The control panel** — the three parameter dictionaries and the knobs that matter
4. **Downscaling** — the mean/variance argument, why a careless downscale produces a *silent*
   network, and how to recalibrate the drive
5. **Amyloid-beta** — from an amyloid-PET map to model parameters
6. **Running the model** — the three-stage pipeline and hash-addressed results
7. **Reading the output** — the pipeline's own figures, in the order you should consult them, plus a
   reusable checklist
8. **The experiment** — healthy vs amyloid, and a seed check that decides which differences are real
9. **Beyond the laptop** — Snakemake, job sizing, and the SLURM batch job
10. Exercises, troubleshooting, references

## Calibrating the external drive

Downscaling leaves the mean synaptic drive short, and at the default `eta_ext = 1.1` a downscaled
network emits nothing at all. `tools/eta_ext_search.py` finds the drive that fixes it:

```bash
cd ~/Codes/Cortical-Model-Tutorial

python tools/eta_ext_search.py                   # simulate + bisect (default)
python tools/eta_ext_search.py --method analytic # instant closed-form estimate
python tools/eta_ext_search.py --method theory   # NNMT mean-field, ~5 s per point
python tools/eta_ext_search.py --help
```

Measured at `N_scaling = 0.005, K_scaling = 0.1`: the network is silent below `eta_ext` **1.38**, and
reaches the 3–5 Hz excitatory target band at **3.45** — the value the tutorial experiments use.
Results and a rate-vs-`eta_ext` figure go to `results/eta_search/`.

## Distributing this

Verified by running the whole pipeline from a relocated copy of this folder: the results were
bit-identical (E 3.016 Hz, I 12.116 Hz, 198 figures). Nothing depends on an absolute path, an
environment variable, or a second repository.

A participant needs exactly three things:

| Requirement | Notes |
|---|---|
| **This repository** | ~9 MB, model and data included |
| **NEST 3.8 + the Python stack** | `mamba env create -f environment.yml` — 222 packages, 158 MB download, solve verified |
| **A Jupyter kernel with NEST on it** | registered as `nest38`, or selected by hand |

plus 8–16 GB RAM, ~10 GB free disk (~120 MB per completed run), and Linux/macOS or WSL2 on Windows.

`results/`, the pipeline's stage markers and generated seed variants are gitignored, so the
repository stays at its clone size however many runs you do.

**On the bundled copy.** `model/` is a snapshot, not a submodule: it does not track the upstream
model and will not receive its fixes. Refresh it by copying the source and data directories over
again. The amyloid distribution in `model/experimental_data/abeta/` is derived from ADNI data, whose
data use agreement applies wherever this repository is shared.

## Individual data

`model/experimental_data/ADNI/` holds one complete participant (`023_S_6661`): structural
connectivity from diffusion MRI, 835 frames of resting-state BOLD, and an amyloid-PET-derived burden
map — all in the same 34-area Desikan-Killiany parcellation the model uses.
`experiments/tutorial_subject.py` wires the three together, so the simulated functional connectivity
becomes a prediction about *that person*, checkable against *their own* fMRI. Part 7.4 of the
notebook runs it (opt-in) and discusses what it does to the working point.

ADNI's Data Use Agreement applies to this data wherever the repository is shared, and requires the
standard ADNI acknowledgement in any publication.

## Model reference

> Pronold, J., van Meegen, A., Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C.,
> Bakker, R., & van Albada, S. J. (2024). Multi-scale spiking network model of human cerebral
> cortex. *Cerebral Cortex*. https://doi.org/10.1093/cercor/bhae409
