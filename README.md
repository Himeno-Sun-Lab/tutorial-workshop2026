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
downscaled, on a laptop, then shows a full-scale run computed on a server.

## The interface you learn

```
experiments/my_experiment.py         you write this: three dictionaries
        |
        v
  network  ->  simulate  ->  analyse        three commands
        |
        v
results/<run>/<hash>/<hash>/<hash>/plots/   ~200 figures and a set of result files
```

On the laptop every run starts from scratch: `run_all("tutorial_CN")` empties `results/tutorial_CN/`
and runs all three stages again, so nothing is reused between runs or shared between experiments.
Reuse - re-running only the stages whose parameters changed - belongs to the server's Snakemake
workflow, where the full-scale runs were computed. Participants do not run anything there: the
notebook shows three such runs, as figures rendered from the runs themselves (`figures/fullscale/`).

## Contents
|File|Detail|
|---|---|
| `Cortical_Model_Tutorial.ipynb` | The tutorial. 108 cells, English |
| `environment.yml` | The conda environment, pinned — `mamba env create -f environment.yml` |
| `experiments/tutorial_CN.py` | Healthy cortex — a fully commented experiment file, the model's control panel |
| `experiments/tutorial_amyloid.py` | The same network with an amyloid-PET-derived pathology distribution |
| `experiments/fullscale_rates/` | Full-scale rates for each experiment, from which the downscaling DC compensation is computed |
| `tools/fullscale_rates.py` | Generates those rates by self-consistent iteration; re-run after changing an experiment's drive, weights or anatomy |
| `tools/eta_ext_search.py` | Calibrates the external drive for a downscaled network (see Appendix A) |
| `tools/pipeline.py` | The notebook's plumbing: paths, running an experiment from scratch, the hash files, loading the figures and tables of a laptop run or the imported full-scale run |
| `model/` | The model: source and all the data it reads. A snapshot — see `model/README.md` |
| `model/experimental_data/amyloid/` | The amyloid data: the amyloid-PET map of the amyloid experiment, and one participant's structural connectivity and resting-state BOLD — see its README |
| `figures/fullscale/<run>/` | The figures of Part 6.2, rendered from the full-scale runs, and the numbers the text quotes (`numbers.json`) |
| `tools/make_fullscale_figures.py` | For developers: re-renders `figures/fullscale/` from the runs in `fullscale/` - see below |
| `fullscale/` | Three full-scale runs from the server, imported unchanged. **Local only, gitignored** - see below |
| `results/` | Created on the first run, one folder per run. Safe to delete |

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

Compute time is about **12 minutes**, in three marked blocks: two full pipeline runs (~3 min each)
and an optional seed check (~6 min - every extra seed builds and simulates its own network).


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

Edit `N_scaling` in `experiments/tutorial_CN.py` (and `tutorial_amyloid.py`):

| `N_scaling` | Neurons | Synapses | Peak RAM | Simulate 5 s |
|---|---|---|---|---|
| 0.002 | 6.9 k | 8.6 M | 1.3 GB | ~7 s |
| **0.005** (default) | 17 k | 21 M | 3.1 GB | ~19 s |
| 0.02 | 69 k | 86 M | ~10.6 GB | ~75 s |

Hold `K_scaling` at 0.1 — it is the knob that changes the dynamics, and 0.1 is the value the project
validates. See Appendix A of the notebook.

## What the tutorial covers

0. **Quick start** — environment, a one-click check that NEST works
1. **What the model represents** — architecture, the data behind every number, the neuron and
   synapse equations, and what each simulated signal maps onto experimentally
2. **The control panel** — the experiment file, its key parameters, and choosing a size for your machine
3. **Running the model** — the three-stage pipeline and hash-addressed results
4. **Reading the output** — the pipeline's own figures, in the order you should consult them, plus a
   reusable checklist
5. **Your own data** — a connectome, a comparison target, a pathology map, your own neuron parameters
6. **Beyond the laptop** — the Snakemake workflow, and three full-scale runs (6.2) in three different
   network states, shown as pre-rendered figures
then references, troubleshooting, and two appendices: **A** downscaling and recalibrating the drive;
**B** amyloid-beta - from an amyloid-PET map to model parameters, a healthy-vs-amyloid experiment
with a seed check that decides which differences are real, and one person's amyloid map.

## Calibrating the external drive

Downscaling keeps only `sqrt(K_scaling)` of the mean synaptic input; the model restores the rest as a
DC current computed from the rates of the *full-scale* network (`fullscale_rates`). Those rates must
belong to the configuration being run, and they depend on `eta_ext`, so the experiments ship their
own, generated by self-consistent iteration (rates → DC → simulate → rates):

```bash
python tools/fullscale_rates.py experiments/tutorial_CN.py --iters 40 \
    --out experiments/fullscale_rates/tutorial_CN.pkl      # ~9 min at N_scaling = K_scaling = 0.01
```

Re-run it for an experiment whenever you change its drive, weights or anatomy. `eta_ext` itself sets
the working point; `tools/eta_ext_search.py` scans it. Like `fullscale_rates.py` it reads the
experiment file itself - every `net_params` entry, its full-scale rates and its `master_seed` - and
varies only `eta_ext`; flags such as `--n-scaling` or `--seeds` override single values:

```bash
cd ~/Codes/Cortical-Model-Tutorial

python tools/eta_ext_search.py --experiment experiments/tutorial_CN.py \
    --method scan --eta-list 4.0 4.5 5.0 5.5     # the experiment as it is
python tools/eta_ext_search.py                   # simulate + bisect (default)
python tools/eta_ext_search.py --method analytic # instant closed-form estimate
python tools/eta_ext_search.py --method theory   # NNMT mean-field, ~5 s per point
python tools/eta_ext_search.py --help
```

Measured at `N_scaling = 0.005, K_scaling = 0.1` with self-consistent full-scale rates
(population means; E 3–5 Hz and I 5–15 Hz are the target bands):

| `eta_ext` | 1.1 | 2.0 | 3.0 | 4.5 | **5.0** | 5.5 | 7.0 |
|---|---|---|---|---|---|---|---|
| E (Hz) | 0.31 | 0.86 | 1.66 | 2.99 | **3.58** | 3.81 | 5.11 |
| I (Hz) | 1.5 | 4.3 | 7.5 | 12.2 | **13.7** | 15.1 | 19.7 |

Both bands hold only for `eta_ext` ≈ 4.5–5.4; the experiments use **5.0** (CN E 3.58 Hz, I 13.7 Hz;
amyloid E 3.46 Hz, I 14.0 Hz, at seed 2903). Results and a
rate-vs-`eta_ext` figure go to `results/eta_search/`.

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

`results/` (every laptop run, with its own stage markers) and `fullscale/` are gitignored, so the
repository stays at its clone size however many runs you do.

**On the bundled copy.** `model/` is a snapshot, not a submodule: it does not track the upstream
model and will not receive its fixes. Refresh it by copying the source and data directories over
again. The amyloid data in `model/experimental_data/amyloid/` are derived from ADNI data, whose
data use agreement applies wherever this repository is shared.

## The full-scale result

Part 6.2 of the notebook shows three full-scale runs - 3.47 M neurons, 42.8 G synapses, 10 s each -
computed on our server with Snakemake: `sub007_L`, `sub016_L` and `sub011_R`, one
hemisphere each of three participants of the THU glioma cohort, each on that person's own connectome
and compared against their own resting-state BOLD. All parameters but the connectome are identical,
and the three end up in different network states: one switches once from a bursting to a quiet state,
one switches back and forth, one never leaves the bursting state. The two that switch have the
cohort's highest and lowest correlation between the FC of the simulated synaptic currents and the
person's fMRI (r = 0.47 and -0.08; median 0.20); Part 6.2 shows what those numbers do and do not mean.

The notebook shows them as figures in `figures/fullscale/<run>/`; nothing else is needed to read
Part 6.2.

**For developers: re-rendering the figures.** The figures, and the numbers the notebook text quotes
(`numbers.json`), are rendered by `tools/make_fullscale_figures.py`. Participants never run it, and
the notebook does not refer to it. It needs the runs themselves, imported exactly as the workflow
wrote them, so their hashes still name their directories; only the files in `experiments/` and
`logs/` are named after the run, `sub<NNN>_<L|R>` (shown for `sub007_L`):

```
fullscale/experiments/sub007_L.py                                     the experiment file
fullscale/experiments/sub007_L.{network,simulation,analysis}_hash
fullscale/out/<net_hash>/<sim_hash>/<ana_hash>/                       all three levels, spikes included
fullscale/logs/sub007_L_*                                             the Snakemake rule logs
fullscale/experimental_data/thu-glioma/thumatrix/sub007_L.npy         the person's BOLD, under its server path
```

The runs are patient-derived and therefore **not part of the repository** (`fullscale/` is
gitignored); the rendered figures are. Another run goes in the same way: copy its
experiment file and hash files into `fullscale/experiments/` under the run's name, its `<net_hash>/` tree into
`fullscale/out/`, the BOLD file its `ana.yaml` names into `fullscale/experimental_data/` under the
same relative path, add it to `RUNS` in `tools/make_fullscale_figures.py`, re-run the script and
commit `figures/fullscale/`.

## Individual data

`model/experimental_data/amyloid/` holds the amyloid-PET-derived burden map of the amyloid
experiment (`amyloid_ratio.csv`, the `ratio_E` of its `"Abeta"` neuron type), and one participant's
structural connectivity from diffusion MRI (`amyloid_SC.mat`) and 835 frames of resting-state BOLD
(`amyloid_bold.npy`) — all in the same 34-area Desikan-Killiany parcellation the model uses. Point
`predictive_connectomic_params["con_path"]` and `ana_params["experimental_fc"]` of an experiment at
the last two to simulate that person and compare against their own fMRI. The amyloid map comes
from a different ADNI participant; replace it with the person's own map for a fully individual model.

ADNI's Data Use Agreement applies to this data wherever the repository is shared, and requires the
standard ADNI acknowledgement in any publication.

## Model reference

> Pronold, J., van Meegen, A., Shimoura, R. O., Vollenbröker, H., Senden, M., Hilgetag, C. C.,
> Bakker, R., & van Albada, S. J. (2024). Multi-scale spiking network model of human cerebral
> cortex. *Cerebral Cortex*. https://doi.org/10.1093/cercor/bhae409
