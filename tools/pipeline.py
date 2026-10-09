#!/usr/bin/env python3
"""Plumbing for the tutorial notebook: paths, running a model, reading its output.

None of this is part of the model.  It is the bookkeeping that sits between the
notebook and the three pipeline stages:

* where the tutorial, the bundled model and the results live,
* how a run is started (``subprocess`` into ``model/src/snakemake_*.py``),
* how the hash files a run writes are turned into the directory it wrote to,
* how a figure or a result file of a finished run is loaded - a laptop run, or
  a full-scale result brought back from the cluster.

**No reuse on the laptop.**  Every ``run_all()`` starts from an empty
``results/<run>/`` and runs all of its stages again: nothing is skipped because
a hash matches, nothing is read from an earlier run's cache, and two runs never
share a directory.  A run costs minutes here, and a result you can trust to
come from the parameters on screen is worth more than the minutes.  Reuse - only
the stages whose parameters changed are re-run - is what the cluster workflow
(``cluster/Snakefile``) is for, where a stage costs hours.

Layout of one run::

    results/<run>/experiment.py            driver: experiments/<experiment>.py,
                                           with outpath pointed here
    results/<run>/{network,simulation,analysis}_hash
    results/<run>/log/<stage>.log
    results/<run>/<net_hash>/<sim_hash>/<ana_hash>/plots/ ...

A full-scale result is a bundle the cluster workflow writes to
``fullscale/<run>/`` (``run.json``, the result tables, the summary figures).
Every reader below - ``show``, ``rates``, ``load_result``, ``area_mean_rate`` -
accepts its name in place of a laptop run.

The notebook imports from here so that its cells show the *model* - parameters,
rates, pathology - rather than path juggling.  Everything is importable outside
the notebook as well::

    import sys; sys.path.insert(0, "tools")
    from pipeline import run_all, rates

Nothing here writes into ``model/``: every output goes to ``results/``.
"""

import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------- paths --
# Resolved from this file, not from the working directory, so the notebook works
# no matter which directory the kernel was started in.
TUTORIAL = Path(__file__).resolve().parent.parent
EXPERIMENTS = TUTORIAL / "experiments"
RESULTS = TUTORIAL / "results"
FULLSCALE = TUTORIAL / "fullscale"
MODEL_ROOT = TUTORIAL / "model"

STAGES = ("network", "simulate", "analyse")

# The hash file each stage writes into its run directory.
STAGE_HASH_FILE = {"network": "network_hash", "simulate": "simulation_hash",
                   "analyse": "analysis_hash"}

# The tutorial's own helpers (the drive-calibration tool doubles as a spike reader).
if str(TUTORIAL / "tools") not in sys.path:
    sys.path.insert(0, str(TUTORIAL / "tools"))


def check_environment(verbose=True):
    """Verify the bundled model and NEST are there, and report the setup.

    Raises ``SystemExit`` if the model is missing and ``ImportError`` if NEST is
    not installed - both failures are better seen in the first cell than halfway
    through a run.
    """
    import numpy as np

    if not (MODEL_ROOT / "src" / "snakemake_network.py").exists():
        raise SystemExit(
            f"The bundled model is missing from {MODEL_ROOT}.\n"
            "Re-clone this tutorial repository, or run the notebook from its own folder.")

    RESULTS.mkdir(parents=True, exist_ok=True)

    import nest                                 # fails early if NEST is missing

    if verbose:
        print(f"bundled model   : {MODEL_ROOT}")
        print(f"Results go to   : {RESULTS}")
        print(f"NEST            : {nest.__version__}")
        print(f"numpy / pandas  : {np.__version__} / {pd.__version__}")
        print(f"CPU cores       : {os.cpu_count()}")
    return nest.__version__


# ---------------------------------------------------------------- running a run --
def run_all(experiment, stages=STAGES, threads=4, quiet=False, title=None,
            seed=None):
    """Run `experiment` from scratch: wipe its run directory, then every stage.

    The run is called `experiment`, or ``<experiment>_seed<seed>`` when `seed`
    replaces the experiment's ``master_seed``.  Whatever an earlier run left in
    ``results/<run>/`` is deleted first, so every number and figure read back
    afterwards was produced by this call.  Each line printed is one stage: its
    time and the hash it wrote.  Returns the run's name.
    """
    run = experiment if seed is None else f"{experiment}_seed{seed}"
    root = RESULTS / run
    shutil.rmtree(root, ignore_errors=True)
    (root / "log").mkdir(parents=True)
    _write_driver(experiment, root, seed)

    if not quiet:
        print(title or run)
    total = sum(run_stage(stage, run, threads=threads, quiet=quiet)
                for stage in stages)
    if not quiet:
        print(f"  {'total':9s} {total:6.1f} s")
    return run


def _write_driver(experiment, root, seed=None):
    """The experiment file the stages are given: yours, with outpath set to `root`.

    It executes ``experiments/<experiment>.py`` unchanged and only redirects the
    output (and, for a seed variant, ``master_seed``), so the run directory is
    private to this run and the experiment file stays the single source of truth.
    """
    lines = [
        f'"""Driver for one run of {experiment}.py, written by tools/pipeline.py.',
        "",
        f"Edit experiments/{experiment}.py, not this file: it is rewritten on every run.",
        '"""',
        "import runpy",
        "",
        f"_exp = runpy.run_path({str(EXPERIMENTS / f'{experiment}.py')!r})",
        'net_params = _exp["net_params"]',
        'sim_params = _exp["sim_params"]',
        'ana_params = _exp["ana_params"]',
        "",
        f'net_params["outpath"] = {str(root)!r}',
    ]
    if seed is not None:
        lines.append(f'sim_params["master_seed"] = {int(seed)}')
    (root / "experiment.py").write_text("\n".join(lines) + "\n")


def run_stage(stage, run, threads=4, quiet=False):
    """One stage of the run ``run_all`` has just started; one subprocess call.

    It is the same command the Snakefile issues on a cluster.  The stages talk
    to each other only through the hash files in ``results/<run>/``; a stage
    whose predecessor has not run in this run directory fails.
    """
    root = RESULTS / run
    exp = root / "experiment.py"
    log = root / "log" / f"{stage}.log"
    h = {s: root / f for s, f in STAGE_HASH_FILE.items()}
    cmd = {
        "network":  ["python", "src/snakemake_network.py", str(exp),
                     str(h["network"]), str(log)],
        "simulate": ["python", "src/snakemake_simulation.py", str(exp),
                     str(h["network"]), str(h["simulate"]), str(threads), str(log)],
        "analyse":  ["python", "src/snakemake_analysis.py", str(exp),
                     str(h["network"]), str(h["simulate"]), str(h["analyse"]),
                     str(MODEL_ROOT), str(log)],
    }[stage]

    t0 = time.time()
    # cwd = the model repository: it resolves its data paths from the cwd.
    proc = subprocess.run(cmd, cwd=MODEL_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"stage '{stage}' failed for {run} - log: {log}")
    if not quiet:
        print(f"  {stage:9s} {elapsed:6.1f} s   {stage_hash(run, stage)}")
    return elapsed


# ------------------------------------------------------------ hashes and paths --
def fullscale_bundle(run):
    """The directory of a full-scale bundle called `run`, or None."""
    d = FULLSCALE / run
    return d if (d / "run.json").exists() else None


def has_fullscale(run):
    """Whether the full-scale result `run` has been brought back from the cluster."""
    return fullscale_bundle(run) is not None


def fullscale_info(run):
    """What a full-scale bundle was run with: scale, drive, seed, commit, hashes."""
    d = fullscale_bundle(run)
    if d is None:
        raise FileNotFoundError(_missing(run))
    return json.loads((d / "run.json").read_text())


def hashes(run):
    """The hashes the run has written so far, as a dict."""
    d = fullscale_bundle(run)
    if d is not None:
        return fullscale_info(run)["hashes"]
    out = {}
    for stage, name in STAGE_HASH_FILE.items():
        f = RESULTS / run / name
        if f.exists():
            out[name.removesuffix("_hash")] = f.read_text().strip()
    return out


def stage_hash(run, stage):
    """The hash one stage wrote, or '-' if its file is not there."""
    f = RESULTS / run / STAGE_HASH_FILE[stage]
    return f.read_text().strip() if f.exists() else "-"


def print_hashes(run):
    """Show the hash of every stage that has run."""
    for stage, h in hashes(run).items():
        print(f"{stage:11s} hash  {h}")


def _missing(run):
    if run.startswith("full"):
        return (f"no full-scale result '{run}' in {FULLSCALE} - run the cluster "
                "workflow (Part 7) and copy its fullscale/ folder here")
    return f"no completed run '{run}' - run_all({run!r}) first"


def _hashes_for(run, *stages):
    h = hashes(run)
    if not all(s in h for s in stages):
        raise FileNotFoundError(_missing(run))
    return [h[s] for s in stages]


def net_dir(run):
    """The network directory of a completed stage 1 (laptop runs only)."""
    (nh,) = _hashes_for(run, "network")
    return RESULTS / run / nh


def sim_dir(run):
    """The simulation directory of a completed stage 2 (a bundle holds its tables)."""
    d = fullscale_bundle(run)
    if d is not None:
        return d
    nh, sh = _hashes_for(run, "network", "simulation")
    return RESULTS / run / nh / sh


def run_dir(run):
    """The analysis directory of a completed run, or a full-scale bundle."""
    d = fullscale_bundle(run)
    if d is not None:
        return d
    nh, sh, ah = _hashes_for(run, "network", "simulation", "analysis")
    return RESULTS / run / nh / sh / ah


def describe_outputs(run):
    """Inventory of what the analysis stage wrote: figures, figure sets, result files."""
    d = run_dir(run)
    print(f"analysis directory: {d.relative_to(TUTORIAL)}\n")

    plots = sorted(p for p in (d / "plots").glob("*.png"))
    groups = sorted(p for p in (d / "plots").iterdir() if p.is_dir())
    data = sorted(p for p in d.glob("*.pkl"))

    print(f"{len(plots)} summary figures, e.g.")
    for p in plots[:6]:
        print("   ", p.name)
    print(f"\n{len(groups)} per-area figure sets:")
    for g in groups:
        print(f"    {g.name}/  ({len(list(g.iterdir()))} areas)")
    print(f"\n{len(data)} result files you can load directly:")
    for p in data:
        print("   ", p.name)

    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**2
    print(f"\ntotal {size:.0f} MB")


# --------------------------------------------------------- reading a run's data --
def show(run, name, width=900):
    """Display one figure produced by the analysis stage."""
    from IPython.display import Image, display

    path = run_dir(run) / "plots" / name
    if not path.exists():
        print(f"not found: {name}")
        return
    display(Image(filename=str(path), width=width))


def rates(run):
    """Population firing rates (Hz), MultiIndex (area, layer, pop)."""
    return pd.read_pickle(sim_dir(run) / "rates.pkl")


def pop_level(series):
    """Name of the population level of a rate index ('E' / 'I' live there)."""
    return series.index.names[-1]


def area_mean_rate(run, pop="E"):
    """Mean rate per area for one population, as a Series indexed by area."""
    r = rates(run)
    return r.xs(pop, level=pop_level(r)).groupby(level="area").mean()


def load_result(run, name):
    """Load one pickled result file of the analysis stage, e.g. 'lfp_proxy_kernel.pkl'."""
    return pd.read_pickle(run_dir(run) / name)


def fullscale_estimate(experiment):
    """The full-scale rates an experiment's DC compensation is computed from.

    ``experiments/fullscale_rates/<experiment>.pkl``: the self-consistent
    estimate of what the full-scale network fires at, generated by
    ``tools/fullscale_rates.py`` - available without a cluster run.
    """
    return pd.read_pickle(_experiment(experiment)["net_params"]["fullscale_rates"])


def amyloid_burden(csv="abeta_ratio.csv"):
    """Per-area amyloid burden from a PET-derived CSV, as a Series sorted descending.

    The 'default' row (the fallback for areas the scan does not cover) is dropped.
    """
    path = MODEL_ROOT / "experimental_data" / "abeta" / csv
    table = pd.read_csv(path)
    return (table[table["area"] != "default"]
            .set_index("area")["ratio"].sort_values(ascending=False))


# ------------------------------------------------- checking your own data files --
def dk_areas():
    """The 34 Desikan-Killiany area names, in the order the model's asserts demand.

    Taken from the atlas volume file, which is what ``connectivity.py`` compares
    a connectome's own ``DK_Names`` against.
    """
    from scipy.io import loadmat

    path = MODEL_ROOT / "experimental_data" / "hcp_dti" / "DKAtlas_VolumeAndNames.mat"
    names = loadmat(path)["HumanDKNames"]
    import numpy as np
    return [str(s).split("-")[-1] for s in np.concatenate(names.flatten()).tolist()]


def check_data(path):
    """Report whether one of your own data files matches what the model expects.

    Dispatches on the suffix: ``.mat`` as a structural connectome, ``.npy`` as a
    BOLD / functional-connectivity reference, ``.csv`` as a pathology map. Prints
    what was found and flags every mismatch; returns True if nothing was flagged.
    """
    import numpy as np

    path = Path(path)
    print(f"{path.name}")
    problems = []

    if path.suffix == ".mat":
        from scipy.io import loadmat

        data = loadmat(path)
        required = ["DK_Names", "C_R", "C_L", "Dist",
                    "LowResDK_EstimatedDist_RHem", "LowResDK_EstimatedDist_LHem"]
        missing = [k for k in required if k not in data]
        print(f"  variables : {sorted(k for k in data if not k.startswith('__'))}")
        if missing:
            problems.append(f"missing variable(s): {', '.join(missing)}")
        for key in (k for k in required[1:] if k in data):
            if data[key].shape != (34, 34):
                problems.append(f"{key} is {data[key].shape}, expected (34, 34)")
        atlas = dk_areas()
        if "DK_Names" in data:
            names = [str(s).split("-")[-1]
                     for s in np.concatenate(data["DK_Names"].flatten()).tolist()]
            print(f"  areas     : {len(names)}  first: {names[0]}  last: {names[-1]}")
            if len(names) != len(atlas):
                problems.append(f"DK_Names has {len(names)} entries, expected 34")
            elif names != atlas:
                i, got, want = next((i, n, a) for i, (n, a) in enumerate(zip(names, atlas))
                                    if n != a)
                problems.append(f"DK_Names must match the atlas exactly, in order; "
                                f"position {i} is {got!r}, expected {want!r}")
        if data.get("C_R", np.empty(0)).shape == (34, 34):
            nos = data["C_R"]
            off = ~np.eye(34, dtype=bool)
            print(f"  C_R       : {int((nos[off] > 0).sum())} of 1122 off-diagonal "
                  f"entries non-zero, max {nos.max():.0f} (the model reads C_R, "
                  f"the right-hemisphere variable)")
            empty = [atlas[i] for i in range(34) if nos[i][off[i]].sum() == 0]
            if empty:
                problems.append(f"area(s) with no outgoing connection: {', '.join(empty)}")
        if data.get("LowResDK_EstimatedDist_RHem", np.empty(0)).shape == (34, 34):
            d = data["LowResDK_EstimatedDist_RHem"]
            print(f"  fibre len : {d[d > 0].min():.0f} - {d.max():.0f} mm "
                  f"(becomes the inter-area delays)")

    elif path.suffix == ".npy":
        data = np.load(path)
        print(f"  shape     : {data.shape}")
        nodes = data.shape[1] if data.ndim >= 2 else None
        if nodes not in (68, 84):
            problems.append(f"{nodes} nodes; the analysis accepts 68 (cortex only) "
                            "or 84 (full DK)")
        elif data.ndim == 2 and data.shape[0] == data.shape[1]:
            print("  read as   : a correlation matrix, lh 0:34 / rh 34:68"
                  if nodes == 68 else
                  "  read as   : a correlation matrix, lh 0:34 / rh 49:83")
        else:
            frames = data.shape[0]
            people = data.shape[2] if data.ndim == 3 else 1
            print(f"  read as   : BOLD time series, {frames} frames x {nodes} areas "
                  f"x {people} participant(s)")
            if frames < 100:
                problems.append(f"only {frames} frames; a noisy correlation reference")

    elif path.suffix == ".csv":
        table = pd.read_csv(path)
        print(f"  columns   : {list(table.columns)}")
        value_col = next((c for c in table.columns
                          if c.lower() in ("ratio", "abeta_ratio", "fraction", "value")), None)
        if "area" not in table.columns:
            problems.append("no 'area' column")
        if value_col is None:
            problems.append("no value column (name it ratio, fraction or value)")
        if "area" in table.columns and value_col:
            has_default = (table["area"] == "default").any()
            named = table.loc[table["area"] != "default", "area"]
            unknown = sorted(set(named) - set(dk_areas()))
            v = table[value_col]
            print(f"  areas     : {named.nunique()} named"
                  f"{' + a default row' if has_default else ''}, "
                  f"{value_col} in [{v.min():.3f}, {v.max():.3f}]")
            if "layer" in table.columns and table["layer"].notna().any():
                print(f"  layers    : per-layer rows for "
                      f"{table.loc[table['layer'].notna(), 'area'].nunique()} area(s)")
            if not has_default:
                print("  note      : no 'default' row - unlisted areas get 0.0")
            if unknown:
                problems.append(f"area name(s) not in the DK atlas: {', '.join(unknown)}")
            if not v.between(0, 1).all():
                problems.append(f"{value_col} outside [0, 1]")
    else:
        problems.append(f"unsupported suffix {path.suffix}")

    for p in problems:
        print(f"  PROBLEM   : {p}")
    if not problems:
        print("  -> usable as it stands")
    return not problems


# ------------------------------------------------------------- seed replication --
def _experiment(experiment):
    """The three dictionaries of ``experiments/<experiment>.py``."""
    cwd = os.getcwd()
    os.chdir(MODEL_ROOT)          # as the stages see it: paths built on the cwd
    try:
        return runpy.run_path(str(EXPERIMENTS / f"{experiment}.py"))
    finally:
        os.chdir(cwd)


def master_seed(experiment):
    """The master seed an experiment runs with (its own, or the model default)."""
    seed = _experiment(experiment).get("sim_params", {}).get("master_seed")
    if seed is None:
        defaults = runpy.run_path(str(MODEL_ROOT / "src" / "default_sim_params.py"))
        seed = defaults["params"]["master_seed"]
    return seed


def rates_for_seed(experiment, seed, t_start=1000.0, t_stop=5000.0, threads=4):
    """Population rates of `experiment` run with `seed` - network and simulation.

    The experiment's own seed returns the finished run's ``rates.pkl``.  Any other
    seed is a run of its own, ``<experiment>_seed<seed>``, started from scratch
    like every other: its network is built again rather than borrowed, and the
    rates are read straight from the spike files (the analysis stage is skipped).
    """
    if seed == master_seed(experiment):
        return rates(experiment)

    from eta_ext_search import read_population_rates      # tutorial's own helper

    run = run_all(experiment, stages=("network", "simulate"), threads=threads,
                  quiet=True, seed=seed)
    return read_population_rates(sim_dir(run), t_start, t_stop)


# ---------------------------------------------------------------- file excerpts --
def print_file(path):
    """Print a whole file (experiment files are meant to be read)."""
    print(Path(path).read_text())


def print_excerpt(path, marker=None, before=0, after=1500, label=None):
    """Print the part of a file around `marker`, then say how long the file is.

    `before` / `after` are character counts, snapped outward to whole lines so the
    excerpt never starts or ends mid-word; `label` replaces the marker itself in
    the printed text (useful when the marker is a long rule of dashes).
    """
    path = Path(path)
    text = path.read_text()
    if marker is None:
        chunk = text if after is None else text[:after]
    else:
        i = text.index(marker)
        start, end = max(0, i - before), i + len(marker) + after
        if start:                       # begin at a line start
            start = text.find('\n', start) + 1
        if end < len(text):             # end at a line end
            end = text.rfind('\n', start, end)
        shown_marker = marker if label is None else label
        chunk = text[start:i] + shown_marker + text[i + len(marker):end]
    print(chunk)
    if len(chunk) < len(text):
        print(f"\n... {len(text.splitlines())} lines in total: {path}")
