#!/usr/bin/env python3
"""Plumbing for the tutorial notebook: paths, stage running, hashes, output reading.

None of this is part of the model.  It is the bookkeeping that sits between the
notebook and the three pipeline stages:

* where the tutorial, the bundled model and the results live,
* how a stage is invoked (``subprocess`` into ``model/src/snakemake_*.py``),
* how the ``.network_hash`` / ``.simulation_hash`` / ``.analysis_hash`` files are
  read back and turned into the directory a run wrote to,
* how a figure or a result file of a finished run is loaded.

The notebook imports from here so that its cells show the *model* - parameters,
rates, pathology - rather than path juggling.  Everything is importable outside
the notebook as well::

    import sys; sys.path.insert(0, "tools")
    from tutorial_pipeline import run_all, rates

Nothing here writes into ``model/``: every output goes to ``results/``.
"""

import ast
import os
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
MODEL_ROOT = TUTORIAL / "model"

STAGES = ("network", "simulate", "analyse")

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

    (RESULTS / "log").mkdir(parents=True, exist_ok=True)

    import nest                                 # fails early if NEST is missing

    if verbose:
        print(f"bundled model   : {MODEL_ROOT}")
        print(f"Results go to   : {RESULTS}")
        print(f"NEST            : {nest.__version__}")
        print(f"numpy / pandas  : {np.__version__} / {pd.__version__}")
        print(f"CPU cores       : {os.cpu_count()}")
    return nest.__version__


# ------------------------------------------------------------- running a stage --
def run_stage(stage, experiment, threads=4, quiet=False):
    """Run one pipeline stage for `experiment`, with everything outside the clone.

    Each stage is the same command the Snakefile issues on a cluster; only the
    hash files are passed between them.
    """
    exp = EXPERIMENTS / f"{experiment}.py"
    base = EXPERIMENTS / experiment
    log = RESULTS / "log" / f"{experiment}_{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = {
        "network":  ["python", "src/snakemake_network.py", str(exp),
                     f"{base}.network_hash", str(log)],
        "simulate": ["python", "src/snakemake_simulation.py", str(exp),
                     f"{base}.network_hash", f"{base}.simulation_hash",
                     str(threads), str(log)],
        "analyse":  ["python", "src/snakemake_analysis.py", str(exp),
                     f"{base}.network_hash", f"{base}.simulation_hash",
                     f"{base}.analysis_hash", str(MODEL_ROOT), str(log)],
    }[stage]

    t0 = time.time()
    # cwd = the model repository: it resolves its data paths from the cwd.
    proc = subprocess.run(cmd, cwd=MODEL_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:])
        raise RuntimeError(f"stage '{stage}' failed for {experiment}")
    if not quiet:
        print(f"  {stage:9s} {elapsed:6.1f} s")
    return elapsed


def run_all(experiment, stages=STAGES, threads=4, quiet=False, title=None):
    """Run the three stages in order and print a per-stage timing table."""
    if title is None:
        title = experiment
    if not quiet:
        print(title)
    total = sum(run_stage(stage, experiment, threads=threads, quiet=quiet)
                for stage in stages)
    if not quiet:
        print(f"  {'total':9s} {total:6.1f} s")
    return total


# ------------------------------------------------------------ hashes and paths --
def hashes(experiment):
    """The hashes written so far, as a dict."""
    base = EXPERIMENTS / experiment
    out = {}
    for stage in ("network", "simulation", "analysis"):
        f = Path(f"{base}.{stage}_hash")
        if f.exists():
            out[stage] = f.read_text().strip()
    return out


def print_hashes(experiment):
    """Show the hash of every stage that has run."""
    for stage, h in hashes(experiment).items():
        print(f"{stage:11s} hash  {h}")


def net_dir(experiment):
    """The network directory of a completed stage 1."""
    return RESULTS / hashes(experiment)["network"]


def sim_dir(experiment):
    """The simulation directory of a completed stage 2."""
    h = hashes(experiment)
    return RESULTS / h["network"] / h["simulation"]


def run_dir(experiment):
    """The analysis directory of a completed run."""
    h = hashes(experiment)
    return RESULTS / h["network"] / h["simulation"] / h["analysis"]


def describe_outputs(experiment):
    """Inventory of what the analysis stage wrote: figures, figure sets, result files."""
    d = run_dir(experiment)
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
def show(experiment, name, width=900):
    """Display one figure produced by the analysis stage."""
    from IPython.display import Image, display

    path = run_dir(experiment) / "plots" / name
    if not path.exists():
        print(f"not found: {name}")
        return
    display(Image(filename=str(path), width=width))


def rates(experiment):
    """Population firing rates (Hz), MultiIndex (area, layer, pop)."""
    return pd.read_pickle(sim_dir(experiment) / "rates.pkl")


def pop_level(series):
    """Name of the population level of a rate index ('E' / 'I' live there)."""
    return series.index.names[-1]


def area_mean_rate(experiment, pop="E"):
    """Mean rate per area for one population, as a Series indexed by area."""
    r = rates(experiment)
    return r.xs(pop, level=pop_level(r)).groupby(level="area").mean()


def load_result(experiment, name):
    """Load one pickled result file of the analysis stage, e.g. 'lfp_proxy_kernel.pkl'."""
    return pd.read_pickle(run_dir(experiment) / name)


# ------------------------------------------------------------- seed replication --
def master_seed(experiment):
    """The master seed an experiment file declares."""
    text = (EXPERIMENTS / f"{experiment}.py").read_text()
    for line in text.splitlines():
        if '"master_seed"' in line:
            return ast.literal_eval(line.split(":", 1)[1].strip().rstrip(","))
    raise ValueError(f"no master_seed in {experiment}.py")


def seed_variant(experiment, seed):
    """An experiment identical to `experiment` but with another seed.

    ``net_params`` is untouched, so the network hash is unchanged: the copied
    ``.network_hash`` lets stage 1 be skipped entirely.
    """
    name = f"{experiment}_seed{seed}"
    base_seed = master_seed(experiment)
    text = (EXPERIMENTS / f"{experiment}.py").read_text()
    (EXPERIMENTS / f"{name}.py").write_text(
        text.replace(f'"master_seed": {base_seed}', f'"master_seed": {seed}'))
    (EXPERIMENTS / f"{name}.network_hash").write_text(
        (EXPERIMENTS / f"{experiment}.network_hash").read_text())
    return name


def rates_for_seed(experiment, seed, t_start=1000.0, t_stop=5000.0, threads=4):
    """Population rates of `experiment` run with `seed` - simulation stage only.

    The experiment's own seed returns the finished run's ``rates.pkl``; any other
    seed writes a variant experiment, simulates it, and reads the rates straight
    from the spike files (the analysis stage is skipped, as in ``ana_params``).
    """
    if seed == master_seed(experiment):
        return rates(experiment)

    from eta_ext_search import read_population_rates      # tutorial's own helper

    name = seed_variant(experiment, seed)
    run_stage("simulate", name, threads=threads, quiet=True)
    return read_population_rates(sim_dir(name), t_start, t_stop)


# ---------------------------------------------------------------- file excerpts --
def print_file(path):
    """Print a whole file (experiment files are meant to be read)."""
    print(Path(path).read_text())


def print_excerpt(path, marker=None, before=0, after=1500, label=None):
    """Print the part of a file around `marker`, then say how long the file is.

    `before` / `after` are character counts; `label` replaces the marker itself
    in the printed text (useful when the marker is a long rule of dashes).
    """
    path = Path(path)
    text = path.read_text()
    if marker is None:
        chunk = text if after is None else text[:after]
    else:
        i = text.index(marker)
        start, end = max(0, i - before), i + len(marker) + after
        shown_marker = marker if label is None else label
        chunk = text[start:i] + shown_marker + text[i + len(marker):end]
    print(chunk)
    if len(chunk) < len(text):
        print(f"\n... {len(text.splitlines())} lines in total: {path}")
