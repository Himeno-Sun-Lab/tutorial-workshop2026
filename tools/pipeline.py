#!/usr/bin/env python3
"""Plumbing for the tutorial notebook: paths, running a model, reading its output.

None of this is part of the model.  It is the bookkeeping that sits between the
notebook and the three pipeline stages:

* where the tutorial, the bundled model and the results live,
* how a run is started (``subprocess`` into ``model/src/snakemake_*.py``),
* how the hash files a run writes are turned into the directory it wrote to,
* how a figure or a result file of a finished run is loaded - a laptop run, or
  a full-scale run imported from the server it was computed on.

**No reuse on the laptop.**  Every ``run_all()`` starts from an empty
``results/<run>/`` and runs all of its stages again: nothing is skipped because
a hash matches, nothing is read from an earlier run's cache, and two runs never
share a directory.  A run costs minutes here, and a result you can trust to
come from the parameters on screen is worth more than the minutes.  Reuse - only
the stages whose parameters changed are re-run - belongs to the server's
Snakemake workflow, where a stage costs hours.

Layout of one laptop run::

    results/<run>/experiment.py            driver: experiments/<experiment>.py,
                                           with outpath pointed here
    results/<run>/{network,simulation,analysis}_hash
    results/<run>/log/<stage>.log
    results/<run>/<net_hash>/<sim_hash>/<ana_hash>/plots/ ...

A full-scale run is imported unchanged, in the layout the server's Snakemake
workflow wrote it - nothing is converted, so the hashes still name the
directories they were computed for::

    fullscale/experiments/<run>.py                     the experiment file
    fullscale/experiments/<run>.{network,simulation,analysis}_hash
    fullscale/out/<net_hash>/<sim_hash>/<ana_hash>/    the three levels
    fullscale/logs/<run>_*                             the stage logs
    fullscale/experimental_data/...                    the empirical BOLD it was
                                                       compared against, under
                                                       its server path

Every reader below - ``show``, ``rates``, ``load_result``, ``area_mean_rate`` -
accepts its name in place of a laptop run.  The notebook only reads it.

The notebook imports from here so that its cells show the *model* - parameters,
rates, pathology - rather than path juggling.  Everything is importable outside
the notebook as well::

    import sys; sys.path.insert(0, "tools")
    from pipeline import run_all, rates

Nothing here writes into ``model/``: every output goes to ``results/``.
"""

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
LEVELS = ("network", "simulation", "analysis")


def _imported_hash_file(run, level):
    return FULLSCALE / "experiments" / f"{run}.{level}_hash"


def has_fullscale(run):
    """Whether the full-scale run `run` has been imported into fullscale/."""
    return all(_imported_hash_file(run, level).exists() for level in LEVELS)


def _store(run):
    """Where `run` lives: its hash file per level, and the directory they index."""
    if has_fullscale(run):
        return ({level: _imported_hash_file(run, level) for level in LEVELS},
                FULLSCALE / "out")
    return ({level: RESULTS / run / f"{level}_hash" for level in LEVELS},
            RESULTS / run)


def hashes(run):
    """The hashes the run has written so far, as a dict."""
    files, _ = _store(run)
    return {level: f.read_text().strip() for level, f in files.items() if f.exists()}


def stage_hash(run, stage):
    """The hash one stage wrote, or '-' if its file is not there."""
    f = RESULTS / run / STAGE_HASH_FILE[stage]
    return f.read_text().strip() if f.exists() else "-"


def print_hashes(run):
    """Show the hash of every stage that has run."""
    for stage, h in hashes(run).items():
        print(f"{stage:11s} hash  {h}")


def _missing(run):
    if (EXPERIMENTS / f"{run}.py").exists():
        return f"no completed run '{run}' - run_all({run!r}) first"
    return (f"no run '{run}': not a laptop run, and no full-scale run of that name "
            f"in {FULLSCALE / 'experiments'}")


def _hashes_for(run, *levels):
    h = hashes(run)
    if not all(level in h for level in levels):
        raise FileNotFoundError(_missing(run))
    return [h[level] for level in levels]


def net_dir(run):
    """The network directory of a completed stage 1."""
    (nh,) = _hashes_for(run, "network")
    return _store(run)[1] / nh


def sim_dir(run):
    """The simulation directory of a completed stage 2."""
    nh, sh = _hashes_for(run, "network", "simulation")
    return _store(run)[1] / nh / sh


def run_dir(run):
    """The analysis directory of a completed run."""
    nh, sh, ah = _hashes_for(run, "network", "simulation", "analysis")
    return _store(run)[1] / nh / sh / ah


def fullscale_info(run):
    """What an imported full-scale run was computed with, as a Series.

    Read from the parameter dumps the model wrote next to its output
    (``network_params.yaml``, ``sim.yaml``) and from the network's own tables -
    not from the experiment file, whose paths point at the server.
    """
    import yaml

    if not has_fullscale(run):
        raise FileNotFoundError(_missing(run))
    net = yaml.unsafe_load((net_dir(run) / "network_params.yaml").read_text())
    sim = yaml.unsafe_load((sim_dir(run) / "sim.yaml").read_text())
    gains = net["scaling_factors_recurrent"]
    types = list(net.get("neuron_types") or {})
    if not types and net.get("Abeta"):      # dumped before neuron_types existed
        types = ["Abeta"]
    return pd.Series({
        "N_scaling": net["N_scaling"],
        "K_scaling": net["K_scaling"],
        "scaling_type": net["scaling_type"],
        "neurons": int(pd.read_pickle(net_dir(run) / "neuron_numbers.pkl").sum()),
        "internal synapses": f"{pd.read_pickle(net_dir(run) / 'synapses_internal.pkl').values.sum():.3g}",
        "eta_ext": net["input_params"]["eta_ext"],
        "cc_scalingEtoE / EtoI": f"{gains['cc_scalingEtoE']} / {gains['cc_scalingEtoI']}",
        "neuron types": ", ".join(types) or "none",
        "connectome": Path(net["predictive_connectomic_params"]["con_path"]).name,
        "t_sim (ms)": sim["t_sim"],
        "master_seed": sim["master_seed"],
        **{f"{level} hash": h for level, h in hashes(run).items()},
    }, name=run)


def fullscale_fc(run, signal="synaptic"):
    """Simulated FC and the person's empirical FC of an imported run.

    Returns ``(simulated, empirical)``, two 34x34 DataFrames over the same
    alphabetically sorted areas - the matrices behind ``exp__sim_syn__corr_rh``
    (``signal="synaptic"``) or ``exp__sim_bold__corr_rh`` (``signal="bold"``) in
    ``correlations.txt``.  Both are computed the way ``analysis.py`` does it.

    * synaptic: correlation of the area-averaged synaptic input currents
      (``input_current.pkl``) from ``plotConnectivities.tmin`` on;
    * bold: the stage-2 cache ``bold_correlation.pkl``;
    * empirical: correlation of the BOLD time series the run was compared
      against.  ``ana.yaml`` names it by its server path; it is imported under
      the same path below ``experimental_data/``, i.e.
      ``fullscale/experimental_data/...``.  Its right half (nodes 34:68) is the
      simulated hemisphere; the nodes are labelled with the FreeSurfer order of
      the model's own ``ROIs.txt``, as ``analysis.py`` does.
    """
    import numpy as np
    import yaml

    if not has_fullscale(run):
        raise FileNotFoundError(_missing(run))
    ana = yaml.unsafe_load((run_dir(run) / "ana.yaml").read_text())
    remote = Path(ana["experimental_fc"])
    local = FULLSCALE / Path(*remote.parts[remote.parts.index("experimental_data"):])
    if not local.exists():
        raise FileNotFoundError(f"the empirical BOLD of {run} is not imported: {local}")

    roi_file = MODEL_ROOT / "experimental_data" / "senden" / "rsData_7T_DKparcel" / "ROIs.txt"
    areas = [line.strip().split("-")[-1] for line in roi_file.read_text().splitlines()[14:48]]
    bold = np.load(local)[:, 34:68, 0]
    empirical = pd.DataFrame(bold, columns=areas).corr()

    if signal == "synaptic":
        currents = pd.read_pickle(run_dir(run) / "input_current.pkl").T   # time x area
        simulated = currents[currents.index >= ana["plotConnectivities"]["tmin"]].corr()
    elif signal == "bold":
        simulated = pd.read_pickle(sim_dir(run) / "bold_correlation.pkl")
    else:
        raise ValueError(f"signal must be 'synaptic' or 'bold', not {signal!r}")
    return (simulated.sort_index().sort_index(axis=1),
            empirical.sort_index().sort_index(axis=1))


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
                          if c.lower() in ("ratio", "fraction", "value")), None)
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


def rates_for_seed(experiment, seed, t_start=None, t_stop=None, threads=4):
    """Population rates of `experiment` run with `seed` - network and simulation.

    The experiment's own seed returns the finished run's ``rates.pkl``.  Any other
    seed is a run of its own, ``<experiment>_seed<seed>``, started from scratch
    like every other: its network is built again rather than borrowed, and the
    rates are read straight from the spike files (the analysis stage is skipped).

    The window defaults to the one ``rates.pkl`` is computed over - from the
    experiment's ``ana_params["meanFiringRate"]["t_start"]`` to its ``t_sim`` - so
    that the rates of every seed are comparable.
    """
    if seed == master_seed(experiment):
        return rates(experiment)

    from eta_ext_search import read_population_rates      # tutorial's own helper

    exp = _experiment(experiment)
    if t_start is None:
        defaults = runpy.run_path(str(MODEL_ROOT / "src" / "default_ana_params.py"))["params"]
        t_start = exp.get("ana_params", {}).get("meanFiringRate", {}).get(
            "t_start", defaults["meanFiringRate"]["t_start"])
    if t_stop is None:
        defaults = runpy.run_path(str(MODEL_ROOT / "src" / "default_sim_params.py"))["params"]
        t_stop = exp.get("sim_params", {}).get("t_sim", defaults["t_sim"])

    run = run_all(experiment, stages=("network", "simulate"), threads=threads,
                  quiet=True, seed=seed)
    return read_population_rates(sim_dir(run), t_start, t_stop)


# ------------------------------------------------------ working-point calibration --
# The two tools of the calibration, tools/eta_ext_search.py and
# tools/fullscale_rates.py, are command-line programs. These wrappers read the
# settings they need from an experiment file and write everything under
# results/calibration/<experiment>/ - never into experiments/, so a calibration
# run cannot overwrite the rates an experiment ships with.
CALIBRATION = RESULTS / "calibration"


def _run_tool(script, args, log, show):
    """Run ``tools/<script>`` and stream the lines `show` returns text for."""
    cmd = [sys.executable, str(TUTORIAL / "tools" / script), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(log, "w") as fh, subprocess.Popen(
            cmd, cwd=TUTORIAL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env) as proc:
        for line in proc.stdout:
            fh.write(line)
            text = show(line)
            if text:
                print(text, flush=True)
    if proc.returncode != 0:
        print(Path(log).read_text()[-3000:])
        raise RuntimeError(f"{script} failed - log: {log}")
    return time.time() - t0


def _net_params(experiment):
    """``(net_params, defaults)``: an experiment's top-level network parameters,
    completed from the model defaults, and the defaults themselves."""
    cwd = os.getcwd()
    os.chdir(MODEL_ROOT)          # the defaults build their data paths on the cwd
    try:
        defaults = runpy.run_path(str(MODEL_ROOT / "src" / "default_net_params.py"))["params"]
    finally:
        os.chdir(cwd)
    return {**defaults, **_experiment(experiment)["net_params"]}, defaults


def eta_search(experiment, etas, seeds=None, t_sim=2000.0, threads=4):
    """Mean E / I rates of `experiment` for every external drive in `etas`.

    Runs ``tools/eta_ext_search.py --method scan``: for every ``eta_ext`` a
    network is built with the experiment's scale, long-range gains, connectome
    and full-scale rates, simulated for `t_sim` ms (the first 500 ms discarded)
    and its population rates counted.

    Only a healthy experiment can be scanned.  One with neuron types (e.g. the
    amyloid experiment) raises: it keeps the drive of its healthy control, and
    scanning it would invite pulling it back to the healthy working point.
    Changed neuron parameters cannot be passed to the tool either; they are
    reported, since the scan then runs the default neurons.

    Returns a DataFrame with one row per ``eta_ext`` (mean rates in Hz, drive
    as a fraction of the distance to threshold, ``alive``).
    """
    net, defaults = _net_params(experiment)
    if net.get("neuron_types"):
        raise ValueError(
            f"{experiment} declares neuron types ({', '.join(net['neuron_types'])}). "
            "A pathological experiment keeps the drive of its healthy control - "
            "recalibrating it would pull it back to the healthy working point. "
            "Scan the healthy control instead (e.g. tutorial_CN).")
    out = CALIBRATION / experiment / "eta_search"
    out.mkdir(parents=True, exist_ok=True)
    gains = net["scaling_factors_recurrent"]
    args = ["--method", "scan", "--eta-list", *etas,
            "--n-scaling", net["N_scaling"], "--k-scaling", net["K_scaling"],
            "--scaling-type", net["scaling_type"],
            "--cc-e-to-e", gains.get("cc_scalingEtoE", 1.0),
            "--cc-e-to-i", gains.get("cc_scalingEtoI", 2.0),
            "--seeds", *(seeds or [master_seed(experiment)]),
            "--t-sim", t_sim, "--threads", threads,
            "--outdir", out, "--log-level", "WARNING", "--no-plot"]
    if net.get("fullscale_rates"):
        args += ["--fullscale-rates", net["fullscale_rates"]]
    con = net["predictive_connectomic_params"].get("con_path")
    if con and con != defaults["predictive_connectomic_params"]["con_path"]:
        args += ["--con-path", con]          # an individual connectome

    ignored = [k for k in ("neuron_params_E", "neuron_params_I",
                           "neuron_model_E", "neuron_model_I")
               if net[k] != defaults[k]]
    print(f"eta_ext scan of {experiment}: N_scaling={net['N_scaling']}, "
          f"K_scaling={net['K_scaling']}, {len(etas)} values x {t_sim:.0f} ms")
    if ignored:
        print(f"  WARNING: {experiment} sets {', '.join(ignored)}, which the scan cannot\n"
              f"  pass on: it runs the default neurons, with {experiment}'s full-scale\n"
              f"  rates - the rates below do not belong to {experiment}.")

    def show(line):
        line = line.rstrip()
        return f"  {line.strip()}" if line.strip().startswith("eta=") else None

    elapsed = _run_tool("eta_ext_search.py", args, out / "eta_search.log", show)
    table = pd.read_csv(out / "eta_search.csv")
    print(f"  done in {elapsed:.0f} s - {out.relative_to(TUTORIAL)}/")
    return table[["eta", "E", "I", "drive_fraction", "alive"]].rename(
        columns={"E": "E (Hz)", "I": "I (Hz)", "drive_fraction": "drive / threshold"})


def fullscale_rates(experiment, eta=None, iters=40, threads=8):
    """Self-consistent full-scale rates of `experiment` (optionally at `eta`).

    Runs ``tools/fullscale_rates.py``: starting from the mean-field rates of the
    default model, it iterates rates -> DC compensation -> simulate -> rates with
    damping until the downscaled network reproduces the rates its DC current was
    computed from.  Prints one line per iteration; the last columns, the mean
    change per step, must have levelled off at the finite-size noise.

    Returns ``(rates, path)``: the population rates (Hz) and the pickle they
    were written to, ready for ``net_params["fullscale_rates"]``.
    """
    import json

    tag = "" if eta is None else f"_eta{float(eta):g}"
    out = CALIBRATION / experiment / f"fullscale_rates{tag}.pkl"
    work = CALIBRATION / experiment / f"fullscale_rates{tag}"
    work.mkdir(parents=True, exist_ok=True)
    args = [EXPERIMENTS / f"{experiment}.py", "--out", out, "--iters", iters,
            "--threads", threads, "--workdir", work]
    if eta is not None:
        args += ["--eta", eta]
    print(f"full-scale rates of {experiment}"
          f"{'' if eta is None else f' at eta_ext = {eta}'}: {iters} iterations")
    print("    it    E in   I in   E sim   I sim   mean |step| E / I  (Hz)")

    def show(line):
        if not line.startswith("{"):
            return None
        r = json.loads(line)
        return (f"  {r['it']:4d}  {r['E_in']:6.2f} {r['I_in']:6.2f}  "
                f"{r['E_sim']:6.2f} {r['I_sim']:6.2f}   {r['dE']:6.2f} / {r['dI']:.2f}")

    elapsed = _run_tool("fullscale_rates.py", args, work / "fullscale_rates.log", show)
    rates_ = pd.read_pickle(out)
    level = rates_.index.get_level_values(-1)
    print(f"  done in {elapsed / 60:.1f} min: E {rates_[level == 'E'].mean():.2f} Hz, "
          f"I {rates_[level == 'I'].mean():.2f} Hz - {out.relative_to(TUTORIAL)}")
    return rates_, out


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
