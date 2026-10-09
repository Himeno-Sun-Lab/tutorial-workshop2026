#!/usr/bin/env python3
"""Find the external drive (``eta_ext``) that sets a DOWNSCALED network's working point.

Why this exists
---------------
Downscaling multiplies in-degrees by ``K_scaling`` and weights by
``1/sqrt(K_scaling)``.  That preserves the *variance* of the synaptic input but
shrinks its *mean* by ``sqrt(K_scaling)``.  ``Network.extraDCforScaledNetwork()``
adds back the missing ``(1-sqrt(k))*mu`` as a DC current (van Albada et al.
2015), so the downscaled network approximates the full-scale network with the
same parameters -- provided ``fullscale_rates`` are the rates of *that*
full-scale network.  The model's ``base_theory_rates.pkl`` matches only the
repository defaults; for any other configuration generate matching rates with
``tools/fullscale_rates.py`` and pass them with ``--fullscale-rates``.  The
rates depend on ``eta_ext`` itself, so a final candidate must be confirmed with
rates computed at that candidate (see the README).

This script searches for the ``eta_ext`` that (a) wakes the network up and
(b) puts it in the rate band the project targets elsewhere
(``src/roi_parameter_search.py``): area-averaged excitatory rates 3-5 Hz,
inhibitory rates 5-15 Hz.

Three evaluation methods, cheapest first
----------------------------------------
``analytic``  Closed-form mean drive.  For an iaf_psc_exp neuron the stationary
              mean depolarisation is
                  dV = (tau_m/C_m) * (w_ext*tau_syn*K_ext*nu_ext + I_e)
              and it is *affine* in eta_ext (nu_ext is proportional to it, and
              so is the external part of I_e).  Two network builds therefore
              determine dV(eta) exactly, and the eta that restores any target
              fraction of threshold follows in closed form.  No simulation.
              Cost: ~2 network builds.  Does NOT predict firing rates.

``theory``    Mean-field (NNMT) self-consistent rates, with the compensating DC
              current added to the mean input -- ``src/theory/rates.py`` omits
              it, which matters precisely for downscaled networks.  Predicts
              rates without simulating.  Cost: ~5 s per eta.  It is a diffusion
              approximation that ignores weight and delay distributions and all
              correlations; with the earlier, incorrect DC compensation it
              under-predicted simulated rates by ~20-25% at
              ``N_scaling=0.005, K_scaling=0.1`` (not re-measured since).  Use
              it to bracket, then verify with ``--method scan``.

``simulate``  Build the network, run NEST, count spikes.  Ground truth.
              Cost: ~12 s per eta at N_scaling=0.005 on 4 threads.

A caveat for all rate-based searching: recurrent inhibition tracks the
external drive, so rates rise slowly with eta (measured at N_scaling=0.005,
K_scaling=0.1 with self-consistent full-scale rates: E 0.3 Hz at eta 1.1,
3.0 Hz at 4.5, 5.1 Hz at 7.0).  Seed-to-seed differences of ~0.2 Hz are then
comparable to the effect of a 0.2 change in eta, so pass several ``--seeds``
if you need the boundary to a couple of decimals.

Where this lives
----------------
The model ships with this tutorial, in ``../model``; ``--model-root`` overrides
that if you keep it elsewhere.  Network dumps, spike files and results all go to
``--outdir`` (default ``../results/eta_search/``), never into the model folder.

Examples
--------
    # default: simulate, bisect for the 3-5 Hz excitatory band
    python tools/eta_ext_search.py

    # instant estimate only
    python tools/eta_ext_search.py --method analytic

    # fast mean-field sweep, then verify the winner by simulation
    python tools/eta_ext_search.py --method theory
    python tools/eta_ext_search.py --method scan --eta-list 3.2 3.4 3.6 3.8

    # 8 GB laptop, noisier but quicker
    python tools/eta_ext_search.py --n-scaling 0.002 --t-sim 1500

    # calibrate for one individual: their connectome and their amyloid map
    python tools/eta_ext_search.py \
        --con-path   ../model/experimental_data/ADNI/AD_SC.mat \
        --abeta-ratio ../model/experimental_data/ADNI/AD_abeta_ratio.csv

    # average three seeds per candidate (slower, much less noisy)
    python tools/eta_ext_search.py --seeds 2903 1234 5678

Outputs (under ``--outdir``)
----------------------------
    eta_search.csv     every evaluation, one row per (eta, seed)
    eta_search.json    the chosen eta values and the full configuration
    eta_search.png     rate vs eta with the target bands (unless --no-plot)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import warnings
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# The model's modules are imported in bootstrap(), once the repository has been
# located -- this script lives outside the repository and must never write into
# it. Names bound there (Network, Simulation, ...) are module globals by the
# time any function below runs.
MODEL_ROOT = None


def find_model_repo(explicit=None):
    """The model bundled with this tutorial, or --model-root / $MODEL_ROOT."""
    bundled = HERE.parent / "model"
    for cand in (explicit, os.environ.get("MODEL_ROOT")):
        if cand:
            if (Path(cand) / "src" / "snakemake_network.py").exists():
                return Path(cand).resolve()
            raise SystemExit(f"{cand} does not contain the model source.")
    if (bundled / "src" / "snakemake_network.py").exists():
        return bundled.resolve()
    raise SystemExit(
        f"The bundled model is missing from {bundled}.\n"
        "Re-clone the tutorial repository, or pass --model-root <path>.")


def bootstrap(root):
    """Put the repository on sys.path and import the modules this script needs.

    ``src/`` itself goes on the path (the repository's modules import each other
    by bare name), and the working directory becomes the repository root
    because several data paths are built from ``os.getcwd()``.
    """
    global MODEL_ROOT
    MODEL_ROOT = Path(root).resolve()
    src = str(MODEL_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    os.chdir(MODEL_ROOT)

    import default_net_params
    import default_sim_params
    from data_preprocessing.cytoarchitecture import NeuronNumbers
    from data_preprocessing.connectivity import SynapseNumbers
    from network import Network
    from simulation import Simulation
    from helpers.snakemake import nested_dict_update, build_loggers

    g = globals()
    g["default_net_params"] = default_net_params
    g["default_sim_params"] = default_sim_params
    g["NeuronNumbers"] = NeuronNumbers
    g["SynapseNumbers"] = SynapseNumbers
    g["Network"] = Network
    g["Simulation"] = Simulation
    g["nested_dict_update"] = nested_dict_update
    g["build_loggers"] = build_loggers
    return MODEL_ROOT


# --------------------------------------------------------------------------- #
#  Amyloid neuron parameters (only used when --abeta-ratio is given)
# --------------------------------------------------------------------------- #
ABETA_E = dict(E_L=-65.0, V_th=-50.0, V_reset=-70.0, C_m=220.0,
               tau_m=10.0, tau_syn_ex=2.0, tau_syn_in=2.0, t_ref=2.0)
ABETA_I = dict(E_L=-70.0, V_th=-45.0, V_reset=-70.0, C_m=100.0,
               tau_m=10.0, tau_syn_ex=2.0, tau_syn_in=2.0, t_ref=2.0)
NO_SPREAD = {p: {"distribution": "lognormal", "rel_sd": 0.0}
             for p in ("V_th", "C_m", "tau_m")}


# --------------------------------------------------------------------------- #
#  Network construction
# --------------------------------------------------------------------------- #
class NetworkFactory:
    """Builds the expensive anatomy once and cheap networks per ``eta_ext``."""

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self._cache: dict[float, Network] = {}

        base = deepcopy(default_net_params.params)
        t0 = time.time()
        print("building cytoarchitecture and connectome (one-off, ~60 s) ...",
              flush=True)
        self.NN = NeuronNumbers(surface_area=base["surface_area"],
                                logger=logger,
                                **base["cytoarchitecture_params"])
        self.SN = SynapseNumbers(NN=self.NN, logger=logger,
                                 **base["predictive_connectomic_params"])
        print(f"  done in {time.time() - t0:.0f} s", flush=True)

    def overrides(self, eta):
        args = self.args
        o = {
            "outpath": str(args.outdir / "networks"),
            "mode": args.log_level,
            "N_scaling": args.n_scaling,
            "K_scaling": args.k_scaling,
            "scaling_type": args.scaling_type,
            "fullscale_rates": str(args.fullscale_rates),
            "scaling_factors_recurrent": {"cc_scalingEtoE": args.cc_e_to_e,
                                          "cc_scalingEtoI": args.cc_e_to_i},
            "input_params": {"eta_ext": float(eta)},
            "Abeta": args.abeta_ratio is not None,
        }
        if args.con_path is not None:
            # An individual connectome replaces the group-average one; the file
            # must have the same variable layout (C_L, C_R, Dist, DK_Names).
            o["predictive_connectomic_params"] = {"con_path": str(args.con_path)}
        if args.abeta_ratio is not None:
            o.update({
                "Abeta_ratio_E": args.abeta_ratio,
                "Abeta_ratio_I": 0.0,
                "neuron_model_Abeta_E": "iaf_psc_exp",
                "neuron_model_Abeta_I": "iaf_psc_exp",
                "neuron_params_Abeta_E": ABETA_E,
                "neuron_params_Abeta_I": ABETA_I,
                "neuron_param_dist_Abeta_E": NO_SPREAD,
                "neuron_param_dist_Abeta_I": NO_SPREAD,
            })
        return o

    def build(self, eta):
        key = round(float(eta), 6)
        if key not in self._cache:
            params = deepcopy(default_net_params.params)
            nested_dict_update(params, self.overrides(key), self.logger)
            net = Network(self.NN, self.SN, params, logger=self.logger)
            net.dump(params["outpath"])
            self._cache[key] = net
        return self._cache[key]


# --------------------------------------------------------------------------- #
#  Method 1: closed-form mean drive
# --------------------------------------------------------------------------- #
def mean_drive(net_dict, pop):
    """Stationary mean depolarisation of ``pop`` and its distance to threshold.

    Returns (dV_mV, to_threshold_mV).  For iaf_psc_exp with current-based
    exponential synapses the mean input is
        dV = (tau_m/C_m) * (w_ext * tau_syn * K_ext * nu_ext * 1e-3 + I_e)
    with nu_ext in Hz, w_ext and I_e in pA, tau in ms, C_m in pF.
    """
    kind = pop[-1]
    npar = net_dict[f"neuron_params_{kind}"]
    tau_m, C_m, tau_syn = npar["tau_m"], npar["C_m"], npar["tau_syn_ex"]

    n = net_dict["neuron_numbers"].loc[pop]
    if n <= 0:
        raise ValueError(f"population {pop} is empty")
    k_ext = net_dict["synapses_external"].loc[pop] / n
    w_ext = net_dict["weights_ext"].loc[pop]
    nu_ext = net_dict["rate_ext"].loc[pop]
    i_dc = net_dict["dc_drive"].loc[pop]

    dv = (tau_m / C_m) * (w_ext * tau_syn * k_ext * nu_ext * 1e-3 + i_dc)
    return float(dv), float(npar["V_th"] - npar["E_L"])


def analytic_eta(factory, pop, target_fraction, eta_a=1.1, eta_b=3.0):
    """Solve the affine dV(eta) for the eta reaching ``target_fraction``.

    dV is affine in eta, so two evaluations pin it down exactly.
    """
    dv_a, thr = mean_drive(factory.build(eta_a).getNetwork(), pop)
    dv_b, _ = mean_drive(factory.build(eta_b).getNetwork(), pop)
    slope = (dv_b - dv_a) / (eta_b - eta_a)
    if slope <= 0:
        raise RuntimeError("mean drive does not increase with eta_ext; "
                           "check the network parameters")
    intercept = dv_a - slope * eta_a
    eta = (target_fraction * thr - intercept) / slope
    return dict(eta=float(eta), slope_mV_per_eta=float(slope),
                intercept_mV=float(intercept), to_threshold_mV=thr,
                fraction_at_1p1=float(dv_a / thr))


# --------------------------------------------------------------------------- #
#  Method 2: mean-field rates, with the compensating DC current included
# --------------------------------------------------------------------------- #
def theory_rates(net_dict, t_max=60.0, nu0=1.0):
    """Self-consistent NNMT mean-field rates including ``dc_drive``.

    ``src/theory/rates.py`` builds the mean input without the DC compensation,
    which is exactly the term that keeps a downscaled network alive, so the DC
    contribution (tau_m * I_e / C_m, in mV) is added here.
    """
    from scipy.integrate import solve_ivp
    from nnmt.lif.exp import _firing_rate_shift
    from nnmt.lif._general import _mean_input, _std_input
    from theory.rates import _build_inputs

    p = _build_inputs(net_dict)
    index = p["N"].index
    tau_m, t_ref, v_th, v_r = p["tau_m"], p["t_ref"], p["V_th"], p["V_r"]
    prm, prm_scaled = p["input_prms"], p["input_prms_scld"]

    c_m = pd.Series(0.0, index=index)
    c_m.loc[(slice(None), slice(None), "E")] = net_dict["neuron_params_E"]["C_m"]
    c_m.loc[(slice(None), slice(None), "I")] = net_dict["neuron_params_I"]["C_m"]
    dc = net_dict["dc_drive"].sort_index().reindex(index).fillna(0.0)
    mu_dc = ((tau_m * 1e3) * dc / c_m).values          # tau_m is in s -> ms

    def ode(_, nu):
        mu = _mean_input(nu=nu, **prm) + mu_dc
        sigma = _std_input(nu=nu, **prm)
        tau_eff = sigma ** 2 / _std_input(nu=nu, **prm_scaled) ** 2
        nu_out = _firing_rate_shift(V_0_rel=v_r, V_th_rel=v_th, mu=mu,
                                    sigma=sigma, tau_m=tau_m, tau_r=t_ref,
                                    tau_s=tau_eff)
        return -nu + nu_out

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sol = solve_ivp(ode, (0.0, t_max), np.full(len(index), float(nu0)),
                        method="LSODA", t_eval=[t_max])
    if not sol.success:
        raise RuntimeError(f"mean-field solver failed: {sol.message}")
    if caught and not getattr(theory_rates, "_warned", False):
        theory_rates._warned = True
        print(f"  [theory] NNMT raised {len(caught)} numerical warning(s) "
              f"(Taylor-expansion range); mean-field stays a bracket, not an "
              f"answer.", flush=True)
    return pd.Series(np.clip(sol.y[:, -1], 0.0, None), index=index)


# --------------------------------------------------------------------------- #
#  Method 3: simulation
# --------------------------------------------------------------------------- #
def read_population_rates(sim_folder, t_start, t_stop):
    """Population firing rates straight from the NEST ASCII output.

    Deliberately bypasses ``Analysis``: that class caches ``rates.pkl`` with the
    first ``t_start`` it ever sees, which silently returns stale numbers inside
    a search loop.
    """
    sim_folder = Path(sim_folder)
    gid_file = sim_folder / "population_GIDs.dat"
    pops, lo, hi = [], [], []
    with open(gid_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, a, b = line.rsplit(";", 2)
            pops.append(ast.literal_eval(key))   # "('area', 'layer', 'E')"
            lo.append(int(a))
            hi.append(int(b))

    order = np.argsort(lo)
    pops = [pops[i] for i in order]
    lo = np.asarray(lo)[order]
    hi = np.asarray(hi)[order]
    size = (hi - lo + 1).astype(float)

    gids, times = [], []
    for dat in sorted((sim_folder / "spikes").glob("*.dat")):
        arr = np.loadtxt(dat, skiprows=3, ndmin=2)
        if arr.size and arr.ndim == 2 and arr.shape[1] >= 2:
            gids.append(arr[:, 0].astype(np.int64))
            times.append(arr[:, 1])
    if gids:
        gids = np.concatenate(gids)
        times = np.concatenate(times)
        keep = (times >= t_start) & (times < t_stop)
        gids = gids[keep]
    else:
        gids = np.empty(0, dtype=np.int64)

    counts = np.zeros(len(pops))
    if gids.size:
        idx = np.searchsorted(lo, gids, side="right") - 1
        valid = (idx >= 0) & (gids <= hi[np.clip(idx, 0, len(hi) - 1)])
        np.add.at(counts, idx[valid], 1.0)

    duration_s = (t_stop - t_start) / 1000.0
    rates = pd.Series(counts / size / duration_s,
                      index=pd.MultiIndex.from_tuples(
                          pops, names=["area", "layer", "pop"]))
    return rates


def simulate_rates(net, args, seed, loggers):
    """Run one NEST simulation and return the population rate Series."""
    import nest

    # NEST's kernel prints a block of M_INFO messages per simulation; at this
    # call rate it buries the search output. Warnings and errors still show.
    nest.set_verbosity("M_WARNING")

    main_logger, rank_logger = loggers
    sim_params = deepcopy(default_sim_params.params)
    nested_dict_update(sim_params, {
        "t_sim": float(args.t_sim),
        "master_seed": int(seed),
        "rec_dev": ["spike_recorder"],
        "mode": args.log_level,
    }, main_logger)

    net_folder = Path(args.outdir) / "networks" / net.getHash()
    sim = Simulation(sim_params, net.getNetwork(), main_logger, rank_logger)
    sim.dump(str(net_folder))
    data_path = net_folder / sim.getHash()

    sim.setup(str(data_path), args.threads)
    sim.simulate()
    nest.ResetKernel()          # flushes the ASCII spike files -- do not skip

    return read_population_rates(data_path, args.t_transient, args.t_sim)


# --------------------------------------------------------------------------- #
#  Evaluation wrapper shared by every search strategy
# --------------------------------------------------------------------------- #
class Evaluator:
    def __init__(self, factory, args, loggers):
        self.factory = factory
        self.args = args
        self.loggers = loggers
        self.rows = []
        self._cache = {}

    def __call__(self, eta):
        key = round(float(eta), 6)
        if key in self._cache:
            return self._cache[key]

        net = self.factory.build(key)
        net_dict = net.getNetwork()
        dv, thr = mean_drive(net_dict, self.args.probe_pop)

        t0 = time.time()
        if self.args.eval_method == "theory":
            rates = theory_rates(net_dict)
            seeds_used = [None]
            per_seed = [split_ei(rates)]
        else:
            seeds_used = self.args.seeds
            per_seed = []
            for seed in seeds_used:
                rates = simulate_rates(net, self.args, seed, self.loggers)
                per_seed.append(split_ei(rates))
        elapsed = time.time() - t0

        e_vals = np.array([p["E"] for p in per_seed])
        i_vals = np.array([p["I"] for p in per_seed])
        result = dict(eta=key,
                      E=float(e_vals.mean()), I=float(i_vals.mean()),
                      E_sd=float(e_vals.std(ddof=1)) if len(e_vals) > 1 else 0.0,
                      I_sd=float(i_vals.std(ddof=1)) if len(i_vals) > 1 else 0.0,
                      drive_mV=dv, drive_fraction=dv / thr,
                      alive=bool(e_vals.mean() > self.args.alive_rate),
                      seconds=elapsed)
        for seed, p in zip(seeds_used, per_seed):
            self.rows.append(dict(eta=key, seed=seed, E=p["E"], I=p["I"],
                                  drive_fraction=dv / thr,
                                  method=self.args.eval_method))
        self._cache[key] = result

        flag = "alive" if result["alive"] else "SILENT"
        print(f"  eta={key:6.3f} | drive {100*dv/thr:5.1f}% of threshold | "
              f"E {result['E']:6.3f} Hz | I {result['I']:6.3f} Hz | "
              f"{flag:6s} | {elapsed:5.1f} s", flush=True)
        return result


def split_ei(rates):
    level = rates.index.names[-1]
    return {"E": float(rates.xs("E", level=level).mean()),
            "I": float(rates.xs("I", level=level).mean())}


# --------------------------------------------------------------------------- #
#  Search strategies
# --------------------------------------------------------------------------- #
def bracket(evaluate, lo, hi, predicate, max_expand=12):
    """Smallest interval [a, b] inside [lo, hi] with predicate(a) False, True at b.

    ``predicate`` must be monotone in eta (more drive -> more activity).
    """
    a = lo
    if predicate(evaluate(a)):
        return a, a                       # already satisfied at the low end
    b = a
    step = max((hi - lo) / 8.0, 0.25)
    for _ in range(max_expand):
        b = min(b + step, hi)
        if predicate(evaluate(b)):
            return a, b
        a = b
        step *= 1.5
        if b >= hi:
            break
    return None, None


def bisect(evaluate, a, b, predicate, iterations, tol=1e-2):
    """Refine a bracket where predicate flips from False (a) to True (b)."""
    for _ in range(iterations):
        if b - a < tol:
            break
        mid = 0.5 * (a + b)
        if predicate(evaluate(mid)):
            b = mid
        else:
            a = mid
    return a, b


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_pop(text):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "probe population must be 'area,layer,pop', "
            "e.g. 'precuneus,II/III,E'")
    return tuple(parts)


def parse_abeta(text):
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text            # a CSV path, resolved by helpers/abeta.py


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Search for the eta_ext that keeps a downscaled network "
                    "alive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("network scale")
    g.add_argument("--n-scaling", type=float, default=0.005,
                   help="fraction of neurons kept")
    g.add_argument("--k-scaling", type=float, default=0.1,
                   help="fraction of in-degree kept")
    g.add_argument("--scaling-type", choices=["sqrt", "linear"], default="sqrt")
    g.add_argument("--cc-e-to-e", type=float, default=2.5)
    g.add_argument("--cc-e-to-i", type=float, default=5.0)
    g.add_argument("--abeta-ratio", type=parse_abeta, default=None,
                   help="fraction of amyloid-affected E neurons, or a path to "
                        "an amyloid CSV; omit for a healthy network")
    g.add_argument("--con-path", type=Path, default=None,
                   help="structural connectivity .mat to use instead of the "
                        "group-average one, e.g. an individual's tractography")
    g.add_argument("--fullscale-rates", type=Path, default=None,
                   help="full-scale population rates (pkl) the DC compensation "
                        "is computed from; they must belong to the configuration "
                        "being calibrated (see tools/fullscale_rates.py). "
                        "Default: the model's base_theory_rates.pkl, which "
                        "matches only the repository defaults")

    g = p.add_argument_group("simulation")
    g.add_argument("--t-sim", type=float, default=2000.0,
                   help="simulated time per candidate, ms")
    g.add_argument("--t-transient", type=float, default=500.0,
                   help="discarded before measuring rates, ms")
    g.add_argument("--threads", type=int, default=4, help="OpenMP threads")
    g.add_argument("--seeds", type=int, nargs="+", default=[2903],
                   help="master seeds averaged per candidate")

    g = p.add_argument_group("search")
    g.add_argument("--method",
                   choices=["simulate", "theory", "analytic", "scan"],
                   default="simulate",
                   help="how each candidate is evaluated; 'scan' is a synonym "
                        "for 'simulate' together with --eta-list")
    g.add_argument("--eta-min", type=float, default=1.0)
    g.add_argument("--eta-max", type=float, default=8.0)
    g.add_argument("--eta-list", type=float, nargs="+", default=None,
                   help="explicit eta values for --method scan")
    g.add_argument("--target-e", type=float, nargs=2, default=[3.0, 5.0],
                   metavar=("LO", "HI"),
                   help="target band for the mean excitatory rate, Hz")
    g.add_argument("--target-i", type=float, nargs=2, default=[5.0, 15.0],
                   metavar=("LO", "HI"),
                   help="target band for the mean inhibitory rate, Hz")
    g.add_argument("--alive-rate", type=float, default=0.05,
                   help="mean E rate above which the network counts as alive, Hz")
    g.add_argument("--target-rate", type=float, default=None,
                   help="mean excitatory rate the search aims at, Hz "
                        "(default: the lower edge of --target-e; pass the band "
                        "midpoint for a result less sensitive to seed noise)")
    g.add_argument("--target-fraction", type=float, default=1.1,
                   help="external + DC drive relative to threshold that "
                        "--method analytic solves for")
    g.add_argument("--iterations", type=int, default=6,
                   help="bisection steps per boundary")
    g.add_argument("--probe-pop", type=parse_pop,
                   default=("precuneus", "II/III", "E"),
                   help="population used for the analytic drive read-out")

    g = p.add_argument_group("output")
    g.add_argument("--model-root", type=Path, default=None,
                   help="path to the model source "
                        "(default: the copy bundled with this tutorial)")
    g.add_argument("--outdir", type=Path,
                   default=HERE.parent / "results" / "eta_search",
                   help="results directory; nothing is ever written into the "
                        "model folder")
    g.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    g.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)
    # bootstrap() chdirs into the model directory, so any path given relative to
    # the invocation directory has to be resolved before that happens.
    if args.con_path is not None:
        args.con_path = Path(args.con_path).resolve()
    if isinstance(args.abeta_ratio, str):
        args.abeta_ratio = str(Path(args.abeta_ratio).resolve())
    if args.model_root is not None:
        args.model_root = Path(args.model_root).resolve()
    if args.fullscale_rates is not None:
        args.fullscale_rates = Path(args.fullscale_rates).resolve()
    # The evaluator and the search strategy are independent: --eta-list turns
    # any method into a scan, and --method theory swaps the evaluator.
    args.eval_method = "theory" if args.method == "theory" else "simulate"
    args.scan = args.method == "scan" or args.eta_list is not None
    return args


# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    args.outdir = Path(args.outdir).resolve()
    root = bootstrap(find_model_repo(args.model_root))
    if args.fullscale_rates is None:
        args.fullscale_rates = root / "simulated_data" / "base_theory_rates.pkl"
    print(f"model             : {root}")
    print(f"results directory : {args.outdir}")
    (args.outdir / "networks").mkdir(parents=True, exist_ok=True)
    (args.outdir / "log").mkdir(parents=True, exist_ok=True)

    loggers = build_loggers(str(args.outdir / "log" / "eta_search.log"),
                            mode=args.log_level)
    factory = NetworkFactory(args, loggers[0])

    print()
    print(f"scale        : N_scaling={args.n_scaling}  "
          f"K_scaling={args.k_scaling}  ({args.scaling_type})")
    print(f"method       : {args.method}"
          f"{'  (scan)' if args.scan and args.method != 'scan' else ''}")
    if args.eval_method == "simulate" and args.method != "analytic":
        print(f"per candidate: {args.t_sim:.0f} ms x {len(args.seeds)} seed(s) "
              f"on {args.threads} threads")
    print(f"targets      : E in {args.target_e} Hz, I in {args.target_i} Hz")
    print()

    # ---- the closed-form estimate is always reported: it is nearly free ----
    ana = analytic_eta(factory, args.probe_pop, args.target_fraction)
    print(f"analytic mean drive for {args.probe_pop}:")
    print(f"  dV(eta) = {ana['slope_mV_per_eta']:.3f} * eta "
          f"{ana['intercept_mV']:+.3f} mV,   threshold at "
          f"{ana['to_threshold_mV']:.1f} mV")
    print(f"  at the repository default eta_ext = 1.1 the drive reaches "
          f"{100 * ana['fraction_at_1p1']:.0f}% of threshold")
    print(f"  eta_ext reaching {args.target_fraction:.2f} x threshold: "
          f"{ana['eta']:.3f}")
    print()

    summary = {"config": {k: (str(v) if isinstance(v, (Path, tuple)) else v)
                          for k, v in vars(args).items()},
               "analytic": ana}

    if args.method == "analytic":
        summary["recommended_eta"] = ana["eta"]
        write_outputs(args, pd.DataFrame(), summary)
        print("Analytic estimate only. Verify it with:")
        print(f"  python tools/eta_ext_search.py --method scan --eta-list "
              f"{ana['eta']:.2f}")
        return 0

    evaluate = Evaluator(factory, args, loggers)

    # ------------------------------------------------------------------ scan
    if args.scan:
        etas = args.eta_list or list(np.linspace(args.eta_min, args.eta_max, 8))
        print("scanning:")
        for eta in etas:
            evaluate(eta)
        results = pd.DataFrame(sorted(evaluate._cache.values(),
                                      key=lambda r: r["eta"]))
        summary.update(scan_summary(results, args))
        finish(args, evaluate, results, summary)
        return 0

    # ------------------------------------------- boundary 1: alive at all? --
    print("stage 1 - locating the silent/alive boundary:")
    alive = lambda r: r["alive"]                                    # noqa: E731
    a, b = bracket(evaluate, args.eta_min, args.eta_max, alive)
    if a is None:
        print(f"\nNo eta_ext in [{args.eta_min}, {args.eta_max}] woke the "
              f"network up. Raise --eta-max, or raise --k-scaling.")
        results = pd.DataFrame(sorted(evaluate._cache.values(),
                                      key=lambda r: r["eta"]))
        finish(args, evaluate, results, summary)
        return 1
    a, b = bisect(evaluate, a, b, alive, args.iterations)
    eta_alive = b
    print(f"  --> network wakes up at eta_ext ~ {eta_alive:.3f} "
          f"(silent at {a:.3f})")

    # ------------------------------------------- boundary 2: target band ----
    print("\nstage 2 - reaching the target excitatory rate:")
    lo_e, hi_e = args.target_e
    want = args.target_rate if args.target_rate is not None else lo_e
    reached = lambda r: r["E"] >= want                               # noqa: E731
    a2, b2 = bracket(evaluate, eta_alive, args.eta_max, reached)
    if a2 is None:
        print(f"\nNo eta_ext up to {args.eta_max} reached {want} Hz. "
              f"Raise --eta-max, or lower --target-rate.")
        eta_target = None
    else:
        a2, b2 = bisect(evaluate, a2, b2, reached, args.iterations)
        eta_target = b2
        print(f"  --> mean excitatory rate reaches {want} Hz at "
              f"eta_ext ~ {eta_target:.3f}")

    results = pd.DataFrame(sorted(evaluate._cache.values(),
                                  key=lambda r: r["eta"]))
    summary["eta_alive"] = float(eta_alive)
    summary["eta_target"] = None if eta_target is None else float(eta_target)
    summary.update(scan_summary(results, args))
    finish(args, evaluate, results, summary)
    return 0


def scan_summary(results, args):
    """Best candidate: inside both target bands, else closest to the E midpoint."""
    if results.empty:
        return {}
    lo_e, hi_e = args.target_e
    lo_i, hi_i = args.target_i
    both = results[(results.E.between(lo_e, hi_e)) & (results.I.between(lo_i, hi_i))]
    pool = both if not both.empty else results[results.alive]
    if pool.empty:
        return {"best": None}
    mid = 0.5 * (lo_e + hi_e)
    best = pool.iloc[(pool.E - mid).abs().argsort().iloc[0]]
    return {"best": {k: (None if pd.isna(v) else float(v))
                     for k, v in best.items() if k != "alive"},
            "best_in_both_bands": bool(not both.empty)}


def write_outputs(args, results, summary):
    if not results.empty:
        results.to_csv(args.outdir / "eta_search.csv", index=False)
    with open(args.outdir / "eta_search.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)


def plot(args, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(*args.target_e, color="#3b6ea5", alpha=0.12,
               label=f"target E {args.target_e[0]}-{args.target_e[1]} Hz")
    ax.axhspan(*args.target_i, color="#c0392b", alpha=0.10,
               label=f"target I {args.target_i[0]}-{args.target_i[1]} Hz")
    r = results.sort_values("eta")
    ax.errorbar(r.eta, r.E, yerr=r.E_sd, marker="o", ms=5, lw=1.4,
                color="#3b6ea5", capsize=3, label="excitatory")
    ax.errorbar(r.eta, r.I, yerr=r.I_sd, marker="s", ms=5, lw=1.4,
                color="#c0392b", capsize=3, label="inhibitory")
    ax.axvline(1.1, ls=":", c="k", lw=1)
    ax.text(1.1, ax.get_ylim()[1], " repository default", va="top", fontsize=8)
    ax.set_xlabel("eta_ext")
    ax.set_ylabel("mean population firing rate (Hz)")
    ax.set_title(f"Working point vs external drive\n"
                 f"N_scaling={args.n_scaling}, K_scaling={args.k_scaling}, "
                 f"method={args.method}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = args.outdir / "eta_search.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def finish(args, evaluate, results, summary):
    if evaluate.rows:
        pd.DataFrame(evaluate.rows).to_csv(args.outdir / "eta_search_raw.csv",
                                           index=False)
    write_outputs(args, results, summary)

    print("\n" + "=" * 72)
    if not results.empty:
        show = results[["eta", "drive_fraction", "E", "E_sd", "I", "I_sd",
                        "alive"]].copy()
        show["drive_fraction"] = (100 * show.drive_fraction).round(1)
        show = show.rename(columns={"drive_fraction": "drive_%"})
        print(show.to_string(index=False,
                             float_format=lambda v: f"{v:.3f}"))
    best = summary.get("best")
    if best:
        tag = ("inside both target bands" if summary.get("best_in_both_bands")
               else "closest to the target, but OUTSIDE at least one band")
        print(f"\nrecommended eta_ext = {best['eta']:.3f}   ({tag})")
        print(f"   mean rates: E {best['E']:.2f} Hz, I {best['I']:.2f} Hz   "
              f"mean drive {100 * best['drive_fraction']:.0f}% of threshold")
    if summary.get("eta_alive") is not None:
        print(f"   network stops being silent at eta_ext ~ "
              f"{summary['eta_alive']:.3f}")
    if args.eval_method == "theory":
        print("\nNOTE: the mean-field ignores weight/delay distributions and "
              "correlations;\n      treat its eta_ext as a bracket, not an "
              "answer.")
        if best:
            print(f"      Verify with:  python tools/eta_ext_search.py "
                  f"--method scan --eta-list {best['eta']:.2f}")
    if not args.no_plot and not results.empty:
        print(f"\nfigure : {plot(args, results)}")
    print(f"results: {args.outdir / 'eta_search.csv'}")
    print(f"summary: {args.outdir / 'eta_search.json'}")


if __name__ == "__main__":
    sys.exit(main())
