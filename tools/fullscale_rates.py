#!/usr/bin/env python3
"""Full-scale population rates for an experiment, by self-consistent iteration.

Why this exists
---------------
A downscaled network (``K_scaling = k < 1``) keeps only ``sqrt(k)`` of the mean
synaptic input; ``Network.extraDCforScaledNetwork()`` supplies the rest as a DC
current computed from the rates of the *full-scale* network
(``net_params["fullscale_rates"]``).  Those rates must belong to the very
configuration being run -- same weights, drive, connectome and neuron
parameters.  The model's ``simulated_data/base_theory_rates.pkl`` matches only
the repository defaults (``eta_ext = 1.1``, ``cc_scalingEtoE = 1``,
``cc_scalingEtoI = 2``).

With the compensation in place the downscaled network approximates the
full-scale one, so the full-scale rates are the fixed point of

    nu  ->  DC(nu)  ->  simulate the downscaled network  ->  nu_sim

The plain iteration ``nu <- nu_sim`` oscillates: inhibition dominates the mean
recurrent input, so rates that are too high produce a DC that silences the
network and vice versa (gain around -5 at ``N_scaling=0.005, K_scaling=0.1``).
The damped update ``nu <- nu + alpha * (nu_sim - nu)`` with ``alpha ~ 0.17``
converges within a few iterations and averages finite-size noise on the way.

Usage
-----
    python tools/fullscale_rates.py experiments/tutorial_CN.py \\
        --out experiments/fullscale_rates/tutorial_CN.pkl

    # calibrating eta_ext: the rates depend on it, so override it here
    python tools/fullscale_rates.py experiments/tutorial_CN.py --eta 2.5 \\
        --out /tmp/cn_eta2.5.pkl

Progress (one row per iteration) goes to ``--workdir`` (default
``../results/fullscale_rates/<experiment>/``), never into the model folder.
Runs in the tutorial's NEST environment.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eta_ext_search as ees  # noqa: E402  (bootstrap, simulate_rates)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Self-consistent full-scale rates for an experiment file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("experiment", type=Path, help="experiments/<name>.py")
    p.add_argument("--out", type=Path, required=True,
                   help="where to write the final rate Series (pkl)")
    p.add_argument("--eta", type=float, default=None,
                   help="override the experiment's eta_ext")
    p.add_argument("--start", type=Path, default=None,
                   help="initial rates (default: base_theory_rates.pkl)")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.17,
                   help="damping of the fixed-point update")
    p.add_argument("--t-sim", type=float, default=2000.0, help="ms per run")
    p.add_argument("--t-transient", type=float, default=500.0, help="ms")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=None,
                   help="master seed (default: the experiment's)")
    p.add_argument("--workdir", type=Path, default=None)
    args = p.parse_args(argv)
    args.experiment = args.experiment.resolve()
    args.out = args.out.resolve()
    if args.start is not None:
        args.start = args.start.resolve()
    if args.workdir is None:
        args.workdir = (HERE.parent / "results" / "fullscale_rates"
                        / args.experiment.stem)
    args.workdir = args.workdir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    for sub in ("networks", "rates", "log"):
        (args.workdir / sub).mkdir(parents=True, exist_ok=True)

    root = ees.bootstrap(ees.find_model_repo())      # chdir into the model
    loggers = ees.build_loggers(str(args.workdir / "log" / "fullscale_rates.log"),
                                mode="WARNING")
    exp = runpy.run_path(str(args.experiment))
    params = deepcopy(ees.default_net_params.params)
    ees.nested_dict_update(params, exp["net_params"], loggers[0])
    params["outpath"] = str(args.workdir / "networks")
    params["mode"] = "WARNING"
    if args.eta is not None:
        params["input_params"]["eta_ext"] = float(args.eta)
    eta = params["input_params"]["eta_ext"]
    seed = args.seed
    if seed is None:
        seed = exp.get("sim_params", {}).get(
            "master_seed", ees.default_sim_params.params["master_seed"])

    print(f"experiment : {args.experiment.name}   eta_ext={eta}   seed={seed}")
    print(f"scale      : N_scaling={params['N_scaling']}  "
          f"K_scaling={params['K_scaling']}  ({params['scaling_type']})")
    NN = ees.NeuronNumbers(surface_area=params["surface_area"],
                           logger=loggers[0], **params["cytoarchitecture_params"])
    SN = ees.SynapseNumbers(NN=NN, logger=loggers[0],
                            **params["predictive_connectomic_params"])

    start = args.start or root / "simulated_data" / "base_theory_rates.pkl"
    nu = pd.read_pickle(start).astype(float)
    is_E = nu.index.get_level_values(-1) == "E"
    sim_args = SimpleNamespace(t_sim=args.t_sim, t_transient=args.t_transient,
                               threads=args.threads, outdir=args.workdir,
                               log_level="WARNING")
    rows = []
    for it in range(args.iters):
        path = args.workdir / "rates" / f"eta{eta:.3f}_it{it}.pkl"
        nu.to_pickle(path)
        p = deepcopy(params)
        p["fullscale_rates"] = str(path)
        net = ees.Network(NN, SN, p, logger=loggers[0])
        net.dump(params["outpath"])

        t0 = time.time()
        r = ees.simulate_rates(net, sim_args, seed, loggers)
        r.index = pd.MultiIndex.from_tuples(list(r.index), names=nu.index.names)
        sim = nu.copy()
        sim.loc[r.index] = r.values        # populations emptied by N_scaling keep nu
        step = (sim - nu).abs()
        rows.append(dict(it=it, eta=eta,
                         E_in=float(nu[is_E].mean()), I_in=float(nu[~is_E].mean()),
                         E_sim=float(r.xs("E", level=-1).mean()),
                         I_sim=float(r.xs("I", level=-1).mean()),
                         dE=float(step[is_E].mean()), dI=float(step[~is_E].mean()),
                         seconds=time.time() - t0))
        pd.DataFrame(rows).to_csv(args.workdir / "iterations.csv", index=False)
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                          for k, v in rows[-1].items()}), flush=True)
        nu = nu + args.alpha * (sim - nu)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    nu.to_pickle(args.out)
    print(f"\nfull-scale rates: E {nu[is_E].mean():.3f} Hz, "
          f"I {nu[~is_E].mean():.3f} Hz (population means)")
    print(f"written to      : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
