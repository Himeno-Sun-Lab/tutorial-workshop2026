#!/usr/bin/env python3
"""Render the full-scale figures the notebook shows, into ``figures/fullscale/<run>/``.

The full-scale runs (``fullscale/``) are patient-derived and 2-3.5 GB each, so
they are not distributed with the tutorial; the notebook shows these pre-rendered
figures instead of computing them. Run this script on a machine that has
``fullscale/`` whenever a run or a figure changes, and commit ``figures/fullscale/``:

    python tools/make_fullscale_figures.py

It also prints every number the notebook text quotes, so text and figures come
from the same data. Runs in the tutorial's NEST environment.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LogNorm                 # noqa: E402
from PIL import Image                                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (TUTORIAL, fullscale_info, fullscale_fc, has_fullscale,  # noqa: E402
                      load_result, rates, run_dir, sim_dir, pop_level)

# Runs to render. t_switch (ms): the first change from the active to the quiet
# state, zoomed on in figure 2 - None detects it from the mean rate (and finds
# none in a run that never enters the quiet state); raster_areas: the areas of
# the raster figure - None takes the two most active areas before the switch and
# the least active one. The same three areas in every run make them comparable.
RASTER_AREAS_ALL = ["postcentral", "pericalcarine", "caudalanteriorcingulate"]
RUNS = {
    "thu_sub007_L": dict(t_switch=4400, raster_areas=RASTER_AREAS_ALL),
    "thu_sub016_L": dict(t_switch=None, raster_areas=RASTER_AREAS_ALL),
    "thu_sub011_R": dict(t_switch=None, raster_areas=RASTER_AREAS_ALL),
}
DPI = 120
T0 = 1000                    # ms, the first second is discarded from every statistic
HEAT_T0 = 500                # ms, the time-resolved figures start here
CURRENT_PER_ROW = 10000      # units of input_current.pkl per row of the current traces
# Full names of the Desikan-Killiany areas (Desikan et al. 2006), for the current traces
FULL_NAMES = {
    "bankssts": "Banks of the superior temporal sulcus",
    "caudalanteriorcingulate": "Caudal anterior cingulate cortex",
    "caudalmiddlefrontal": "Caudal middle frontal gyrus",
    "cuneus": "Cuneus",
    "entorhinal": "Entorhinal cortex",
    "frontalpole": "Frontal pole",
    "fusiform": "Fusiform gyrus",
    "inferiorparietal": "Inferior parietal cortex",
    "inferiortemporal": "Inferior temporal gyrus",
    "insula": "Insula",
    "isthmuscingulate": "Isthmus of the cingulate cortex",
    "lateraloccipital": "Lateral occipital cortex",
    "lateralorbitofrontal": "Lateral orbitofrontal cortex",
    "lingual": "Lingual gyrus",
    "medialorbitofrontal": "Medial orbitofrontal cortex",
    "middletemporal": "Middle temporal gyrus",
    "paracentral": "Paracentral lobule",
    "parahippocampal": "Parahippocampal gyrus",
    "parsopercularis": "Pars opercularis (inferior frontal gyrus)",
    "parsorbitalis": "Pars orbitalis (inferior frontal gyrus)",
    "parstriangularis": "Pars triangularis (inferior frontal gyrus)",
    "pericalcarine": "Pericalcarine cortex",
    "postcentral": "Postcentral gyrus",
    "posteriorcingulate": "Posterior cingulate cortex",
    "precentral": "Precentral gyrus",
    "precuneus": "Precuneus",
    "rostralanteriorcingulate": "Rostral anterior cingulate cortex",
    "rostralmiddlefrontal": "Rostral middle frontal gyrus",
    "superiorfrontal": "Superior frontal gyrus",
    "superiorparietal": "Superior parietal cortex",
    "superiortemporal": "Superior temporal gyrus",
    "supramarginal": "Supramarginal gyrus",
    "temporalpole": "Temporal pole",
    "transversetemporal": "Transverse temporal gyrus",
}
E_COLOR, I_COLOR = "#3b6ea5", "#c0392b"
# The two states. In the quiet state the mean rate over all areas is steady at
# about 0.45 spikes/s in every hemisphere of the cohort; in the active state it
# stays above 1 spikes/s except in pauses of at most ~100 ms. A stretch counts as
# quiet when this mean stays below QUIET_HZ (20 ms bins) for at least QUIET_MIN_MS;
# shorter pauses belong to the active state. TRIM_MS after every change of state
# is left out of the statistics of the state it enters.
QUIET_HZ, QUIET_MIN_MS, STATE_BIN, TRIM_MS = 0.6, 200, 20, 100
# Set per run by main()
RUN = OUT = T_SWITCH = RASTER_AREAS = None
EPISODES = STATES = None
numbers = {}


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {(OUT / name).relative_to(TUTORIAL)}  ({(OUT / name).stat().st_size / 1e6:.1f} MB)")


def binned(v, bin_ms, t0=T0, t1=None):
    """spikes/ms per neuron (1 ms bins) -> Hz in `bin_ms` bins over [t0, t1)."""
    v = np.asarray(v[t0:t1])
    n = len(v) // bin_ms * bin_ms
    return v[:n].reshape(-1, bin_ms).mean(axis=1) * 1000


def detect_switch(area_hist, t_sim, window=500, step=20):
    """Time (ms) of the largest change of the mean area rate between the
    `window` ms before and after it, and that change as a ratio."""
    mean = np.mean([np.asarray(v, dtype=float) for v in area_hist], axis=0) * 1000
    best, t_best = 0.0, None
    for t in range(T0 + window, t_sim - window, step):
        before, after = mean[t - window:t].mean(), mean[t:t + window].mean()
        score = abs(np.log((after + 1e-3) / (before + 1e-3)))
        if score > best:
            best, t_best = score, t
    return t_best, float(np.exp(best))


def quiet_episodes(area_hist, t_sim):
    """[(start, end), ...] in ms of the stretches in the quiet state."""
    mean = np.mean([np.asarray(v, dtype=float) for v in area_hist], axis=0)
    quiet = binned(mean, STATE_BIN, T0, t_sim) < QUIET_HZ
    episodes, start = [], None
    for i, q in enumerate([*quiet, False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * STATE_BIN >= QUIET_MIN_MS:
                episodes.append((T0 + start * STATE_BIN, T0 + i * STATE_BIN))
            start = None
    return episodes


def state_intervals(episodes, t_sim):
    """{"active": [(t0, t1), ...], "quiet": [...]} - each state's stretches, without
    the first TRIM_MS after a change into it."""
    bounds = [T0, *[t for ep in episodes for t in ep], t_sim]
    stretches = [(bounds[i], bounds[i + 1], "active" if i % 2 == 0 else "quiet")
                 for i in range(len(bounds) - 1)]
    states = {"active": [], "quiet": []}
    for t0, t1, state in stretches:
        if t0 > T0:
            t0 += TRIM_MS
        if t1 > t0:
            states[state].append((t0, t1))
    return {k: v for k, v in states.items() if v}


def in_states(times, intervals):
    """Boolean mask of `times` (ms) inside any of `intervals`."""
    times = np.asarray(times)
    return np.any([(times >= t0) & (times < t1) for t0, t1 in intervals], axis=0)


def main():
    global RUN, OUT, T_SWITCH, RASTER_AREAS, EPISODES, STATES, numbers
    for run, cfg in RUNS.items():
        if not has_fullscale(run):
            print(f"{run}: not in fullscale/ on this machine - skipped")
            continue
        RUN, numbers = run, {}
        OUT = TUTORIAL / "figures" / "fullscale" / run
        T_SWITCH, RASTER_AREAS = cfg["t_switch"], cfg["raster_areas"]
        area_hist = load_result(run, "rate_histogram_areas.pkl")
        t_sim = len(area_hist.iloc[0])
        EPISODES = quiet_episodes(area_hist, t_sim)
        STATES = state_intervals(EPISODES, t_sim)
        t_detect, ratio = detect_switch(area_hist, t_sim)
        numbers["switch_detected"] = dict(t=t_detect, rate_ratio=ratio)
        if T_SWITCH is None and any(start > T0 for start, _ in EPISODES):
            T_SWITCH = t_detect
        numbers["states"] = dict(
            t_switch=T_SWITCH, quiet_episodes=EPISODES,
            quiet_fraction=sum(e - s for s, e in EPISODES) / (t_sim - T0),
            changes=sum(1 for s, e in EPISODES for t in (s, e) if T0 < t < t_sim),
            intervals=STATES)
        print(f"{run}: switch at {T_SWITCH} ms (largest change of the mean rate at "
              f"{t_detect} ms, x{ratio:.1f}); quiet episodes {EPISODES}")
        render()


def render():
    OUT.mkdir(parents=True, exist_ok=True)
    info = fullscale_info(RUN)
    numbers["info"] = {k: str(v) for k, v in info.items()}
    area_hist = load_result(RUN, "rate_histogram_areas.pkl")
    pop_hist = load_result(RUN, "rate_histogram.pkl")
    t_sim = len(area_hist.iloc[0])
    r_all = rates(RUN)
    level = pop_level(r_all)
    numbers["whole_run"] = dict(E=r_all.xs("E", level=level).mean(), I=r_all.xs("I", level=level).mean())

    # ---- 1. the whole run: mean rates and per-area heatmap -------------------------
    BIN = 20
    t = (np.arange(HEAT_T0, t_sim, BIN) + BIN / 2) / 1000
    area_rate = pd.DataFrame(np.vstack([binned(v, BIN, HEAT_T0) for v in area_hist]), index=area_hist.index)
    t_sort = T_SWITCH or t_sim
    order = area_rate.iloc[:, (T0 - HEAT_T0) // BIN:(t_sort - HEAT_T0) // BIN].mean(axis=1) \
        .sort_values(ascending=False).index
    area_rate = area_rate.loc[order]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [1, 3]})
    for pop_name, color in [("E", E_COLOR), ("I", I_COLOR)]:
        mean = np.mean([binned(v, BIN, HEAT_T0) for v in pop_hist.xs(pop_name, level=level)], axis=0)
        ax0.plot(t, mean, c=color, lw=1, label=f"mean over {pop_name} populations")
    for i, (s, e) in enumerate(EPISODES):
        ax0.axvspan(s / 1000, e / 1000, color="0.85", lw=0, zorder=0,
                    label="quiet state" if i == 0 else None)
    if T_SWITCH:
        ax0.axvline(T_SWITCH / 1000, c="k", ls=":")
    ax0.set_ylabel("rate (Hz)"); ax0.legend(fontsize=8, loc="upper right")
    im = ax1.imshow(area_rate.clip(lower=0.01).values, aspect="auto", cmap="magma",
                    norm=LogNorm(vmin=0.01, vmax=area_rate.values.max()),
                    extent=[HEAT_T0 / 1000, t_sim / 1000, len(area_rate) - 0.5, -0.5])
    ax1.set_yticks(range(len(area_rate)), area_rate.index, fontsize=7)
    ax1.set_xlabel("time (s)")
    if T_SWITCH:
        ax1.axvline(T_SWITCH / 1000, c="w", ls=":")
        ax1.set_title(f"rate per area, {BIN} ms bins - sorted by the rate before {T_SWITCH / 1000} s")
    else:
        ax1.set_title(f"rate per area, {BIN} ms bins - sorted by the mean rate")
    fig.colorbar(im, ax=ax1, label="rate (Hz)", shrink=0.8)
    save(fig, "state_switch_areas.png")
    render_currents(area_rate.index, t_sim, BIN)

    for state, intervals in STATES.items():
        idx = np.concatenate([np.arange(t0, t1) for t0, t1 in intervals])
        r_win = pop_hist.apply(lambda v: np.asarray(v)[idx].mean() * 1000)
        numbers[f"rates_{state}"] = dict(E=r_win.xs("E", level=level).mean(),
                                         I=r_win.xs("I", level=level).mean(),
                                         silent=int((r_win == 0).sum()), fastest=r_win.max())

    if T_SWITCH:
        render_zoom(area_hist)
    render_rest(area_hist, pop_hist, area_rate, level, t_sim)


def render_currents(order, t_sim, BIN):
    # ---- 1b. synaptic input current per area, as traces: range, bins and order of 1 -----
    # Not normalised: every trace, in every run, is drawn on the same scale - CURRENT_PER_ROW
    # units of input_current.pkl per row - from its own zero line, so that the size of the
    # deflections compares across areas and runs. Large ones run into the rows above.
    currents = load_result(RUN, "input_current.pkl")          # areas x 1 ms
    t = (np.arange(HEAT_T0, t_sim, BIN) + BIN / 2) / 1000
    fig, ax = plt.subplots(figsize=(13, 8.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 0.96, 1))          # room for the scale bar
    zero = np.arange(len(order)) + 0.35                        # zero current of each row
    top = zero[0] - 0.5
    for y0, area in zip(zero, order):
        y = y0 - binned(currents.loc[area].values, BIN, HEAT_T0, t_sim) / 1000 / CURRENT_PER_ROW
        ax.plot(t, y, c="k", lw=0.5)
        top = min(top, y.min() - 0.1)
    if T_SWITCH:
        ax.axvline(T_SWITCH / 1000, c=I_COLOR, ls=":")
    ax.set_xlim(HEAT_T0 / 1000, t_sim / 1000)
    ax.set_ylim(len(order) - 0.5, top)
    ax.set_yticks(zero, [FULL_NAMES.get(a, a) for a in order], fontsize=7)
    # scale bar right of the last row: one row = CURRENT_PER_ROW
    bar = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.plot([1.01, 1.01], [zero[-1], zero[-1] - 1], c="k", lw=1.5, transform=bar, clip_on=False)
    ax.text(1.017, zero[-1] - 0.5, f"{CURRENT_PER_ROW:,} a.u.", transform=bar, rotation=90,
            va="center", ha="left", fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title(f"synaptic input current per area, {BIN} ms bins - one scale for all areas; "
                 "areas in the order of the rate figure")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "synaptic_currents.png")


def render_zoom(area_hist):
    # ---- 2. zoom on the switch: when does each area drop? -------------------------
    ZB, Z0, Z1 = 5, T_SWITCH - 500, T_SWITCH + 500
    zoom = pd.DataFrame(np.vstack([binned(v, ZB, Z0, Z1) for v in area_hist]), index=area_hist.index)
    tz = (np.arange(Z0, Z1, ZB) + ZB / 2) / 1000
    smooth = zoom.T.rolling(4, center=True, min_periods=1).mean().T          # 20 ms
    pre = zoom.loc[:, (tz >= (T_SWITCH - 400) / 1000) & (tz < (T_SWITCH - 50) / 1000)].mean(axis=1)
    post = zoom.loc[:, (tz >= (T_SWITCH + 200) / 1000)].mean(axis=1)
    # The first state is bursty, so an area falls below any threshold between
    # bursts too; its switch is the END of its activity: the last time its
    # smoothed rate is above half way between the two states.
    drop = {}
    for area in zoom.index:
        if pre[area] < 2 * post[area]:
            continue                                          # no clear drop
        half = (pre[area] + post[area]) / 2
        above = np.flatnonzero(smooth.loc[area].values > half)
        if len(above):
            drop[area] = tz[above[-1]]
    drop = pd.Series(drop).sort_values()
    numbers["drop"] = dict(n_areas=len(drop), first=list(drop.index[:3]),
                           first_t=float(drop.iloc[0]), spread_ms=float((drop.max() - drop.min()) * 1000),
                           spread80_ms=float((drop.quantile(0.9) - drop.quantile(0.1)) * 1000),
                           no_drop=sorted(set(zoom.index) - set(drop.index)))
    rows = list(drop.index) + sorted(set(zoom.index) - set(drop.index))
    fig, ax = plt.subplots(figsize=(13, 8), layout="constrained")
    im = ax.imshow(zoom.loc[rows].clip(lower=0.01).values, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=0.01, vmax=zoom.values.max()),
                   extent=[Z0 / 1000, Z1 / 1000, len(rows) - 0.5, -0.5])
    ax.plot(drop.values, range(len(drop)), "c.", ms=5, label="end of activity (last time above half way, 20 ms smoothing)")
    ax.axhline(len(drop) - 0.5, c="w", lw=0.8)
    ax.set_yticks(range(len(rows)), rows, fontsize=7)
    ax.set_xlabel("time (s)")
    ax.set_title(f"the switch, {ZB} ms bins - areas sorted by when their activity ends; "
                 "below the line: no clear drop")
    ax.legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=ax, label="rate (Hz)", shrink=0.8)
    save(fig, "state_switch_zoom.png")


def render_rest(area_hist, pop_hist, area_rate, level, t_sim):
    BIN = 20
    rng = np.random.default_rng(0)

    # ---- 3. populations, by layer -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 9), sharey=False, layout="constrained")
    for ax, pop_name in zip(axes, ["E", "I"]):
        hist = pop_hist.xs(pop_name, level=level)
        layers = ["II/III", "IV", "V", "VI"]
        idx = [k for layer in layers for k in hist.index if k[1] == layer]
        mat = np.vstack([binned(hist[k], BIN, HEAT_T0) for k in idx])
        im = ax.imshow(np.clip(mat, 0.01, None), aspect="auto", cmap="magma",
                       norm=LogNorm(vmin=0.01, vmax=max(mat.max(), 1)),
                       extent=[HEAT_T0 / 1000, t_sim / 1000, len(idx) - 0.5, -0.5])
        bounds = np.cumsum([sum(k[1] == layer for k in idx) for layer in layers])
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, c="w", lw=0.8)
        ax.set_yticks([(b0 + b1) / 2 - 0.5 for b0, b1 in zip([0, *bounds[:-1]], bounds)],
                      [f"{layer} {pop_name}" for layer in layers])
        if T_SWITCH:
            ax.axvline(T_SWITCH / 1000, c="c", ls=":")
        ax.set_xlabel("time (s)")
        ax.set_title(f"{pop_name} populations, grouped by layer (34 areas each)")
        fig.colorbar(im, ax=ax, label="rate (Hz)", shrink=0.6)
    save(fig, "state_switch_layers.png")

    # ---- 4. spikes across the switch ------------------------------------------------
    spikes = pd.read_pickle(sim_dir(RUN) / "spikes.pkl")
    AREAS = RASTER_AREAS
    if AREAS is None:
        pre = area_rate.iloc[:, (T0 - HEAT_T0) // BIN:((T_SWITCH or t_sim) - HEAT_T0) // BIN] \
            .mean(axis=1).sort_values()
        AREAS = [pre.index[-1], pre.index[-2], pre.index[0]]
    numbers["raster_areas"] = AREAS
    # 3 s: from 1.5 s before the switch, or the middle of a run without one
    start = max(T0, T_SWITCH - 1500) if T_SWITCH else t_sim // 2 - 1500
    WINDOW = (start, start + 3000)
    numbers["raster_window"] = WINDOW
    # the other changes of state inside the window, dashed
    changes = [t for ep in EPISODES for t in ep if WINDOW[0] < t < WINDOW[1]
               and not (T_SWITCH and abs(t - T_SWITCH) <= 50)]
    fig, axes = plt.subplots(1, len(AREAS), figsize=(15, 5.5), layout="constrained")
    for ax, area in zip(axes, AREAS):
        row, ticks = 0, []
        for (layer, pop_name), trains in spikes.loc[area].items():
            n = max(1, int(0.05 * len(trains))) if len(trains) else 0
            for i in rng.choice(len(trains), n, replace=False):
                s = trains[i][(trains[i] >= WINDOW[0]) & (trains[i] < WINDOW[1])]
                # a marker sized in points, so that it does not depend on the dpi
                ax.plot(s / 1000, np.full(len(s), row), ".", ms=0.8, mew=0,
                        c=E_COLOR if pop_name == "E" else I_COLOR)
                row += 1
            ticks.append((row - n / 2, f"{layer} {pop_name}"))
        for b in changes:
            ax.axvline(b / 1000, c="0.5", ls="--", lw=0.8)
        if T_SWITCH:
            ax.axvline(T_SWITCH / 1000, c="k", ls=":")
        ax.set_xlim(WINDOW[0] / 1000, WINDOW[1] / 1000)
        ax.set_yticks([y for y, _ in ticks], [name for _, name in ticks])
        ax.set_ylim(row, 0)
        ax.set_xlabel("time (s)"); ax.set_title(area)
        ax.legend(handles=[plt.Line2D([], [], color=c, marker=".", markersize=8, linestyle="None",
                                      label=lab) for lab, c in [("E", E_COLOR), ("I", I_COLOR)]],
                  loc="lower right", fontsize=8, framealpha=0.9)
    save(fig, "state_switch_rasters.png")

    # ---- 5. rate, irregularity and synchrony in each state --------------------------
    # Each state may consist of several stretches: ISIs are taken within a
    # stretch, spike counts are concatenated over them.
    def population_stats(trains, intervals, n_max=200, bin_ms=5):
        pieces = [[s[(s >= t0) & (s < t1)] for t0, t1 in intervals] for s in trains[:n_max]]
        pieces = [p for p in pieces if sum(len(x) for x in p) >= 5]
        if len(pieces) < 2:
            return np.nan, np.nan
        isis = [np.concatenate([np.diff(x) for x in p]) for p in pieces]
        cv = np.mean([d.std() / d.mean() for d in isis if len(d) >= 2])
        counts = [np.concatenate([np.histogram(x, np.arange(t0, t1 + bin_ms, bin_ms))[0]
                                  for x, (t0, t1) in zip(p, intervals)]) for p in pieces]
        cc = np.corrcoef(np.array(counts))
        return cv, np.nanmean(cc[np.triu_indices(len(pieces), 1)])

    rows = []
    for state, intervals in STATES.items():
        idx = np.concatenate([np.arange(t0, t1) for t0, t1 in intervals])
        for pop_id, trains in spikes.items():
            order = rng.permutation(len(trains))
            cv, cc = population_stats([trains[i] for i in order], intervals)
            rows.append(dict(state=state, pop=pop_id[2], rate=np.asarray(pop_hist[pop_id])[idx].mean() * 1000,
                             cv=cv, cc=cc))
    del spikes
    stats = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for ax, (column, label) in zip(axes, [("rate", "rate (Hz)"), ("cv", "CV of the ISI"),
                                          ("cc", "spike-count correlation (5 ms bins)")]):
        data = [stats.loc[(stats.state == s) & (stats["pop"] == p), column].dropna()
                for s in STATES for p in ("E", "I")]
        box = ax.boxplot(data, patch_artist=True, widths=0.6,
                         tick_labels=[f"{p} {s}" for s in STATES for p in ("E", "I")])
        for patch, color in zip(box["boxes"], [E_COLOR, I_COLOR] * len(STATES)):
            patch.set_facecolor(color); patch.set_alpha(0.5)
        ax.set_title(label)
    axes[0].set_yscale("symlog", linthresh=1)
    save(fig, "state_statistics.png")
    for state in STATES:
        s = stats[stats.state == state]
        numbers[f"stats_{state}"] = dict(
            median=s.groupby("pop")[["rate", "cv", "cc"]].median().round(3).to_dict(),
            pops_rate_above_10=int((s.rate > 10).sum()), max_rate=float(s.rate.max()),
            pops_cv_above_1_2=int((s.cv > 1.2).sum()), pops_cc_above_0_1=int((s.cc > 0.1).sum()),
            max_cc=float(s.cc.max()))

    # ---- 6. FC: one figure, kept small ------------------------------------------------
    sim_fc, exp_fc = fullscale_fc(RUN)
    currents = load_result(RUN, "input_current.pkl").T
    low = np.tril_indices(len(exp_fc), -1)

    def fc_r(intervals):
        window = currents[in_states(currents.index, intervals)]
        sim = window.corr().sort_index().sort_index(axis=1)
        return float(np.corrcoef(sim.values[low], exp_fc.values[low])[0, 1])

    numbers["fc"] = {"whole": fc_r([(T0, t_sim)]), **{state: fc_r(iv) for state, iv in STATES.items()},
                     "sim_mean": float(sim_fc.values[low].mean())}
    for line in (run_dir(RUN) / "correlations.txt").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("exp__sim_syn__corr_rh", "exp__sim_bold__corr_rh"):
            numbers["fc"][key] = float(value)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, (title, fc) in zip(axes, [("simulated (synaptic currents)", sim_fc),
                                      ("empirical, same person", exp_fc)]):
        im = ax.imshow(fc.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation")
    save(fig, "fc_matrices.png")

    # ---- 7. the server's own summary figure, resampled from its dpi to DPI -----------
    server = Image.open(run_dir(RUN) / "plots" / "figure_spike_statistics.png")
    src_dpi = server.info.get("dpi", (DPI, DPI))[0]
    size = tuple(round(n * DPI / src_dpi) for n in server.size)
    server.resize(size, Image.LANCZOS).save(OUT / "server_spike_statistics.png", dpi=(DPI, DPI))
    print(f"  wrote {(OUT / 'server_spike_statistics.png').relative_to(TUTORIAL)}  "
          f"({(OUT / 'server_spike_statistics.png').stat().st_size / 1e6:.1f} MB)")

    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float) + "\n")
    print(json.dumps(numbers, indent=2, default=float))


if __name__ == "__main__":
    main()
