# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Research scripts for a doctoral project. Positions of a moving object are estimated via ArUco
markers (recorded from video, with noise/jitter). The object actually moves along a known
straight line at constant (or near-constant) velocity. These scripts correct/align the noisy
ArUco-estimated positions against that known straight-line ground truth, to quantify estimation
error. There is no application code, tests, or build — just independent, run-as-a-script analyses
that read a CSV, do a numeric fit, write a CSV, and pop up a matplotlib plot.

## Running

No virtualenv/requirements file exists yet. Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`.

```
python method1.py       # reads samples.csv (repo root), writes optimized_positions.csv
python method1_v2.py    # reads every CSV in input/, writes matching files to output/
python method2.py       # reads samples.csv (repo root), writes optimized_positions_method2.csv
```

Each script loops over parameter combinations and calls `plt.show()` per combination — a plot
window pops up and **blocks execution** until closed, for every iteration.

## The three methods

All three take a set of estimated 2D points (`P_est`, from ArUco) and fit them onto the known
straight line of real motion, but differ in how correspondence/optimization is set up:

- **`method1.py`** — original version. Hardcodes `samples.csv` as input (`;`-separated,
  `,`-decimal). Iterates over every combination of `sampleSpaceMillis` (`[0, 1000, 2000]`) and
  `multipleMarkersBehaviour` (`CLOSEST/WEIGHTED_AVERAGE/AVERAGE/WEIGHTED_MEDIAN/MEDIAN`) found in
  the CSV, filtering rows for each combo. The real line's endpoints are taken from that combo's
  first/last `realX/realY` values. Least-squares fit (`scipy.optimize.minimize`, L-BFGS-B) finds
  each point's scalar position along the line; **endpoints are pinned** to `s=0`/`s=length` and
  not optimized. Plots the trajectory with markers (from `distribucion_markers_1_rev1.json`) drawn
  per frame. Appends results across all combos into one `optimized_positions.csv`.

- **`method1_v2.py`** — generalized version of Método 1 for arbitrary experiments. Reads
  **every** `*.csv` in `input/` (`,`-separated, `.`-decimal — different convention from
  method1/2's `samples.csv`), one output file per input file written to `output/` with the same
  filename. The real line's endpoints (`P_real`) are **hardcoded at the top of the file**
  (currently `[3.733, 0.691] → [0.693, 0.882]`) instead of read from the data — update this
  constant when the ground-truth line changes between experiments. Unlike `method1.py`, **all**
  points (including the first/last) are free in the optimization — none are pinned. Output CSVs
  preserve all original input columns plus `alineatedRealX`/`alineatedRealY`.

- **`method2.py`** — same `samples.csv` input, but aligns sequences instead of doing a per-point
  least-squares projection: it runs a **Needleman-Wunsch** global sequence alignment (similarity =
  negative Euclidean distance) between the ArUco estimates of each combo and a single fixed
  reference — the real interpolated trajectory at `sampleSpaceMillis=0` for the first behaviour in
  `multiple_markers_behaviour_set`. It then saves the *aligned reference real positions*
  (`optimized_positions_method2.csv`), i.e. this method resamples/matches the ground truth to the
  estimate's timing rather than projecting estimates onto a line.

When comparing methods, note `method1_v2.py` uses a different CSV dialect (`,`/`.`) and
input/output layout (`input/`, `output/`) than `method1.py`/`method2.py` (`;`/`,`, repo root).

## Data files

- `samples.csv`, `optimized_positions.csv`, `optimized_positions_method2.csv` — tracked in git,
  live at repo root, used by `method1.py`/`method2.py`.
- `input/*.csv`, `output/*.csv` — **gitignored** (only the folders themselves are tracked, via
  `input/.gitignore`/`output/.gitignore`); used by `method1_v2.py`. Populate `input/` yourself
  before running it.
- `distribucion_markers_1_rev1.json` — static ArUco marker layout (`id` → `position.{x,y,z}`,
  `rotation.{roll,pitch,yaw}`), used by all three scripts to plot marker positions relative to
  each estimate.
- `markers_info` column (in the source CSVs) holds a Java/Kotlin-style string of
  `PositionFromMarker(markerId=N, ...)` entries; all scripts parse marker IDs out of it with the
  same regex (`markerId=(\d+)`) rather than a structured parser.
