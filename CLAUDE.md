# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Research scripts for a doctoral project. Positions of a moving object are estimated by some
positioning system (originally ArUco markers from video; also used with BLE RSSI-based position
estimates) with noise/jitter. The object actually moves along a known straight line at constant
(or near-constant) velocity. These scripts measure how far the noisy estimated positions deviate
from that known straight-line, constant-velocity ground truth. There is no application code,
tests, or build — just independent, run-as-a-script analyses that read a CSV, do a numeric
fit/calculation, write a CSV, and pop up a matplotlib plot.

## Running

No virtualenv/requirements file exists yet. Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`.

```
python method1_least_squares_fixed_endpoints.py        # reads samples.csv (repo root), writes optimized_positions.csv
python method1_v2_least_squares_batch.py                # reads every CSV in input/, writes matching files to output/
python method2_needleman_wunsch_alignment.py            # reads samples.csv (repo root), writes optimized_positions_method2.csv
python method3_constant_velocity_time_estimation.py     # reads every CSV in input/, writes output/<name>_temporal.csv
```

Each script loops over parameter combinations (or input files) and calls `plt.show()` per
iteration — a plot window pops up and **blocks execution** until closed.

## The four scripts — two fundamentally different error models

`method1*`/`method2` fit each estimated point to whatever position on the known line best
explains it (least-squares projection, or Needleman-Wunsch sequence alignment) — this leaves the
along-line component free, so they can only detect error **perpendicular** to the line
(cross-track). `method3` does not fit anything: it computes where the point *should* be from
elapsed time and an assumed constant velocity, so its error captures **both** the along-line and
perpendicular components. See `README.md` for the full derivation (in Spanish) — it's the primary
reference for this repo's methodology; this file is a condensed map of the code.

`method1`/`method1_v2` fit points onto the known straight line of real motion, but differ in how
correspondence/optimization is set up:

- **`method1_least_squares_fixed_endpoints.py`** — original version. Hardcodes `samples.csv` as input (`;`-separated,
  `,`-decimal). Iterates over every combination of `sampleSpaceMillis` (`[0, 1000, 2000]`) and
  `multipleMarkersBehaviour` (`CLOSEST/WEIGHTED_AVERAGE/AVERAGE/WEIGHTED_MEDIAN/MEDIAN`) found in
  the CSV, filtering rows for each combo. The real line's endpoints are taken from that combo's
  first/last `realX/realY` values. Least-squares fit (`scipy.optimize.minimize`, L-BFGS-B) finds
  each point's scalar position along the line; **endpoints are pinned** to `s=0`/`s=length` and
  not optimized. Plots the trajectory with markers (from `distribucion_markers_1_rev1.json`) drawn
  per frame. Appends results across all combos into one `optimized_positions.csv`.

- **`method1_v2_least_squares_batch.py`** — generalized version of Método 1 for arbitrary experiments. Reads
  **every** `*.csv` in `input/` (`,`-separated, `.`-decimal — different convention from
  method1/2's `samples.csv`), one output file per input file written to `output/` with the same
  filename. The real line's endpoints (`P_real`) are **hardcoded at the top of the file**
  (currently `[3.733, 0.691] → [0.693, 0.882]`) instead of read from the data — update this
  constant when the ground-truth line changes between experiments. Unlike `method1_least_squares_fixed_endpoints.py`, **all**
  points (including the first/last) are free in the optimization — none are pinned. Output CSVs
  preserve all original input columns plus `alineatedRealX`/`alineatedRealY` and
  `errorX`/`errorY`/`euclideanError` (distance from the raw estimate to its projection onto the
  line — cross-track error only, see the "two error models" note above).

- **`method2_needleman_wunsch_alignment.py`** — same `samples.csv` input, but aligns sequences instead of doing a per-point
  least-squares projection: it runs a **Needleman-Wunsch** global sequence alignment (similarity =
  negative Euclidean distance) between the ArUco estimates of each combo and a single fixed
  reference — the real interpolated trajectory at `sampleSpaceMillis=0` for the first behaviour in
  `multiple_markers_behaviour_set`. It then saves the *aligned reference real positions*
  (`optimized_positions_method2.csv`), i.e. this method resamples/matches the ground truth to the
  estimate's timing rather than projecting estimates onto a line.

- **`method3_constant_velocity_time_estimation.py`** — the odd one out: no optimizer, no
  alignment. For each file in `input/` it takes the first/last row as `P0`/`Pn` and their
  timestamps as `t0`/`t1`, then for every row computes `alpha=(timestamp-t0)/(t1-t0)` and
  `expected = P0 + alpha*(Pn-P0)` — the same formula that produces `realX`/`realY` in
  `samples.csv` (see README), but computed explicitly here instead of arriving pre-baked in the
  input. Saves `expectedX/Y`, `errorX/Y`, `euclideanError` to `output/<name>_temporal.csv` and
  prints a summary (mean/median/std/max). Only needs `timestamp`, `rawX`, `rawY` in the input CSV.

When comparing scripts, note `method1_v2_least_squares_batch.py`/`method3_constant_velocity_time_estimation.py`
use a different CSV dialect (`,`/`.`) and input/output layout (`input/`, `output/`) than
`method1_least_squares_fixed_endpoints.py`/`method2_needleman_wunsch_alignment.py` (`;`/`,`, repo root).

## Data files

- `samples.csv`, `optimized_positions.csv`, `optimized_positions_method2.csv` — tracked in git,
  live at repo root, used by `method1_least_squares_fixed_endpoints.py`/`method2_needleman_wunsch_alignment.py`.
- `input/*.csv`, `output/*.csv` — **gitignored** (only the folders themselves are tracked, via
  `input/.gitignore`/`output/.gitignore`); used by `method1_v2_least_squares_batch.py` (output: same
  filename) and `method3_constant_velocity_time_estimation.py` (output: `<name>_temporal.csv`).
  Populate `input/` yourself before running either.
- `distribucion_markers_1_rev1.json` — static ArUco marker layout (`id` → `position.{x,y,z}`,
  `rotation.{roll,pitch,yaw}`), loaded by `method1_least_squares_fixed_endpoints.py`,
  `method1_v2_least_squares_batch.py` and `method2_needleman_wunsch_alignment.py`, but only
  actually plotted by `method1_least_squares_fixed_endpoints.py` — in the other two it's loaded/parsed
  and then unused (dead code left over from copying method1). `method3` doesn't touch it at all.
- `markers_info` column (in the source CSVs) holds a Java/Kotlin-style string of
  `PositionFromMarker(markerId=N, ...)` entries; method1/method1_v2/method2 parse marker IDs out
  of it with the same regex (`markerId=(\d+)`) rather than a structured parser. `method3` doesn't
  require this column.
