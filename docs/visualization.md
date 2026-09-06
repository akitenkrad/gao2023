# Visualization

The Python package `s3-tools` (module `s3_tools`) reads the Rust simulation outputs and renders figures. Install it once at the workspace root with `uv sync`, then call subcommands with `uv run s3-tools …`.

Reading a run directory is left to the `runvault` package (`runvault.read`). Omitting `--results_dir` / `--sweep_dir` targets whatever `runvault path --latest` returns, so `runvault` must be on `PATH`. Figures are made after a run has ended, so they go **outside** the run directory (`<results_root>/<experiment>/figures/<run_slug>/`) — `manifest.csv` is settled by `finish()`, so anything added to `artifacts/` afterwards would carry no hash.

## `visualize`

```bash
uv run s3-tools visualize [--results_dir RUN_DIR] [--output_dir DIR] [--no-graph]
```

Reads the run directory's `metrics.csv` (pivoting the long form into one row per round) and the `parameters` of its `config.json`, and writes to `<experiment>/figures/<run_slug>/`:

- `propagation_timeseries.png` — a 4-panel figure of the population-level propagation:
  - **attitude** — `attitude_positive_frac` over rounds (attitude propagation curve).
  - **emotion** — a stackplot of calm / moderate / intense fractions over rounds.
  - **behavior** — `behavior_adoption_rate` over rounds (adoption curve).
  - **cascade** — `info_cascade_size` over rounds (cumulative reach; monotone non-decreasing).
- `network_snapshot.png` — an *approximate* directed-graph snapshot (networkx, capped at 200 nodes) reconstructed from the `parameters` of `config.json` for a qualitative sense of the topology. It does not match the Rust graph bit-for-bit (different RNG stream); use `--no-graph` to skip it.

## `visualize-sweep`

```bash
uv run s3-tools visualize-sweep [--sweep_dir SWEEP_DIR] [--output_dir DIR]
```

Collects the sweep parent's child runs and rebuilds the one-row-per-trial table (`sweep_summary.csv` is no longer written), then writes to `<experiment>/figures/<run_slug>/`:

- `sweep_attitude_heatmap.png` — final positive-attitude fraction over the network × population grid (trial-averaged).
- `sweep_cascade_heatmap.png` — final info-cascade size over the grid.
- `sweep_metrics_vs_population.png` — attitude fraction / behavior adoption / cascade size vs population, one line per network kind.

## `show-experiment-settings`

```bash
uv run s3-tools show-experiment-settings [--results-dir RUN_DIR] [--json]
```

Reads the run directory's `config.json` under `parameters`, tells a run from a sweep by `run.json`'s `subcommand`, and prints a formatted summary. The LLM's model, provider and temperature come from `run.json`'s `llm` block; the call count and cache-hit rate come from the run-scope metrics in `metrics.csv`. Flat pre-runvault `config.json` / `sweep_config.json` are still readable. `--json` emits a machine-readable payload instead.

## `reproduce`

```bash
uv run s3-tools reproduce [--results_dir RUN_DIR] [--output_dir DIR]
```

Reads the run directory produced by `cargo run -- reproduce` (the band checks in `events.jsonl` plus the S³ and `baseline_*` series in `metrics.csv`), re-prints the observed-vs-paper checks (PASS / off) and the classical-baseline comparison, and writes two figures: `reproduce_propagation.png` (S³'s attitude / emotion / behavior / cascade time series) and `reproduce_comparison.png` (S³ vs LT / IC / Voter / DeGroot active fraction and cumulative reach). The Rust side already computed the verdicts; this tool only renders and re-summarises.

## Interpreting the outputs

Because the local LLM (`llama3.2:latest`) differs from the paper's GPT models, read the figures **qualitatively**:

- the **attitude** curve should move toward consensus or split, with a clear trend rather than noise;
- the **emotion** stackplot should show mass shifting between calm/moderate/intense as messages propagate;
- the **cascade** curve should be monotone non-decreasing and grow faster on BA (scale-free hubs) than on ER;
- in the comparison figure, the LLM-driven S³ curve sits among the classical baselines — typically spreading faster than IC and reaching levels comparable to LT / Voter / DeGroot once the topic activates;
- exact MSED / Cor matching to the paper's real-data series is out of reach for a local model, so `reproduce` checks qualitative bands (trend and sign), not the paper's exact numbers.
