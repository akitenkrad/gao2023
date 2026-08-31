# CLI

The Rust binary `s3` has four subcommands: `run` (a single simulation), `sweep` (a grid over network × population), `reproduce` (a one-shot paper reproduction with observed-vs-paper checks and classical-baseline comparison), and `baseline` (a single classical diffusion / opinion-dynamics baseline).

## LLM environment variables

The LLM layer reads its provider configuration from the environment (never hard-coded):

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (tried first). |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model. |
| `OPENAI_API_KEY` | — | Enables the OpenAI fallback. If unset, only Ollama is used. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model (fallback). |

Provider order is fixed: **Ollama first → OpenAI fallback** (`socsim-llm`'s `FallbackClient`).

## `run`

```bash
cargo run --release -- run [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--network <er\|ws\|ba>` | `ba` | Network generator. BA/ER use socsim-net directed generators directly; WS builds an undirected topology then assigns directions. |
| `--population <N>` | `100` | Number of agents (nodes). |
| `--p <P>` | `0.05` | Erdős–Rényi (directed) connection probability. |
| `--ws-k <K>` | `4` | Watts–Strogatz initial degree (even). |
| `--ws-beta <BETA>` | `0.1` | Watts–Strogatz rewiring probability. |
| `--ws-p-mutual <P>` | `0.5` | WS only: probability an undirected edge becomes a mutual (bidirectional) follow in `to_directed` (ER/BA ignore this). |
| `--m <M>` | `3` | Barabási–Albert attachments per new node. |
| `--rounds <T>` | `20` | Propagation rounds (= engine ticks). |
| `--top-k <K>` | `3` | Number of important messages selected per agent in perception. |
| `--llm-perception` | off | Use LLM-driven perception: the candidate messages are listed to the LLM, which returns the most-relevant indices (see `prompts::perception_prompt`). Falls back to the rule-based score if the LLM returns no usable numbers. When off, the rule-based path runs with zero LLM perception calls and is bit-identical to before the flag existed. |
| `--seed-posters <N>` | `3` | Agents that always post in round 0 (seed the cascade). |
| `--tol <TOL>` | `1e-9` | Convergence threshold on the round-to-round change of the positive-attitude fraction. |
| `--seed <SEED>` | random | Core-layer seed (controls socsim only; the LLM layer is cache-determinised). |
| `--temperature <T>` | `0.0` | LLM generation temperature. |
| `--llm-seed <S>` | `0` | LLM generation seed (passed to the backend). |
| `--cache-path <PATH>` | `.llm_cache/cache.json` | Prompt→response cache file. |
| `--output-dir <DIR>` | `results` | Output base directory. |

Outputs are written to a runvault run directory. The run directory *is* the output location, so neither a timestamped subdirectory nor a `latest` symlink is created. Ask `runvault` for the path of the most recent finished run (a sweep child also has `subcommand=run`, so add `--standalone` to get a single hand-started one):

```bash
runvault path --experiment s3 --latest --subcommand run --standalone
```

```
results/
└── s3/                                             ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← the last run that finished
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<time>_<cfg8>_<exec4>
    │   ├── run.json                                 ← metadata (git commit / env / LLM / paper)
    │   ├── config.json                              ← envelope; the conditions live under ["parameters"]
    │   ├── metrics.csv                              ← long form (step / step_unit / scope / name / value)
    │   ├── status.json                              ← final state and duration
    │   └── manifest.csv                             ← hashes of artifacts/ and logs/
    └── figures/                                     ← written by the visualisation scripts (outside the run)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── propagation_timeseries.png
```

`metrics.csv` is long form, one value per row. The six per-round metrics (`attitude_positive_frac`, `emotion_calm`, `emotion_moderate`, `emotion_intense`, `behavior_adoption_rate`, `info_cascade_size`) carry a `step` with `step_unit=round`; the ones that describe the whole run with a single value (`converged` as 0.0 / 1.0, `final_round`, `llm_calls`, `llm_cache_hits`, `llm_cache_hit_rate`) sit at `scope=run` with no `step`. The LLM's model, provider and temperature live in the `llm` block of `run.json` (no `run_metadata.json` is written).

### Example (small, real Ollama)

```bash
OLLAMA_MODEL=llama3.2:latest \
cargo run --release -- run --network ba --population 20 --rounds 3 --seed 42
```

## `sweep`

```bash
cargo run --release -- sweep [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--network <list>` | `er,ws,ba` | Comma-separated network kinds. |
| `--population-values <list>` | `50,100,200` | Comma-separated population sizes. |
| `--p / --ws-k / --ws-beta / --ws-p-mutual / --m` | as `run` | Generator parameters (shared across the grid). |
| `--rounds <T>` | `20` | Propagation rounds. |
| `--top-k <K>` | `3` | Perception top-K. |
| `--seed-posters <N>` | `3` | Round-0 posters. |
| `--runs <N>` | `3` | Independent trials per condition. |
| `--tol <TOL>` | `1e-9` | Convergence threshold. |
| `--seed <SEED>` | `42` | Base seed; each trial derives an independent seed via `derive_seed(seed, [hash(network), population, run])`. |
| `--temperature / --llm-seed / --cache-path` | as `run` | LLM settings (the cache is shared across the grid to maximise the hit rate). |
| `--output-dir <DIR>` | `results` | Output base directory. |

A sweep is recorded as one parent run plus one child run per condition. The children sit beside the parent in the experiment directory rather than under it, and point at the parent through `lineage.parent_run_uid`. No one-row-per-trial summary CSV is written — the same values are in each child's `metrics.csv`.

```
results/
└── s3/
    ├── sweep_20260405_160827_48d033b7_ee20/   ← the parent; its parameters are the grid definition
    │   ├── run.json                            ← carries lineage.sweep_id; rng.master_seed is null
    │   └── config.json
    ├── run_20260405_160828_174916dd_955d/     ← a child; lineage.parent_run_uid = the parent's run_uid
    │   └── metrics.csv                         ← its master_seed is the derive_seed-derived seed
    └── ...
```

The parent's path comes from `runvault path --experiment s3 --latest --subcommand sweep`. `s3-tools visualize-sweep` takes that parent, collects the children and rebuilds the summary table.

### Example

```bash
cargo run --release -- sweep \
    --network er,ws,ba \
    --population-values 50,100,200 \
    --rounds 20 --runs 3 --seed 42
```

## `reproduce`

Runs S³ once, computes the headline propagation observables, checks them against the paper's qualitative bands (PASS / off), then runs the four classical baselines (LT / IC / Voter / DeGroot) on the **same directed graph and seed** for comparison.

```bash
cargo run --release -- reproduce [OPTIONS]
```

| Flag | Default | Meaning |
|---|---|---|
| `--network <er\|ws\|ba>` | `ba` | Network generator. |
| `--population <N>` | `200` | Number of agents. |
| `--p / --m` | `0.05` / `3` | ER probability / BA attachments. |
| `--rounds <T>` | `20` | Propagation rounds. |
| `--top-k <K>` | `3` | Perception top-K. |
| `--seed-posters <N>` | `3` | Round-0 posters (also the baselines' seed set). |
| `--llm-perception` | off | Use LLM-driven perception in the S³ run. |
| `--seed <SEED>` | `42` | Core-layer seed (shared by S³ and the baselines). |
| `--tol <TOL>` | `1e-12` | Convergence threshold (reproduce does not early-stop by default). |
| `--mock` | off | Run S³ with a scripted offline client (no live LLM). Use this for sandboxed / CI reproduction. |
| `--quick` | off | Smoke mode: caps population/rounds. |
| `--temperature / --llm-seed / --cache-path` | as `run` | LLM settings (live runs only; `--mock` uses an in-memory cache). |
| `--lt-theta <θ>` | `0.3` | Linear-Threshold uniform threshold. |
| `--ic-p <p>` | `0.15` | Independent-Cascade per-edge infection probability. |
| `--degroot-self-weight <w>` | `0.5` | DeGroot self-weight. |
| `--output-dir <DIR>` | `results` | Output base directory. |

The observables and their qualitative bands:

| Indicator | Direction | Band | Meaning |
|---|---|---|---|
| `attitude_rise` | `>=` | `0.05` | positive-attitude fraction `t=end − t=0`. |
| `cascade_growth_ratio` | `>=` | `1.5` | cascade size `end / initial`. |
| `behavior_adoption_final` | `>=` | `0.05` | final repost/post fraction. |
| `emotion_msed` | `<=` | `0.20` | mean-squared error between the final emotion distribution and the paper's reference (MSED proxy). |
| `attitude_corr` | `>=` | `0.30` | Pearson correlation of the attitude time series with a monotone-increasing ramp (Cor proxy). |

Because the local model differs from the paper's GPT, the targets are **qualitative** (trend and sign), not the paper's exact numbers.

Outputs are written to a runvault run directory (`subcommand=reproduce`). S³ and the four baselines share one run:

- `metrics.csv` — the S³ round-by-round population metrics, plus each baseline's round-by-round metrics under the names `baseline_{lt,ic,voter,degroot}_{active_frac,mean_opinion,cumulative_reached}` (five models in one run, so the name is what tells them apart). The headline observables (`attitude_rise` and friends) and `checks_passed` / `checks_total` sit at `scope=run` with no `step`.
- `events.jsonl` — one line per band check (`schema` is `x.gao2023.check`). The direction of the comparison and PASS / off are categories rather than numbers, so they go here. The band each observable is checked against is a qualitative anchor this replication chose, not a value the paper reports, so it is kept out of `reference.csv`, which demands a source.
- `config.json` — the conditions (under `parameters`).
- Figures are written by `s3-tools reproduce` into `<experiment>/figures/<run_slug>/`, outside the run.

### Example (offline)

```bash
cargo run --release -- reproduce --mock --quick --seed 42
uv run s3-tools reproduce
```

## `baseline`

Runs a single classical diffusion / opinion-dynamics model on the same directed follow-graph as S³ (zero LLM calls, bit-deterministic).

```bash
cargo run --release -- baseline --model <lt|ic|voter|degroot> [OPTIONS]
```

| Model | Dynamics |
|---|---|
| `lt` | Linear Threshold (Granovetter 1978 / KKT 2003): a node activates once the active fraction among the people it follows reaches θ (progressive). |
| `ic` | Independent Cascade (KKT 2003): a newly active node tries to infect each follower with probability `p`, once (progressive). |
| `voter` | Voter model (Clifford–Sudbury 1973): each round a node copies the binary opinion of a random followee (non-progressive). |
| `degroot` | DeGroot (1974): linear consensus — a node's continuous opinion moves toward the mean opinion of the people it follows. |

| Flag | Default | Meaning |
|---|---|---|
| `--model <...>` | `lt` | Baseline model. |
| `--network / --population / --p / --m / --rounds / --seed-posters / --seed / --tol` | as `reproduce` | Topology and run parameters (the directed graph matches S³ for the same seed). |
| `--lt-theta / --ic-p / --degroot-self-weight` | `0.3` / `0.15` / `0.5` | Model parameters. |
| `--output-dir <DIR>` | `results` | Output base directory. |

Outputs are written to a runvault run directory (`subcommand=baseline`). The per-round metrics in `metrics.csv` are the unprefixed `active_frac` / `mean_opinion` / `cumulative_reached`; which model they belong to is in `config.json` under `parameters.model`. No LLM is ever called, so `run.json` carries no `llm` block.

```bash
cargo run --release -- baseline --model ic --network ba --population 200 --rounds 20 --seed 42
```

---
*This file was generated by Claude Code.*
