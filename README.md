<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

**English** | [日本語](README.ja.md)

# S3: Social-network Simulation System with Large Language Model-Empowered Agents — Gao et al. (2023)

A reimplementation of the S³ model of Gao et al. (2023), "S3: Social-network Simulation System with Large Language Model-Empowered Agents" (arXiv:2307.14984). A population of LLM-driven agents sits on a **directed** social network (follow relationships); each round, agents perceive the most important messages from people they follow and an LLM jointly updates their **emotion** (calm / moderate / intense), **attitude** (negative / positive) and **interaction behavior** (repost / post / inactive). When an agent posts or reposts, the content reaches its followers in the next round, so individual-level updates give rise to **population-level propagation** of information, attitude and emotion. The deterministic [socsim](https://github.com/akitenkrad/rs-social-simulation-tools) core handles the directed network, message delivery, activation order and metrics; the non-deterministic LLM layer is confined to one mechanism and pseudo-determinised via the `socsim-llm` crate (prompt→response cache + `temperature=0` + fixed seed).

## Two-layer determinism (read this first)

LLM output is **outside** socsim's bit-reproducibility. The design therefore splits into two layers:

- **Deterministic socsim core** — directed-network generation, message delivery along follow edges, `RandomActivationScheduler` activation order (`ctx.rng`, ChaCha20), metrics and convergence. Given a seed this reproduces bit-for-bit.
- **Non-deterministic LLM layer** — the joint emotion/attitude/behavior decision and the generated post content. Pseudo-determinised by `socsim-llm`'s `CachingClient` (a `hash(prompt+model)` → response cache), `temperature=0` and a fixed seed. The provider order is **Ollama first → OpenAI fallback** via `socsim-llm`'s `FallbackClient`.

The cache — not the model — is the reproducibility mechanism: a warm cache replays identical responses, so a rerun is free and stable. Each run writes `run_metadata.json` recording the model, endpoint, temperature, seed and cache-hit rate. Because the local default model (`llama3.2:latest`) differs from the paper's GPT models, reproduction targets are **qualitative** (the trend and sign of the propagation curves: attitude fractions, emotion distribution, behavior adoption and cascade growth), not the paper's exact MSED / Cor numbers.

## Directed follow-graph

The paper uses a directed follow-graph; this is supported in socsim-net (`DiSocialNetwork`, issue #18), which since **issue #28** also ships directed generators. Convention: a directed edge **`A → B` means "A follows B"**, so B's post reaches B's **followers** = the in-neighbours of B (`in_neighbors(B)`). BA and ER are built directly with the directed generators (`barabasi_albert_directed` / `erdos_renyi_directed`); WS has no directed generator, so an undirected `watts_strogatz` graph is converted with `to_directed(p_mutual)` (default `0.5`, see `--ws-p-mutual`). See [docs/architecture.md](docs/architecture.md).

## Install & Quick start

```bash
# Build the Rust simulation (fetches socsim incl. socsim-llm with the Ollama+OpenAI backends)
cargo build --release

# Make sure a local Ollama is running and a model is pulled, e.g.:
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
# Optional OpenAI fallback:
#   export OPENAI_API_KEY=sk-...   OPENAI_MODEL=gpt-4o-mini

# Run a small simulation (BA directed follow-graph, 20 agents, 3 rounds)
cargo run --release -- run --network ba --population 20 --rounds 3 --seed 42

# Install the Python visualization tools (at the workspace root)
uv sync

# Visualize the most recent run (attitude / emotion / behavior / cascade time series + network snapshot)
uv run s3-tools visualize

# Inspect the run's settings and LLM metadata
uv run s3-tools show-experiment-settings --results-dir results/latest
```

### Offline smoke (no live LLM)

```bash
# Exercise the full pipeline with a mock LLM client (no network egress)
cargo run --release --example mock_smoke -- results
uv run s3-tools visualize
```

## Documentation

- [Use cases](docs/usecases.md) — what you can do with this project, with pointers to the rest of the docs.
- [CLI](docs/cli.md) — the Rust CLI: the `run` and `sweep` subcommands and their flags, plus the LLM environment variables.
- [Visualization](docs/visualization.md) — the Python `s3-tools` and how to interpret the outputs.
- [Architecture](docs/architecture.md) — repository structure, the directed follow-graph, the two-layer determinism, the socsim/`socsim-llm` framework, the mechanisms, the metrics, and references.

## Reproduction & classical baselines

`reproduce` runs S³ once and checks its headline propagation observables against the paper's qualitative bands (attitude rise, cascade growth, behavior adoption, an emotion-distribution MSED proxy, and a Pearson correlation proxy of the attitude time series), writing `reproduce_summary.json` with PASS / off verdicts. For comparison it runs four classical diffusion / opinion-dynamics baselines — **Linear Threshold (LT)**, **Independent Cascade (IC)**, **Voter** and **DeGroot** — on the same directed follow-graph and seed (zero LLM calls, bit-deterministic), and `s3-tools reproduce` overlays them against the LLM-driven S³ curves. The `--llm-perception` flag turns the perception step into a real LLM call (the LLM ranks the candidate messages); the default rule-based path makes zero perception calls. Because the local model differs from the paper's GPT, the reproduction targets are qualitative.

```bash
# Offline reproduction (no live LLM) + figures
cargo run --release -- reproduce --mock --quick --seed 42
uv run s3-tools reproduce --results-dir results/latest

# A single classical baseline on the same directed graph
cargo run --release -- baseline --model ic --network ba --population 200 --rounds 20 --seed 42
```

See [docs/cli.md](docs/cli.md) for the full `reproduce` / `baseline` flags and the observable bands, and [docs/architecture.md](docs/architecture.md) for how the baselines map onto the follow-edge convention.

## License

MIT

---
*This file was generated by Claude Code.*
