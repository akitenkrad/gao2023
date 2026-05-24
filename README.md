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

The paper uses a directed follow-graph; this is now supported in socsim-net (`DiSocialNetwork`, issue #18). Convention: a directed edge **`A → B` means "A follows B"**, so B's post reaches B's **followers** = the in-neighbours of B (`in_neighbors(B)`). socsim-net's random generators only produce *undirected* topologies, so the build generates an undirected graph (ER / WS / BA) and then imposes follow-direction on each edge (≈25% `A→B`, ≈25% `B→A`, ≈50% mutual) to construct the directed graph. See [docs/architecture.md](docs/architecture.md).

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

## Scope

This repository currently implements **Phase 1** (the core directed-network LLM emotion/attitude/behavior contagion model, the two-layer LLM client with Ollama→OpenAI fallback + caching, the `run` subcommand, and population-level metrics) and **Phase 2** (the `sweep` over network × population, plus the Python `visualize` / `visualize-sweep` / `show-experiment-settings` tools). A one-shot paper reproduction (`reproduce`, Table 4/5 with real-data MSED / Cor alignment and LT / IC / Voter / DeGroot baselines) and an LLM-driven perception stage are left as future work (Phase 3); clean extension points are kept throughout.

## License

MIT

---
*This file was generated by Claude Code.*
