# Architecture

This project replicates Gao et al. (2023), "S3: Social-network Simulation System with Large Language Model-Empowered Agents". It is a Cargo + uv monorepo: a Rust crate runs the simulation, and a Python package visualizes the results. It is an **LLM-driven** replication on a **directed** social network.

## Repository structure

```
gao2023/
├── Cargo.toml                  # [workspace] members = ["simulation"]
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust crate `s3-simulation` (bin `s3`)
│   ├── Cargo.toml              # socsim git deps: core / engine / net / llm (features=["live"])
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── config.rs           # Config + enums (NetworkKind / LlmSettings)
│   │   ├── world.rs            # S3World (WorldState) + AgentState (emotion / attitude / behavior / memory) + Profile + Message
│   │   ├── llm.rs              # two-layer LLM client builder (Ollama→OpenAI fallback + cache)
│   │   ├── prompts.rs          # contagion prompt (joint emotion/attitude/behavior + content)
│   │   ├── parse.rs            # parse the LLM contagion response into the 3 updates + content
│   │   ├── perception.rs       # influence score (time decay + relevance + authenticity) + top-K selection
│   │   ├── mechanisms.rs       # NetworkInit / LLMPerception / SocialContagion / PopulationMetrics
│   │   ├── metrics.rs          # attitude fraction / emotion dist / behavior adoption / info-cascade size
│   │   ├── simulation.rs       # init_world (directed graph build) + run driver + output writers
│   │   └── lib.rs              # module exports for tests
│   ├── examples/mock_smoke.rs  # offline (no-network) smoke run for CI / sandboxes
│   └── tests/integration_test.rs  # mock-driven; needs no live LLM
├── tools/                      # Python package `s3-tools` (module `s3_tools`)
│   └── src/s3_tools/{cli,visualize,visualize_sweep,show_experiment_settings}.py
├── docs/                       # this documentation (bilingual)
└── results/                    # runtime output (gitignored)
```

## Directed follow-graph (issues #18, #28)

The paper builds a directed follow-graph. socsim-net supports directed graphs (`DiSocialNetwork = Network<(), Directed>`, issue #18) with `out_neighbors` / `in_neighbors` / `neighbors_directed`, and since **issue #28** it also ships **directed generators**, so the follow-graph is built directly instead of via an undirected-then-impose-direction workaround.

- **Convention:** a directed edge `A → B` means **"A follows B"**. A post by `B` should reach the people who follow `B`, i.e. the nodes `* → B`, which are exactly `in_neighbors(B)`. `S3World::followers(author) = net.in_neighbors(author)`, and `NetworkInitMechanism` delivers each `outbox` message to `followers(author)`. This convention is identical for all three network types.
- **Construction (`build_network` in `init_world`):** one path per network type, all producing `A → B` = "A follows B":
  - **BA** → `DiSocialNetwork::barabasi_albert_directed(ids, m, rng)`. Each new node emits `m` out-arcs `new → target`, with targets chosen ∝ (in-degree + 1). Because attachment favours already-followed nodes, **in-degree (= follower count) is heavy-tailed**, the faithful follow-graph shape.
  - **ER** → `DiSocialNetwork::erdos_renyi_directed(ids, p, rng)`. Each ordered pair `(i, j)` draws an arc independently, so the graph may be asymmetric.
  - **WS** → no directed generator exists, so generate an undirected `watts_strogatz(ids, k, beta, rng)` and call `.to_directed(p_mutual, rng)`: with probability `p_mutual` an edge becomes **mutual** (both arcs), otherwise the RNG keeps one direction. `p_mutual` is exposed via config / `--ws-p-mutual` (default `0.5`).

This resolves the design's UNCONFIRMED note: directed is the **default** (no undirected fallback). The previous `impose_follow_direction` helper is removed; the directed BA/ER topologies are more faithful than the old undirected+imposed graph, so absolute metric values differ (the follow-direction convention and `in_neighbors` delivery are unchanged).

## Two-layer determinism

An LLM is non-deterministic, so it is confined to one layer and pseudo-determinised.

| Layer | What it owns | Reproducibility |
|---|---|---|
| **Deterministic socsim core** | directed-network generation (BA/ER directed generators; WS undirected→`to_directed`), message delivery, activation order via `ctx.rng`, metrics, convergence | bit-for-bit given the seed (ChaCha20 `SimRng` + `derive_seed`) |
| **Non-deterministic LLM layer** | joint emotion/attitude/behavior decision, generated post content | pseudo-determinised by `socsim-llm`'s prompt→response cache + `temperature=0` + fixed `seed` |

RNG streams (core layer only):

- `derive_seed(root, &[0])` → world-init RNG (directed topology generation — including `to_directed`'s mutual/direction draws for WS — and profile / initial-emotion / initial-attitude assignment).
- `derive_seed(root, &[1])` → engine RNG (`RandomActivationScheduler` shuffle each round).

The LLM layer is **not** under `SimRng`. Its reproducibility comes entirely from the cache: with a warm cache, an identical prompt replays an identical response. `run_metadata.json` records model / endpoint / temperature / seed / cache-hit rate.

## The LLM client (`socsim-llm`)

The `socsim-llm` crate (feature `live` = `ollama` + `openai`) provides the building blocks; `src/llm.rs` composes them — the design's original `reqwest` plan is superseded by this layer:

```
CachingClient< Box<dyn LlmClient> >   // erased: FallbackClient< OllamaClient, OpenAiClient > (prod) | ScriptedClient (tests)
```

- `FallbackClient` tries the primary (Ollama) and, on **any** error, falls back to the secondary (OpenAI). Provided by `socsim-llm` — not hand-rolled.
- `CachingClient` wraps it with a `PromptCache` (`hash(prompt+model)` → response, JSON-file-backed). `complete(&mut self, …)` takes a mutable borrow because a miss updates the cache.
- The backend is type-erased to `Box<dyn LlmClient>`, so the same `S3Client` type carries either the live `FallbackClient` (production) or a `mock::ScriptedClient` (tests / `mock_smoke`). `socsim-llm` implements `LlmClient` for `Box<T>` (issue #26), so no local newtype is needed.
- `OllamaClient::from_env()` reads `OLLAMA_HOST` (default `http://localhost:11434`) / `OLLAMA_MODEL` (this project's default is `llama3.2:latest`). `OpenAiClient::from_env()` reads `OPENAI_API_KEY` / `OPENAI_MODEL`.

The client and a `MetadataCollector` are shared between the mechanisms and the run driver via `Rc<RefCell<…>>`, because the engine owns the boxed mechanisms; after the run the driver reads the cache stats and saves the cache.

## WorldState and the mechanisms

`S3World` holds a `socsim_net::DiSocialNetwork`, a `BTreeMap<AgentId, AgentState>` (sorted keys → deterministic `agent_ids()`), an `inbox` (messages delivered this round), an `outbox` (posts to deliver next round) and a `reached` set (cumulative cascade). Each `AgentState` carries a `Profile` (gender / age / occupation), an `Emotion`, an `Attitude`, a `Behavior` and a `memory` (`Vec<Message>`). `#[derive(Clone)]` supports snapshotting.

**Synchronous rounds:** 1 engine tick = 1 simulation round = every agent updated once. Posts/reposts generated during a round are accumulated in `outbox` and delivered at the **start of the next round** (`NetworkInitMechanism`, `PreStep`), so mid-round state changes do not propagate to other agents within the same round.

Mechanisms (six-phase loop, declaration order = fire order):

| Mechanism | Phase | Role |
|---|---|---|
| `NetworkInitMechanism` | `PreStep` | clear inboxes; deliver the previous round's `outbox` to each author's followers (`in_neighbors`); grow the `reached` cascade set. |
| `LLMPerceptionMechanism` | `Decision` | for each agent, select the top-K messages from inbox + memory by influence score (time decay + relevance + authenticity) into scratch. **Rule-based by default**; `--llm-perception` is a kept extension stub. |
| `SocialContagionMechanism` | `Interaction` | the **core LLM call**: per agent (one call), given the selected messages + profile + current state, jointly update emotion / attitude / behavior; on post/repost, generate content and enqueue it to `outbox`. |
| `PopulationMetricsMechanism` | `PostStep` | aggregate the positive-attitude fraction; `request_stop()` when its round-to-round change `< tol`. |

## Metrics

Computed every round over the agent map (see `metrics.rs`):

- **attitude_positive_frac** — fraction of agents holding a positive attitude (attitude propagation; paper Table 5 / Opinion).
- **emotion_dist** — fractions of calm / moderate / intense (emotion propagation; paper Table 5 / Emotion).
- **behavior_adoption_rate** — fraction of agents whose behavior is repost/post (information-action / behavior adoption curve).
- **info_cascade_size** — cumulative number of nodes reached by any message (information cascade; paper Table 4, vs LT / IC).

Real-data MSED / Cor alignment (paper Table 4/5) and the LT / IC / Voter / DeGroot baselines are deferred to Phase 3 (`reproduce`).

## socsim framework

[socsim](https://github.com/akitenkrad/rs-social-simulation-tools) (library mode, git dependency, `branch = "main"`, pinned by `Cargo.lock`):

- `socsim-core` — `WorldState` / `Mechanism` / `Phase` / `StepContext` / `Blackboard` / `AgentId` / `SimClock` / `SimRng` / `derive_seed`.
- `socsim-engine` — `SimulationBuilder`, `Simulation::run_observed`, `RandomActivationScheduler`.
- `socsim-net` — `DiSocialNetwork` (directed, issue #18) with `in_neighbors` / `out_neighbors`, the **directed generators** `erdos_renyi_directed` / `barabasi_albert_directed` and `SocialNetwork::to_directed` (issue #28), plus `SocialNetwork` and its `watts_strogatz` generator.
- `socsim-llm` (`features = ["live"]`) — `LlmClient` / `OllamaClient` / `OpenAiClient` / `FallbackClient` / `CachingClient` / `PromptCache` / `LlmConfig` / `CallMetadata` / `MetadataCollector` / `mock::ScriptedClient`.

## References

- Gao, C., Lan, X., Lu, Z., Mao, J., Piao, J., Wang, H., Jin, D., & Li, Y. (2023). *S3: Social-network Simulation System with Large Language Model-Empowered Agents.* arXiv:2307.14984.
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. (memory pool / generative agents)
- Chen, W., Yuan, Y., & Zhang, L. (2010). *Scalable Influence Maximization … under the Linear Threshold Model.* IEEE ICDM. (LT / IC baselines, Phase 3)
- DeGroot, M. H. (1974). *Reaching a Consensus.* JASA 69(345). (DeGroot baseline, Phase 3)

---
*This file was generated by Claude Code.*
