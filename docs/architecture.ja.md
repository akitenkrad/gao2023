# アーキテクチャ

本プロジェクトは Gao et al. (2023)「S3: Social-network Simulation System with Large Language Model-Empowered Agents」の再現実装である．Cargo + uv のモノレポ構成で，Rust クレットがシミュレーションを実行し，Python パッケージが結果を可視化する．**有向** ソーシャルネットワーク上の **LLM 駆動** 再現実装である．

## リポジトリ構成

```
gao2023/
├── Cargo.toml                  # [workspace] members = ["simulation"]
├── pyproject.toml              # uv workspace (members = ["tools"])
├── simulation/                 # Rust クレット `s3-simulation` (bin `s3`)
│   ├── Cargo.toml              # socsim git 依存: core / engine / net / llm (features=["live"])
│   ├── src/
│   │   ├── main.rs             # clap: run / sweep
│   │   ├── config.rs           # Config + 列挙型 (NetworkKind / LlmSettings)
│   │   ├── world.rs            # S3World (WorldState) + AgentState (感情/態度/行動/memory) + Profile + Message
│   │   ├── llm.rs              # 二層 LLM クライアントビルダ (Ollama→OpenAI フォールバック + キャッシュ)
│   │   ├── prompts.rs          # 社会的伝染プロンプト (感情/態度/行動 + コンテンツの同時生成)
│   │   ├── parse.rs            # 伝染応答を 3 更新 + コンテンツへパース
│   │   ├── perception.rs       # 影響スコア (時間減衰 + 関連性 + 真正性) + 上位 K 選択
│   │   ├── mechanisms.rs       # NetworkInit / LLMPerception / SocialContagion / PopulationMetrics
│   │   ├── metrics.rs          # 態度割合 / 感情分布 / 行動採用 / 情報カスケード規模
│   │   ├── simulation.rs       # init_world (有向グラフ構築) + run ドライバ + 出力
│   │   └── lib.rs              # テスト用モジュール公開
│   ├── examples/mock_smoke.rs  # オフライン (ネットワーク不要) スモーク実行
│   └── tests/integration_test.rs  # mock 駆動; ライブ LLM 不要
├── tools/                      # Python パッケージ `s3-tools` (module `s3_tools`)
│   └── src/s3_tools/{cli,visualize,visualize_sweep,show_experiment_settings}.py
├── docs/                       # 本ドキュメント (bilingual)
└── results/                    # 実行時出力 (gitignore)
```

## 有向フォローグラフ (issue #18, #28)

論文は有向フォローグラフを構築する．socsim-net は有向グラフに対応済み (`DiSocialNetwork = Network<(), Directed>`, issue #18) で `out_neighbors` / `in_neighbors` / `neighbors_directed` を持ち，**issue #28** で **有向生成器** も追加された．これにより「無向生成 → 方向付与」の回避策をやめ，フォローグラフを直接構築する．

- **規約:** 有向辺 `A → B` = 「A が B をフォロー」．B の投稿は B をフォローする人 (`* → B`) = `in_neighbors(B)` に届く．`S3World::followers(author) = net.in_neighbors(author)` であり，`NetworkInitMechanism` が各 `outbox` メッセージを `followers(author)` へ配送する．この規約は 3 種別すべてで同一．
- **構築 (`init_world` 内の `build_network`):** ネットワーク種別ごとに 1 経路で，いずれも `A → B` = 「A が B をフォロー」を生成する:
  - **BA** → `DiSocialNetwork::barabasi_albert_directed(ids, m, rng)`．各新規ノードが `m` 本の出弧 `new → target` を張り，target は (in-degree + 1) に比例して選ばれる．既にフォローされているノードが優先されるため **in-degree (= フォロワ数) が重い裾** を持ち，フォローグラフとして忠実．
  - **ER** → `DiSocialNetwork::erdos_renyi_directed(ids, p, rng)`．順序対 `(i, j)` ごとに独立に弧を引くため非対称になりうる．
  - **WS** → 有向生成器が無いので，無向 `watts_strogatz(ids, k, beta, rng)` を生成してから `.to_directed(p_mutual, rng)` を呼ぶ: 確率 `p_mutual` で相互 (双方向)，残りは RNG で片方向に倒す．`p_mutual` は config / `--ws-p-mutual` で公開 (既定 `0.5`)．

これにより設計の UNCONFIRMED note を解消する: 有向が **既定** (無向フォールバックは置かない)．従来の `impose_follow_direction` ヘルパは削除した．有向 BA/ER のトポロジは旧「無向＋方向付与」より忠実なため，指標の絶対値は変化する (フォロー方向の規約と `in_neighbors` 配送は不変)．

## 二層決定論

LLM は非決定的なので，1 層に閉じ込めて擬似決定論化する．

| 層 | 担当 | 再現性 |
|---|---|---|
| **決定論的 socsim コア** | 有向網生成 (BA/ER 有向生成器; WS は無向→`to_directed`)・メッセージ配送・`ctx.rng` による活性化順序・指標・収束 | seed を与えれば bit 単位 (ChaCha20 `SimRng` + `derive_seed`) |
| **非決定的 LLM レイヤ** | 感情/態度/行動の同時決定・投稿コンテンツ生成 | `socsim-llm` のプロンプト→応答キャッシュ + `temperature=0` + seed 固定で擬似決定論化 |

RNG ストリーム (コア層のみ):

- `derive_seed(root, &[0])` → world-init RNG (有向トポロジ生成 — WS の `to_directed` の相互/方向抽選を含む — ・属性/初期感情/初期態度割当)．
- `derive_seed(root, &[1])` → engine RNG (`RandomActivationScheduler` の毎ラウンドシャッフル)．

LLM レイヤは `SimRng` の支配外であり，再現性はキャッシュに由来する．`run_metadata.json` にモデル / endpoint / 温度 / seed / cache-hit 率を記録する．

## LLM クライアント (`socsim-llm`)

`socsim-llm` クレット (feature `live` = `ollama` + `openai`) が部品を提供し，`src/llm.rs` が合成する．設計当初の `reqwest` 案は本層で置換される．

```
CachingClient< Box<dyn LlmClient> >   // 型消去: FallbackClient< OllamaClient, OpenAiClient > (本番) | ScriptedClient (テスト)
```

- `FallbackClient` は primary (Ollama) を試し，**任意の** エラーで secondary (OpenAI) へフォールバックする (socsim-llm 提供; 自前実装しない)．
- `CachingClient` が `PromptCache` (`hash(prompt+model)` → 応答, JSON ファイル) を被せる．miss でキャッシュ更新するため `complete(&mut self, …)`．
- バックエンドは `Box<dyn LlmClient>` に型消去され，同一 `S3Client` 型が本番 `FallbackClient` とテスト `mock::ScriptedClient` の両方を運ぶ．`socsim-llm` が `impl LlmClient for Box<T>` (issue #26) を提供するため newtype 不要．
- `OllamaClient::from_env()` は `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL` (本プロジェクト既定 `llama3.2:latest`)，`OpenAiClient::from_env()` は `OPENAI_API_KEY` / `OPENAI_MODEL` を読む．

クライアントと `MetadataCollector` はメカニズムと run ドライバで `Rc<RefCell<…>>` 共有する (engine がメカニズムを所有するため)．実行後にドライバがキャッシュ統計を読み，キャッシュを保存する．

## WorldState とメカニズム

`S3World` は `socsim_net::DiSocialNetwork`，`BTreeMap<AgentId, AgentState>` (ソート済みキー → 決定論的 `agent_ids()`)，`inbox` (当該 round 配送分)，`outbox` (次 round 配送分)，`reached` 集合 (累積カスケード) を持つ．各 `AgentState` は `Profile` (性別/年齢/職業)・`Emotion`・`Attitude`・`Behavior`・`memory` (`Vec<Message>`) を持つ．`#[derive(Clone)]`．

**同期 round:** 1 engine tick = 1 round = 全エージェント 1 回更新．round 内で生成された投稿/転送は `outbox` に蓄積され，**次 round 先頭** (`NetworkInitMechanism`, `PreStep`) で配送される．mid-round の状態変化は同 round の他者へ波及しない．

メカニズム (6 フェーズループ; 宣言順 = 発火順):

| Mechanism | Phase | 役割 |
|---|---|---|
| `NetworkInitMechanism` | `PreStep` | inbox クリア; 前 round の `outbox` を著者のフォロワ (`in_neighbors`) へ配送; `reached` を更新 |
| `LLMPerceptionMechanism` | `Decision` | 各エージェントが inbox + memory から影響スコア (時間減衰 + 関連性 + 真正性) 上位 K を scratch へ選択．既定 **規則ベース**; `--llm-perception` は拡張スタブ |
| `SocialContagionMechanism` | `Interaction` | **中核 LLM 呼び出し**: 各エージェント 1 回，選択メッセージ + 属性 + 現状態から感情/態度/行動を同時更新; post/repost 時はコンテンツ生成して `outbox` へ |
| `PopulationMetricsMechanism` | `PostStep` | positive 態度割合を集計; round 間変化 `< tol` で `request_stop()` |

## 指標

各 round でエージェント集合から計算する (`metrics.rs`):

- **attitude_positive_frac** — positive 態度割合 (態度伝播; 論文 Table 5 / Opinion)．
- **emotion_dist** — calm / moderate / intense 割合 (感情伝播; 論文 Table 5 / Emotion)．
- **behavior_adoption_rate** — repost/post 割合 (行動採用曲線)．
- **info_cascade_size** — いずれかのメッセージに到達した累積ノード数 (情報カスケード; 論文 Table 4, vs LT / IC)．

実データ MSED / Cor 整合 (論文 Table 4/5) と LT / IC / Voter / DeGroot ベースラインは Phase 3 (`reproduce`) に委ねる．

## socsim フレームワーク

[socsim](https://github.com/akitenkrad/rs-social-simulation-tools) (ライブラリモード, git 依存, `branch = "main"`, `Cargo.lock` で固定):

- `socsim-core` — `WorldState` / `Mechanism` / `Phase` / `StepContext` / `Blackboard` / `AgentId` / `SimClock` / `SimRng` / `derive_seed`．
- `socsim-engine` — `SimulationBuilder`, `Simulation::run_observed`, `RandomActivationScheduler`．
- `socsim-net` — `DiSocialNetwork` (有向, issue #18) と `in_neighbors` / `out_neighbors`，**有向生成器** `erdos_renyi_directed` / `barabasi_albert_directed` と `SocialNetwork::to_directed` (issue #28)，加えて `SocialNetwork` と `watts_strogatz` 生成器．
- `socsim-llm` (`features = ["live"]`) — `LlmClient` / `OllamaClient` / `OpenAiClient` / `FallbackClient` / `CachingClient` / `PromptCache` / `LlmConfig` / `CallMetadata` / `MetadataCollector` / `mock::ScriptedClient`．

## 参考文献

- Gao, C., Lan, X., Lu, Z., Mao, J., Piao, J., Wang, H., Jin, D., & Li, Y. (2023). *S3: Social-network Simulation System with Large Language Model-Empowered Agents.* arXiv:2307.14984.
- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. (メモリプール / 生成エージェント)
- Chen, W., Yuan, Y., & Zhang, L. (2010). *Scalable Influence Maximization … under the Linear Threshold Model.* IEEE ICDM. (LT / IC ベースライン, Phase 3)
- DeGroot, M. H. (1974). *Reaching a Consensus.* JASA 69(345). (DeGroot ベースライン, Phase 3)

---
*This file was generated by Claude Code.*
