# CLI

Rust バイナリ `s3` には 4 つのサブコマンドがある: `run` (単一実行)，`sweep` (network × population のグリッド)，`reproduce` (observed-vs-paper 照合 + 古典ベースライン比較の一括再現)，`baseline` (古典的拡散・意見ダイナミクスのベースライン単独実行)．

## LLM 環境変数

LLM レイヤはプロバイダ設定を環境変数から読む (ハードコードしない):

| 変数 | 既定 | 意味 |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint (第一候補)． |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama モデル． |
| `OPENAI_API_KEY` | — | OpenAI フォールバックを有効化．未設定なら Ollama のみ． |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI モデル (フォールバック)． |

プロバイダ順序は固定: **Ollama 第一 → OpenAI フォールバック** (`socsim-llm` の `FallbackClient`)．

## `run`

```bash
cargo run --release -- run [OPTIONS]
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--network <er\|ws\|ba>` | `ba` | ネットワーク生成器．BA/ER は socsim-net の有向生成器を直接使用，WS は無向トポロジを生成してから方向を付与． |
| `--population <N>` | `100` | エージェント数 (ノード数)． |
| `--p <P>` | `0.05` | Erdős–Rényi (有向) 接続確率． |
| `--ws-k <K>` | `4` | Watts–Strogatz 初期次数 (偶数)． |
| `--ws-beta <BETA>` | `0.1` | Watts–Strogatz 再配線確率． |
| `--ws-p-mutual <P>` | `0.5` | WS のみ: `to_directed` で無向辺が相互 (双方向) フォローになる確率 (ER/BA では無視)． |
| `--m <M>` | `3` | Barabási–Albert 新規ノードあたり結合数． |
| `--rounds <T>` | `20` | 伝播ラウンド数 (= engine tick)． |
| `--top-k <K>` | `3` | Perception で選ぶ重要メッセージ件数． |
| `--llm-perception` | off | LLM 駆動 Perception を使う: 候補メッセージを LLM に列挙提示し，関連順の番号列を受け取る (`prompts::perception_prompt`)．LLM が有効な番号を返さなければ規則ベーススコアへフォールバックする．off のとき規則ベース経路は Perception の LLM 呼び出し 0 回でフラグ追加前と bit 等価． |
| `--seed-posters <N>` | `3` | round 0 で必ず投稿するエージェント数 (カスケードの種)． |
| `--tol <TOL>` | `1e-9` | positive 態度割合の round 間変化に対する収束しきい値． |
| `--seed <SEED>` | random | コア層 seed (socsim のみ支配; LLM 層はキャッシュ決定論)． |
| `--temperature <T>` | `0.0` | LLM 生成温度． |
| `--llm-seed <S>` | `0` | LLM 生成 seed (バックエンドへ)． |
| `--cache-path <PATH>` | `.llm_cache/cache.json` | プロンプト→応答キャッシュファイル． |
| `--output-dir <DIR>` | `results` | 出力ベースディレクトリ． |

出力は runvault の run ディレクトリへ．run ディレクトリが出力先そのものなので，タイムスタンプ付きサブディレクトリも `latest` symlink も作らない．直近の完了 run のパスは `runvault` に聞く (sweep の子も `subcommand=run` なので，手で回した 1 本を掴むには `--standalone` を付ける):

```bash
runvault path --experiment s3 --latest --subcommand run --standalone
```

```
results/
└── s3/                                             ← experiment
    ├── latest_finished -> run_20260405_153000_...   ← 最後に完了した run
    ├── run_20260405_153000_9f2c41ab_3b1d/           ← <subcommand>_<時刻>_<cfg8>_<exec4>
    │   ├── run.json                                 ← メタデータ (git commit / 環境 / LLM / 論文情報)
    │   ├── config.json                              ← 封筒．実験条件は ["parameters"] の下
    │   ├── metrics.csv                              ← long 形式 (step / step_unit / scope / name / value)
    │   ├── status.json                              ← 終了状態と所要時間
    │   └── manifest.csv                             ← artifacts/ と logs/ のハッシュ
    └── figures/                                     ← 可視化スクリプトの出力 (run の外)
        └── run_20260405_153000_9f2c41ab_3b1d/
            └── propagation_timeseries.png
```

`metrics.csv` は 1 行 1 値の long 形式．ラウンドごとの 6 指標 (`attitude_positive_frac` / `emotion_calm` / `emotion_moderate` / `emotion_intense` / `behavior_adoption_rate` / `info_cascade_size`) は `step_unit=round` の `step` を持ち，run 全体を 1 つの値で表す `converged` (0.0 / 1.0) / `final_round` / `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` は `scope=run` で `step` を持たない．LLM のモデル・provider・温度は `run.json` の `llm` ブロックにある (`run_metadata.json` は書かれない)．

### 実行例 (小規模, 実 Ollama)

```bash
OLLAMA_MODEL=llama3.2:latest \
cargo run --release -- run --network ba --population 20 --rounds 3 --seed 42
```

## `sweep`

```bash
cargo run --release -- sweep [OPTIONS]
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--network <list>` | `er,ws,ba` | カンマ区切りのネットワーク種別． |
| `--population-values <list>` | `50,100,200` | カンマ区切りの人口規模． |
| `--p / --ws-k / --ws-beta / --ws-p-mutual / --m` | `run` と同じ | 生成器パラメータ (グリッド共通)． |
| `--rounds <T>` | `20` | 伝播ラウンド数． |
| `--top-k <K>` | `3` | Perception top-K． |
| `--seed-posters <N>` | `3` | round 0 投稿者数． |
| `--runs <N>` | `3` | 各条件あたりの独立試行数． |
| `--tol <TOL>` | `1e-9` | 収束しきい値． |
| `--seed <SEED>` | `42` | 基点 seed; 各試行は `derive_seed(seed, [hash(network), population, run])` で独立化． |
| `--temperature / --llm-seed / --cache-path` | `run` と同じ | LLM 設定 (キャッシュはグリッド共有でヒット率を最大化)． |
| `--output-dir <DIR>` | `results` | 出力ベースディレクトリ． |

sweep は「親 run 1 本 + 条件ごとの子 run」として記録される．子は親の下ではなく experiment ディレクトリの兄弟として並び，`lineage.parent_run_uid` で親を指す．1 行 1 試行のサマリ CSV は書かない (同じ値は各子 run の `metrics.csv` にある)．

```
results/
└── s3/
    ├── sweep_20260405_160827_48d033b7_ee20/   ← 親．parameters が格子の定義
    │   ├── run.json                            ← lineage.sweep_id を持つ．rng.master_seed は null
    │   └── config.json
    ├── run_20260405_160828_174916dd_955d/     ← 子．lineage.parent_run_uid = 親の run_uid
    │   └── metrics.csv                         ← master_seed は derive_seed で作った派生シード
    └── ...
```

親のパスは `runvault path --experiment s3 --latest --subcommand sweep` で取れる．`s3-tools visualize-sweep` はこの親を受け取り，子 run を集めて従来のサマリ表を組み直す．

### 実行例

```bash
cargo run --release -- sweep \
    --network er,ws,ba \
    --population-values 50,100,200 \
    --rounds 20 --runs 3 --seed 42
```

## `reproduce`

S³ を 1 回実行して headline 伝播観測量を計算し，論文の定性帯と照合 (PASS / off) したうえで，**同一の有向網・同一シード**で 4 つの古典ベースライン (LT / IC / Voter / DeGroot) を実行して比較する．

```bash
cargo run --release -- reproduce [OPTIONS]
```

| フラグ | 既定 | 意味 |
|---|---|---|
| `--network <er\|ws\|ba>` | `ba` | ネットワーク種別． |
| `--population <N>` | `200` | エージェント数． |
| `--p / --m` | `0.05` / `3` | ER 確率 / BA 結合数． |
| `--rounds <T>` | `20` | 伝播ラウンド数． |
| `--top-k <K>` | `3` | Perception top-K． |
| `--seed-posters <N>` | `3` | round 0 投稿者数 (ベースラインの種集合と同一)． |
| `--llm-perception` | off | S³ 実行で LLM 駆動 Perception を使う． |
| `--seed <SEED>` | `42` | コア層 seed (S³ とベースラインで共有)． |
| `--tol <TOL>` | `1e-12` | 収束しきい値 (reproduce は既定で早期停止しない)． |
| `--mock` | off | S³ を scripted オフラインクライアントで実行 (live LLM 不要; サンドボックス / CI 用)． |
| `--quick` | off | スモーク (population/rounds を抑える)． |
| `--temperature / --llm-seed / --cache-path` | `run` と同じ | LLM 設定 (live のみ; `--mock` は in-memory キャッシュ)． |
| `--lt-theta <θ>` | `0.3` | Linear Threshold の一様しきい値． |
| `--ic-p <p>` | `0.15` | Independent Cascade の 1 辺感染確率． |
| `--degroot-self-weight <w>` | `0.5` | DeGroot の自己重み． |
| `--output-dir <DIR>` | `results` | 出力ベースディレクトリ． |

観測量と定性帯:

| 指標 | 向き | 帯 | 意味 |
|---|---|---|---|
| `attitude_rise` | `>=` | `0.05` | positive 態度割合の `t=end − t=0`． |
| `cascade_growth_ratio` | `>=` | `1.5` | カスケード規模 `end / initial`． |
| `behavior_adoption_final` | `>=` | `0.05` | 終端の repost/post 割合． |
| `emotion_msed` | `<=` | `0.20` | 終端感情分布と論文参照分布の平均二乗誤差 (MSED 代理)． |
| `attitude_corr` | `>=` | `0.30` | 態度時系列と単調増加ランプの Pearson 相関 (Cor 代理)． |

ローカルモデルは論文の GPT と異なるため，目標は **定性的** (傾向・符号) であり論文の絶対値ではない．

出力は runvault の run ディレクトリ (`subcommand=reproduce`) へ．S³ と 4 ベースラインは 1 本の run に同居する:

- `metrics.csv` — S³ の round 別集団指標と，`baseline_{lt,ic,voter,degroot}_{active_frac,mean_opinion,cumulative_reached}` という名前の各ベースラインの round 別指標 (5 モデルが同じ run に入るので，名前でモデルを分ける)．headline 観測量 (`attitude_rise` など) と `checks_passed` / `checks_total` は `scope=run` で `step` を持たない．
- `events.jsonl` — 帯照合 1 件が 1 行 (`schema` は `x.gao2023.check`)．比較の向きと PASS / off はカテゴリであって数ではないのでここに置く．照合先の帯はこの再現実装が置いた定性的なアンカーで論文の報告値ではないため，出典を要求する `reference.csv` には書かない．
- `config.json` — 実験条件 (`parameters` の下)．
- 図は `s3-tools reproduce` が run の外の `<experiment>/figures/<run_slug>/` へ生成する．

### 実行例 (オフライン)

```bash
cargo run --release -- reproduce --mock --quick --seed 42
uv run s3-tools reproduce
```

## `baseline`

S³ と同一の有向フォローグラフ上で，古典的拡散・意見ダイナミクスを単独実行する (LLM 呼び出し 0 回・bit 決定論的)．

```bash
cargo run --release -- baseline --model <lt|ic|voter|degroot> [OPTIONS]
```

| モデル | ダイナミクス |
|---|---|
| `lt` | Linear Threshold (Granovetter 1978 / KKT 2003): フォロー先のアクティブ割合が θ に達するとアクティブ化 (進行性)． |
| `ic` | Independent Cascade (KKT 2003): 新規アクティブノードが各フォロワを確率 `p` で 1 回感染試行 (進行性)． |
| `voter` | Voter model (Clifford–Sudbury 1973): 毎 round ランダムなフォロー先の二値意見をコピー (非進行性)． |
| `degroot` | DeGroot (1974): フォロー先意見の平均へ連続意見を更新する線形合意． |

| フラグ | 既定 | 意味 |
|---|---|---|
| `--model <...>` | `lt` | ベースラインモデル． |
| `--network / --population / --p / --m / --rounds / --seed-posters / --seed / --tol` | `reproduce` と同じ | トポロジ・実行パラメータ (同一 seed で S³ と同一網)． |
| `--lt-theta / --ic-p / --degroot-self-weight` | `0.3` / `0.15` / `0.5` | モデルパラメータ． |
| `--output-dir <DIR>` | `results` | 出力ベースディレクトリ． |

出力は runvault の run ディレクトリ (`subcommand=baseline`) へ．`metrics.csv` の round 別指標は接頭辞なしの `active_frac` / `mean_opinion` / `cumulative_reached` で，どのモデルかは `config.json` の `parameters.model` が持つ．LLM を 1 度も呼ばないので `run.json` に `llm` ブロックは付かない．

```bash
cargo run --release -- baseline --model ic --network ba --population 200 --rounds 20 --seed 42
```

---
*This file was generated by Claude Code.*
