//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` / `baseline` の
//! どのサブコマンドでも同一なので，ここ 1 箇所で組み立てる．集団指標の long 形式へ
//! の落とし方と，`reproduce` の帯照合の書き方もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};

use crate::baseline::{BaselineMetrics, BaselineModel};
use crate::metrics::Metrics;
use crate::reproduce::Check;
use crate::simulation::SimulationResult;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "s3";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "gao2023";
/// 分野．`simulation` を名乗ると `master_seed` が必須になる．
///
/// LLM で駆動されるモデルだが `llm-safety` ではない — 測っているのはモデルの
/// 安全性ではなく，有向網上の集団的伝播だからである．LLM 側の同一性は `llm`
/// ブロック ([`llm_block`]) が持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは 1 伝播ラウンドなので語彙の `round` を使う．
const T_UNIT: &str = "round";

/// 指標の粒度．集団指標はどれも母集団全体の集約なので `run`．
const SCOPE: &str = "run";

/// この再現実験が対象としている論文．
///
/// どのサブコマンドも同じ主張を対象とする — `baseline` は古典モデル単独の実行だが，
/// その存在理由は S³ と同一網・同一シードで並べることなので，同じ target に属する．
/// 論文は図表ではなく «局所更新から集団レベルの伝播が創発する» という主張の再現を
/// 狙うので，`Target::claim` を使う．
pub fn replication() -> Replication {
    Work::arxiv("2307.14984")
        .title("S3: Social-network Simulation System with Large Language Model-Empowered Agents")
        .year(2023)
        .source_version("arxiv-v1")
        .target(Target::claim(
            "collective-propagation-emergence",
            "Collective propagation of information, attitude and emotion emerges from local LLM-agent updates",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/gao2023/設計書.md")
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める (`mock://…` はオフラインの scripted クライアント，それ以外はホスト名で
/// Ollama / OpenAI を分ける)．推測しているのは分類だけで，値そのものは記録から採る．
///
/// `model_snapshot` に入るのは `llama3.1` のような動くエイリアスであることが多い．
/// socsim-llm はスナップショット id を持たないので，持っていない値を作らずに
/// 名乗られた名前を書く．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // S³ のプロンプトはエージェントごとに組み立てられ，固定の system prompt を
        // 持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// S³ 本体
// ---------------------------------------------------------------------------

/// S³ 1 本ぶんの記録．
///
/// ラウンドごとの 6 指標 (`t` は時間軸なので値としては書かない) と，run 全体を
/// 1 つの値で表す `converged` / `final_round` / LLM 呼び出しの内訳を書く．
/// 実行時間は `status.json` の `duration_sec` が正本なので指標にはしない．
pub fn log_simulation(run: &mut Run, result: &SimulationResult) {
    for m in &result.metrics_history {
        log_step(run, m);
    }
    run.log_metrics(
        SCOPE,
        &[
            ("converged", if result.converged { 1.0 } else { 0.0 }),
            ("final_round", result.final_round as f64),
            ("llm_calls", result.metadata.total() as f64),
            ("llm_cache_hits", result.metadata.cache_hits() as f64),
            ("llm_cache_hit_rate", result.metadata.cache_hit_rate()),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// [`Metrics`] の 6 フィールドを 1 ラウンドぶんまとめて書く．
///
/// 感情は calm / moderate / intense の 3 本の指標にする．カテゴリそのものに番号を
/// 振るのではなく，各カテゴリの母集団割合という «ラウンドごとの数» を書いている
/// (3 本の和は 1)．
fn log_step(run: &mut Run, m: &Metrics) {
    run.log_metrics_at(
        m.t as u64,
        T_UNIT,
        SCOPE,
        &[
            ("attitude_positive_frac", m.attitude_positive_frac),
            ("emotion_calm", m.emotion_calm),
            ("emotion_moderate", m.emotion_moderate),
            ("emotion_intense", m.emotion_intense),
            ("behavior_adoption_rate", m.behavior_adoption_rate),
            ("info_cascade_size", m.info_cascade_size as f64),
        ],
    )
    .unwrap_or_else(|e| panic!("round {} の指標の記録に失敗: {e}", m.t));
}

// ---------------------------------------------------------------------------
// 古典ベースライン
// ---------------------------------------------------------------------------

/// ベースライン 1 本ぶんの記録．
///
/// `prefix` はモデルラベル (`reproduce` では `baseline_lt` のように接頭辞を付け，
/// `baseline` サブコマンド単独では接頭辞なし)．`reproduce` は 5 モデルを 1 つの run
/// に書くので，`(step, scope, name)` が衝突しないよう名前でモデルを分ける．
pub fn log_baseline(
    run: &mut Run,
    prefix: Option<&str>,
    history: &[BaselineMetrics],
    final_round: usize,
    converged: bool,
) {
    let name = |base: &str| match prefix {
        Some(p) => format!("{p}_{base}"),
        None => base.to_string(),
    };
    for m in history {
        run.log_metrics_at(
            m.t as u64,
            T_UNIT,
            SCOPE,
            &[
                (name("active_frac").as_str(), m.active_frac),
                (name("mean_opinion").as_str(), m.mean_opinion),
                (
                    name("cumulative_reached").as_str(),
                    m.cumulative_reached as f64,
                ),
            ],
        )
        .unwrap_or_else(|e| panic!("round {} のベースライン指標の記録に失敗: {e}", m.t));
    }
    run.log_metrics(
        SCOPE,
        &[
            (name("final_round").as_str(), final_round as f64),
            (
                name("converged").as_str(),
                if converged { 1.0 } else { 0.0 },
            ),
        ],
    )
    .expect("ベースラインの run スコープ指標の記録に失敗");
}

/// `reproduce` でベースラインに付ける接頭辞．
pub fn baseline_prefix(model: BaselineModel) -> String {
    format!("baseline_{}", model.label())
}

// ---------------------------------------------------------------------------
// reproduce の帯照合
// ---------------------------------------------------------------------------

/// 観測量そのものは run 全体を 1 つの値で表す数なので指標に書く．
pub fn log_observations(run: &mut Run, checks: &[Check]) {
    let values: Vec<(&str, f64)> = checks
        .iter()
        .map(|c| (c.indicator.as_str(), c.observed))
        .collect();
    run.log_metrics(SCOPE, &values)
        .expect("headline 観測量の記録に失敗");

    let passed = checks.iter().filter(|c| c.pass).count();
    run.log_metrics(
        SCOPE,
        &[
            ("checks_passed", passed as f64),
            ("checks_total", checks.len() as f64),
        ],
    )
    .expect("帯照合の集計の記録に失敗");
}

/// 帯照合の判定は数ではないので `events.jsonl` へ書く．
///
/// 比較の向き (`>=` / `<=`) と PASS / off はカテゴリであって指標ではない．照合先の
/// 帯は論文が報告した数値ではなく，この再現実装が置いた定性的なアンカー
/// ([`crate::reproduce`] の冒頭を参照) なので，出典を要求する `reference.csv` にも
/// 書かない — 論文の報告値と自前のアンカーが後から見分けられなくなる．
pub fn log_checks(run: &mut Run, checks: &[Check]) {
    for c in checks {
        run.log_event(CHECK_EVENT, c)
            .unwrap_or_else(|e| panic!("帯照合 {} の記録に失敗: {e}", c.indicator));
    }
}

/// 帯照合イベントの種別．コア語彙に無いので `x.<repo_id>.<name>` を使う．
const CHECK_EVENT: &str = "x.gao2023.check";
