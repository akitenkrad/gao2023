//! 一括再現 (`reproduce`) — S³ の headline 伝播ダイナミクスと古典ベースライン比較．
//!
//! 論文 (Gao et al. 2023) の中心的主張は，個々の LLM エージェントの局所更新から
//! **集団レベルの情報・態度・感情の伝播が創発する**ことである．ローカルモデル
//! (llama3.2) と論文の GPT 系は一致しないため，再現目標は **定性的** (伝播曲線の
//! 傾向・符号) とする (README「Two-layer determinism」参照)．本モジュールは:
//!
//! 1. S³ を実行し，集団指標時系列から **headline 観測量** を計算する:
//!    - `attitude_rise`        : positive 態度割合の t=0 → 終端の上昇幅．
//!    - `cascade_growth_ratio` : 情報カスケード規模の 終端/初期 比．
//!    - `behavior_adoption_final` : 終端の行動採用率．
//!    - `emotion_msed`         : 感情分布の終端と論文参照分布との平均二乗誤差
//!      (MSED 整合の代理)．
//!    - `attitude_corr`        : 態度時系列と単調増加参照の Pearson 相関 (Cor 整合)．
//! 2. 同一有向網・同一シードで **4 つの古典ベースライン** (LT/IC/Voter/DeGroot) を
//!    実行し，S³ と並べて伝播の到達割合を比較する (LLM 呼び出し 0 回)．
//! 3. 観測量を論文の定性帯と PASS / off で照合し，`reproduce_summary.json` に書く．
//!
//! 出力は runvault の run ディレクトリに落とす (`crate::record` を参照):
//! - `metrics.csv`   — S³ の round 別集団指標と，`baseline_<model>_*` の名前を持つ
//!   各ベースラインの round 別指標．headline 観測量は step を持たない run スコープ．
//! - `events.jsonl`  — 帯照合の判定 (`x.gao2023.check`)．
//! - `config.json`   — 実験条件 (visualize がネットワーク描画にも使う)．

use serde::Serialize;

use crate::baseline::{run_baseline_observed, BaselineModel, BaselineParams};
use crate::config::Config;
use crate::llm::S3Client;
use crate::metrics::Metrics;
use crate::simulation::{run_with_client_observed, SimulationResult};

// --------------------------------------------------------------------------- //
// 論文の定性帯 (qualitative anchors)
// --------------------------------------------------------------------------- //

/// positive 態度割合の上昇幅の下限 (伝播が起きれば正に増える)．
pub const ATTITUDE_RISE_MIN: f64 = 0.05;
/// カスケード規模 終端/初期 比の下限 (情報が種から広がる)．
pub const CASCADE_GROWTH_MIN: f64 = 1.5;
/// 終端の行動採用率の下限 (一定割合が投稿/転送に参加する)．
pub const BEHAVIOR_ADOPTION_MIN: f64 = 0.05;
/// 感情分布 MSED の上限 (参照分布との整合; 小さいほど良い)．
pub const EMOTION_MSED_MAX: f64 = 0.20;
/// 態度時系列の単調増加参照との相関の下限 (Cor 整合)．
pub const ATTITUDE_CORR_MIN: f64 = 0.30;

/// 論文 §3/Table の感情分布の参照 (calm/moderate/intense)．議論が活性化した話題で
/// 中〜強感情が支配的になる定性形状を代理する (絶対値ではなく形状の整合を測る)．
pub const PAPER_EMOTION_REF: [f64; 3] = [0.20, 0.45, 0.35];

// --------------------------------------------------------------------------- //
// observed 観測量
// --------------------------------------------------------------------------- //

/// 1 指標の観測-参照-PASS 三つ組．
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    /// 指標名．
    pub indicator: String,
    /// 観測値．
    pub observed: f64,
    /// 論文側の参照値 (帯の境界)．
    pub paper: f64,
    /// 比較の向き (">=" なら observed>=paper で PASS; "<=" なら observed<=paper)．
    pub direction: String,
    /// PASS したか．
    pub pass: bool,
}

impl Check {
    fn ge(indicator: &str, observed: f64, paper: f64) -> Self {
        Check {
            indicator: indicator.to_string(),
            observed,
            paper,
            direction: ">=".to_string(),
            pass: observed >= paper,
        }
    }
    fn le(indicator: &str, observed: f64, paper: f64) -> Self {
        Check {
            indicator: indicator.to_string(),
            observed,
            paper,
            direction: "<=".to_string(),
            pass: observed <= paper,
        }
    }
}

/// 1 ベースラインの最終到達割合 (S³ との比較用)．
#[derive(Debug, Clone, Serialize)]
pub struct BaselineComparison {
    /// モデルラベル (lt/ic/voter/degroot)．
    pub model: String,
    /// 最終 active/positive 割合．
    pub final_active_frac: f64,
    /// 最終平均意見 (DeGroot は連続，他は active 割合と一致)．
    pub final_mean_opinion: f64,
    /// 累積到達ノード数 (最終)．
    pub final_reached: usize,
    /// 収束 round．
    pub final_round: usize,
}

/// `reproduce_summary.json` の本体．
#[derive(Debug, Clone, Serialize)]
pub struct ReproduceSummary {
    /// LLM モデル名 (mock 時は mock-*)．
    pub llm_model: String,
    /// mock 実行か (オフライン)．
    pub mock: bool,
    /// S³ の network/population/rounds など主要設定の要約文字列．
    pub setup: String,
    /// observed-vs-paper の各チェック．
    pub checks: Vec<Check>,
    /// 全チェック PASS か．
    pub all_pass: bool,
    /// PASS 数 / 総数．
    pub passed: usize,
    /// 総チェック数．
    pub total: usize,
    /// S³ の最終 positive 態度割合 (比較表のアンカー)．
    pub s3_final_active_frac: f64,
    /// S³ の最終カスケード割合 (到達/人口)．
    pub s3_final_reached_frac: f64,
    /// 4 古典ベースラインとの比較．
    pub baselines: Vec<BaselineComparison>,
    /// LLM cache-hit 率 (live 時に意味を持つ)．
    pub cache_hit_rate: f64,
}

// --------------------------------------------------------------------------- //
// 観測量の計算
// --------------------------------------------------------------------------- //

/// Pearson 相関 (定数列のときは 0)．
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n as f64;
    let my = ys.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= 0.0 || vy <= 0.0 {
        0.0
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

/// 感情分布の平均二乗誤差 (MSED 代理): 終端分布と参照分布の要素差の二乗平均．
fn emotion_msed(last: &Metrics) -> f64 {
    let obs = [
        last.emotion_calm,
        last.emotion_moderate,
        last.emotion_intense,
    ];
    obs.iter()
        .zip(PAPER_EMOTION_REF.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum::<f64>()
        / 3.0
}

/// S³ の集団指標履歴から headline 観測量を計算し，論文帯と照合する．
pub fn build_checks(history: &[Metrics]) -> Vec<Check> {
    let first = &history[0];
    let last = history.last().expect("history must be non-empty");

    let attitude_rise = last.attitude_positive_frac - first.attitude_positive_frac;
    let cascade_growth = if first.info_cascade_size == 0 {
        last.info_cascade_size as f64
    } else {
        last.info_cascade_size as f64 / first.info_cascade_size as f64
    };
    let behavior_adoption = last.behavior_adoption_rate;
    let msed = emotion_msed(last);

    // 態度時系列 vs 単調増加参照 (0..1) の相関 (Cor 整合の代理)．
    let att: Vec<f64> = history.iter().map(|m| m.attitude_positive_frac).collect();
    let ramp: Vec<f64> = (0..att.len())
        .map(|i| i as f64 / (att.len().max(2) - 1) as f64)
        .collect();
    let corr = pearson(&att, &ramp);

    vec![
        Check::ge("attitude_rise", attitude_rise, ATTITUDE_RISE_MIN),
        Check::ge("cascade_growth_ratio", cascade_growth, CASCADE_GROWTH_MIN),
        Check::ge(
            "behavior_adoption_final",
            behavior_adoption,
            BEHAVIOR_ADOPTION_MIN,
        ),
        Check::le("emotion_msed", msed, EMOTION_MSED_MAX),
        Check::ge("attitude_corr", corr, ATTITUDE_CORR_MIN),
    ]
}

// --------------------------------------------------------------------------- //
// オーケストレーション
// --------------------------------------------------------------------------- //

/// reproduce 一括実行の結果 (S³ 結果 + ベースライン比較 + summary)．
pub struct ReproduceOutput {
    /// S³ 実行結果 (記録用)．
    pub s3: SimulationResult,
    /// 各ベースラインの実行結果 (記録用; round 別指標・収束・最終 round を含む)．
    pub baseline_results: Vec<crate::baseline::BaselineResult>,
    /// summary．
    pub summary: ReproduceSummary,
}

/// reproduce の中核: 与えられた S³ クライアントで実行し，4 ベースラインと比較する．
///
/// `mock` は live LLM 不在 (オフライン) を示し summary に記録するだけで，挙動は
/// クライアントが mock か live かで決まる (本関数はクライアント非依存)．
pub fn run_reproduce(
    cfg: &Config,
    client: S3Client,
    mock: bool,
    params: &BaselineParams,
) -> Result<ReproduceOutput, String> {
    run_reproduce_observed(cfg, client, mock, params, |_| {}, |_| {})
}

/// 同一網・同一シードで比較する古典ベースライン．
///
/// 進捗の分母はここから採る — 「4」と書けば，モデルを 1 つ足したときに黙って
/// ずれる分母になる．
pub const BASELINE_MODELS: [BaselineModel; 4] = [
    BaselineModel::LinearThreshold,
    BaselineModel::IndependentCascade,
    BaselineModel::Voter,
    BaselineModel::DeGroot,
];

/// The same, reporting the S³ rounds and the baselines separately.
///
/// Two callbacks and not one, because the two phases do not cost the same: an
/// S³ round is a model call per reached agent, a baseline round is arithmetic
/// over the same network. Counting them together would extrapolate the price of
/// the first onto the second and produce an estimate that is confidently wrong.
/// Both are counted in rounds, which is what each phase's own cost is in.
pub fn run_reproduce_observed(
    cfg: &Config,
    client: S3Client,
    mock: bool,
    params: &BaselineParams,
    on_s3_round: impl FnMut(usize),
    mut on_baseline_round: impl FnMut(usize),
) -> Result<ReproduceOutput, String> {
    let s3 = run_with_client_observed(cfg, client, on_s3_round)?;
    let checks = build_checks(&s3.metrics_history);
    let passed = checks.iter().filter(|c| c.pass).count();
    let total = checks.len();
    let last = s3.metrics_history.last().unwrap();

    // 4 古典ベースラインを同一網・同一シードで実行 (LLM 呼び出し 0 回)．
    let mut baselines = Vec::new();
    let mut baseline_results = Vec::new();
    for model in BASELINE_MODELS {
        let r = run_baseline_observed(cfg, model, params, &mut on_baseline_round);
        let bl = r.last();
        baselines.push(BaselineComparison {
            model: model.label().to_string(),
            final_active_frac: bl.active_frac,
            final_mean_opinion: bl.mean_opinion,
            final_reached: bl.cumulative_reached,
            final_round: r.final_round,
        });
        baseline_results.push(r);
    }

    let n = cfg.population.max(1) as f64;
    let summary = ReproduceSummary {
        llm_model: s3.llm_model.clone(),
        mock,
        setup: format!(
            "network={} population={} rounds={} top_k={} seed_posters={} seed={:?} llm_perception={}",
            cfg.network.label(),
            cfg.population,
            cfg.rounds,
            cfg.top_k,
            cfg.seed_posters,
            cfg.seed,
            cfg.llm_perception,
        ),
        all_pass: passed == total,
        passed,
        total,
        s3_final_active_frac: last.attitude_positive_frac,
        s3_final_reached_frac: last.info_cascade_size as f64 / n,
        baselines,
        cache_hit_rate: s3.metadata.cache_hit_rate(),
        checks,
    };

    Ok(ReproduceOutput {
        s3,
        baseline_results,
        summary,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NetworkKind;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn mock_client() -> S3Client {
        // 受信があれば positive/post に寄せる擬似挙動 (mock_smoke 相当)．
        let backend = ScriptedClient::new("mock-model", |prompt: &str| {
            let received = prompt.matches("user ").count();
            let (emo, att, beh, content) = if received >= 1 {
                ("moderate", "positive", "post", "I agree, worth sharing.")
            } else {
                ("calm", "negative", "inactive", "")
            };
            format!("EMOTION: {emo}\nATTITUDE: {att}\nBEHAVIOR: {beh}\nCONTENT: {content}")
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn cfg() -> Config {
        Config {
            network: NetworkKind::BarabasiAlbert,
            population: 40,
            rounds: 12,
            seed_posters: 3,
            tol: 1e-12,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn reproduce_runs_and_compares_four_baselines() {
        let out = run_reproduce(&cfg(), mock_client(), true, &BaselineParams::default()).unwrap();
        // S³ + 4 ベースライン．
        assert_eq!(out.summary.baselines.len(), 4);
        assert_eq!(out.baseline_results.len(), 4);
        let labels: Vec<&str> = out
            .summary
            .baselines
            .iter()
            .map(|b| b.model.as_str())
            .collect();
        assert_eq!(labels, ["lt", "ic", "voter", "degroot"]);
        // checks の総数・PASS 数は整合．
        assert_eq!(out.summary.total, out.summary.checks.len());
        assert_eq!(
            out.summary.passed,
            out.summary.checks.iter().filter(|c| c.pass).count()
        );
    }

    #[test]
    fn reproduce_check_ordering_is_stable() {
        let out = run_reproduce(&cfg(), mock_client(), true, &BaselineParams::default()).unwrap();
        let names: Vec<&str> = out
            .summary
            .checks
            .iter()
            .map(|c| c.indicator.as_str())
            .collect();
        assert_eq!(
            names,
            [
                "attitude_rise",
                "cascade_growth_ratio",
                "behavior_adoption_final",
                "emotion_msed",
                "attitude_corr",
            ]
        );
    }

    #[test]
    fn pearson_on_monotone_is_one() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 2.0, 4.0, 6.0];
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-9);
        // 定数列は 0．
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[0.0, 1.0, 2.0]), 0.0);
    }
}
