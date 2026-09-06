//! 古典的拡散・意見ダイナミクスのベースライン (LLM 非依存・bit 決定論的)．
//!
//! S³ の LLM 駆動伝播との **対照群** として，同一の有向フォローグラフ上で動く
//! 4 つの古典モデルを実装する:
//!
//! - **LT** (Linear Threshold; Granovetter 1978 / Kempe-Kleinberg-Tardos 2003):
//!   各ノードはしきい値 θ を持ち，アクティブな in-neighbour (= フォロー先) の重み
//!   和が θ を超えると一度だけアクティブ化する (単調・進行性)．
//! - **IC** (Independent Cascade; Goldenberg 2001 / KKT 2003): 新規アクティブ化
//!   したノードは次 round に各 out-neighbour (= フォロワ) を確率 p で 1 回だけ
//!   感染させようと試みる (単調・進行性)．
//! - **Voter** (Clifford-Sudbury 1973 / Holley-Liggett 1975): 各ノードは毎 round
//!   ランダムな in-neighbour 1 体の意見をコピーする (非進行性・確率的合意形成)．
//! - **DeGroot** (DeGroot 1974): 連続意見をフォロー先の意見の平均へ更新する線形
//!   合意モデル (同期更新・収束)．
//!
//! いずれも S³ と同じ「辺 `A → B` = 「A が B をフォロー」」規約に従う:
//! - 「情報がフォロワへ流れる」モデル (IC) は **out-neighbour** (フォロワ) へ拡散，
//! - 「フォロー先から影響を受ける」モデル (LT/Voter/DeGroot) は **in-neighbour**
//!   (フォロー先) を参照する．
//!
//! socsim コア層の二層決定論を踏襲し，`derive_seed(root, &[…])` で初期化・ダイナ
//! ミクスの RNG ストリームを分離する．LLM 呼び出しは **0 回**．

use std::collections::{BTreeMap, BTreeSet};

use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_net::DiSocialNetwork;

use crate::config::Config;
use crate::simulation::build_network_pub;

/// RNG ラベル: 網生成 (S³ の `init_world` と同一ストリーム; 同じ網を得る)．
const RNG_WORLD_INIT: u64 = 0;
/// RNG ラベル: ベースライン固有のダイナミクス (IC のコイン投げ・Voter の選択)．
const RNG_BASELINE_DYN: u64 = 8;

// --------------------------------------------------------------------------- //
// モデル種別
// --------------------------------------------------------------------------- //

/// 古典的拡散・意見ダイナミクスのベースラインモデル種別．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineModel {
    /// Linear Threshold (進行性・しきい値駆動)．
    LinearThreshold,
    /// Independent Cascade (進行性・確率感染)．
    IndependentCascade,
    /// Voter model (非進行性・確率的合意)．
    Voter,
    /// DeGroot (連続意見の線形合意)．
    DeGroot,
}

impl BaselineModel {
    /// 短い識別ラベル (CLI / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            BaselineModel::LinearThreshold => "lt",
            BaselineModel::IndependentCascade => "ic",
            BaselineModel::Voter => "voter",
            BaselineModel::DeGroot => "degroot",
        }
    }

    /// 進行性 (一度アクティブ化すると非アクティブへ戻らない) モデルか．
    pub fn is_progressive(&self) -> bool {
        matches!(
            self,
            BaselineModel::LinearThreshold | BaselineModel::IndependentCascade
        )
    }
}

/// 文字列から [`BaselineModel`] をパースする．
pub fn parse_baseline(s: &str) -> Result<BaselineModel, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "lt" | "linear_threshold" | "linear-threshold" => Ok(BaselineModel::LinearThreshold),
        "ic" | "independent_cascade" | "independent-cascade" => {
            Ok(BaselineModel::IndependentCascade)
        }
        "voter" => Ok(BaselineModel::Voter),
        "degroot" | "de_groot" | "de-groot" => Ok(BaselineModel::DeGroot),
        _ => Err(format!(
            "不正なベースラインモデル: \"{}\" (lt / ic / voter / degroot)",
            s
        )),
    }
}

// --------------------------------------------------------------------------- //
// パラメータ・結果
// --------------------------------------------------------------------------- //

/// ベースライン実行のパラメータ．
#[derive(Debug, Clone)]
pub struct BaselineParams {
    /// LT: 各ノードの一様しきい値 θ (アクティブ in-neighbour 割合に対する) ∈ (0,1)．
    pub lt_theta: f64,
    /// IC: 1 辺あたりの感染確率 p ∈ [0,1]．
    pub ic_p: f64,
    /// DeGroot: 自己重み (残りをフォロー先へ均等配分) ∈ [0,1]．
    pub degroot_self_weight: f64,
    /// DeGroot/Voter の二値化しきい値 (意見 ≥ 0.5 を "active/positive" と数える)．
    pub binary_threshold: f64,
}

impl Default for BaselineParams {
    fn default() -> Self {
        BaselineParams {
            lt_theta: 0.3,
            ic_p: 0.15,
            degroot_self_weight: 0.5,
            binary_threshold: 0.5,
        }
    }
}

/// 1 round 分のベースライン指標 (baseline_metrics.csv の 1 行)．
#[derive(Debug, Clone, Serialize)]
pub struct BaselineMetrics {
    /// round 番号 t．
    pub t: usize,
    /// アクティブ/positive ノード割合 ∈ [0,1] (= S³ の attitude_positive_frac 相当)．
    pub active_frac: f64,
    /// 連続意見の平均 (LT/IC/Voter は active 割合と一致; DeGroot は意見平均)．
    pub mean_opinion: f64,
    /// 累積到達 (= これまでに一度でも active になった) ノード数 (= info_cascade_size 相当)．
    pub cumulative_reached: usize,
}

/// ベースライン実行の結果．
#[derive(Debug, Clone)]
pub struct BaselineResult {
    /// モデル種別．
    pub model: BaselineModel,
    /// 各 round (t=0 を含む) の指標履歴．
    pub history: Vec<BaselineMetrics>,
    /// 収束 (または最終) round 番号．
    pub final_round: usize,
    /// 収束したか (active 割合が定常)．
    pub converged: bool,
}

impl BaselineResult {
    /// 最終 round の指標を返す．
    pub fn last(&self) -> &BaselineMetrics {
        self.history.last().expect("history must be non-empty")
    }
}

// --------------------------------------------------------------------------- //
// 実行ドライバ
// --------------------------------------------------------------------------- //

/// ベースラインを実行する (S³ と同一の有向網・同一シードで; LLM 呼び出し 0 回)．
///
/// `cfg.seed_posters` 体の先頭ノードを round 0 の **種** (active / 意見 1.0) とし，
/// 同期 round で `cfg.rounds` まで進める (進行性モデルは飽和で早期停止)．
pub fn run_baseline(cfg: &Config, model: BaselineModel, params: &BaselineParams) -> BaselineResult {
    run_baseline_observed(cfg, model, params, &mut |_| {})
}

/// The same, calling `on_round` once for every propagation round.
///
/// The callback is where a caller counts its progress. A round is the unit
/// because it is the unit the cost is in: one round walks every node's
/// neighbours. A whole baseline would be a single tick, and with `--tol 0` a
/// single baseline runs for as long as `--rounds` says — measured at 4m33s for
/// 50,000 rounds over 20,000 nodes.
///
/// `&mut dyn FnMut` rather than `impl FnMut`, so the one callback can be handed
/// to whichever of the four models is selected without monomorphising each.
pub fn run_baseline_observed(
    cfg: &Config,
    model: BaselineModel,
    params: &BaselineParams,
    on_round: &mut dyn FnMut(usize),
) -> BaselineResult {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 網は S³ の init_world と同一ストリーム (`&[0]`) で生成し，同じトポロジを得る．
    let ids: Vec<AgentId> = (0..cfg.population as u64).map(AgentId).collect();
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let net = build_network_pub(cfg, &ids, &mut init_rng);

    // 種ノード: 先頭 seed_posters 体 (S³ の発信源と同じ集合)．
    let seeds: BTreeSet<AgentId> = ids
        .iter()
        .take(cfg.seed_posters.min(cfg.population))
        .copied()
        .collect();

    match model {
        BaselineModel::LinearThreshold => {
            run_linear_threshold(cfg, &ids, &net, &seeds, params, on_round)
        }
        BaselineModel::IndependentCascade => {
            run_independent_cascade(cfg, root, &ids, &net, &seeds, params, on_round)
        }
        BaselineModel::Voter => run_voter(cfg, root, &ids, &net, &seeds, params, on_round),
        BaselineModel::DeGroot => run_degroot(cfg, &ids, &net, &seeds, params, on_round),
    }
}

/// active 集合から 1 round の指標を作る (進行性モデル用)．
fn active_metrics(
    t: usize,
    active: &BTreeSet<AgentId>,
    n: usize,
    reached: usize,
) -> BaselineMetrics {
    let frac = if n == 0 {
        0.0
    } else {
        active.len() as f64 / n as f64
    };
    BaselineMetrics {
        t,
        active_frac: frac,
        mean_opinion: frac,
        cumulative_reached: reached,
    }
}

/// 連続意見ベクトルから 1 round の指標を作る (DeGroot 用)．
fn opinion_metrics(
    t: usize,
    opinion: &BTreeMap<AgentId, f64>,
    threshold: f64,
    reached: usize,
) -> BaselineMetrics {
    let n = opinion.len();
    let (sum, pos) = opinion.values().fold((0.0_f64, 0usize), |(s, p), &o| {
        (s + o, p + usize::from(o >= threshold))
    });
    BaselineMetrics {
        t,
        active_frac: if n == 0 { 0.0 } else { pos as f64 / n as f64 },
        mean_opinion: if n == 0 { 0.0 } else { sum / n as f64 },
        cumulative_reached: reached,
    }
}

// --------------------------------------------------------------------------- //
// LT — Linear Threshold (進行性)
// --------------------------------------------------------------------------- //

/// Linear Threshold: アクティブな in-neighbour (= フォロー先) の割合が θ を超えた
/// ノードがアクティブ化する (一度きり; 進行性)．均一しきい値 θ = `lt_theta`．
fn run_linear_threshold(
    cfg: &Config,
    ids: &[AgentId],
    net: &DiSocialNetwork,
    seeds: &BTreeSet<AgentId>,
    params: &BaselineParams,
    on_round: &mut dyn FnMut(usize),
) -> BaselineResult {
    let n = ids.len();
    let mut active: BTreeSet<AgentId> = seeds.clone();
    let mut history = vec![active_metrics(0, &active, n, active.len())];
    let mut final_round = 0;
    let mut converged = false;

    for t in 1..=cfg.rounds {
        let mut newly = Vec::new();
        for &v in ids {
            if active.contains(&v) {
                continue;
            }
            // v が「フォローしている」相手 = out_neighbors(v) (辺 v→u = v が u をフォロー)．
            let followees = net.out_neighbors(v);
            if followees.is_empty() {
                continue;
            }
            let active_followees = followees.iter().filter(|u| active.contains(u)).count();
            let frac = active_followees as f64 / followees.len() as f64;
            if frac >= params.lt_theta {
                newly.push(v);
            }
        }
        for v in &newly {
            active.insert(*v);
        }
        final_round = t;
        on_round(t);
        history.push(active_metrics(t, &active, n, active.len()));
        if newly.is_empty() {
            converged = true;
            break;
        }
    }

    BaselineResult {
        model: BaselineModel::LinearThreshold,
        history,
        final_round,
        converged,
    }
}

// --------------------------------------------------------------------------- //
// IC — Independent Cascade (進行性)
// --------------------------------------------------------------------------- //

/// Independent Cascade: 新規アクティブ化したノードが次 round に各フォロワ
/// (= out 方向の情報流; in_neighbors すなわち follower) を確率 p で 1 回だけ感染
/// させようと試みる (進行性)．
fn run_independent_cascade(
    cfg: &Config,
    root: u64,
    ids: &[AgentId],
    net: &DiSocialNetwork,
    seeds: &BTreeSet<AgentId>,
    params: &BaselineParams,
    on_round: &mut dyn FnMut(usize),
) -> BaselineResult {
    let n = ids.len();
    let mut rng = SimRng::from_seed(derive_seed(root, &[RNG_BASELINE_DYN]));
    let mut active: BTreeSet<AgentId> = seeds.clone();
    let mut frontier: Vec<AgentId> = seeds.iter().copied().collect();
    let mut history = vec![active_metrics(0, &active, n, active.len())];
    let mut final_round = 0;
    let mut converged = false;

    for t in 1..=cfg.rounds {
        let mut newly: Vec<AgentId> = Vec::new();
        // 決定論のため frontier をソート済みで走査する．
        let mut frontier_sorted = frontier.clone();
        frontier_sorted.sort_by_key(|a| a.0);
        for &u in &frontier_sorted {
            // u の投稿が届く相手 = u のフォロワ = in_neighbors(u) (S³ の配送と同一)．
            let mut followers = net.in_neighbors(u);
            followers.sort_by_key(|a| a.0);
            for w in followers {
                if active.contains(&w) || newly.contains(&w) {
                    continue;
                }
                if rng.gen_bool(params.ic_p.clamp(0.0, 1.0)) {
                    newly.push(w);
                }
            }
        }
        for w in &newly {
            active.insert(*w);
        }
        frontier = newly.clone();
        final_round = t;
        on_round(t);
        history.push(active_metrics(t, &active, n, active.len()));
        if newly.is_empty() {
            converged = true;
            break;
        }
    }

    BaselineResult {
        model: BaselineModel::IndependentCascade,
        history,
        final_round,
        converged,
    }
}

// --------------------------------------------------------------------------- //
// Voter (非進行性)
// --------------------------------------------------------------------------- //

/// Voter model: 各 round で各ノードがフォロー先 (= out_neighbors) からランダムに
/// 1 体を選び，その二値意見をコピーする (同期更新)．種は意見 1，残りは 0 で開始．
fn run_voter(
    cfg: &Config,
    root: u64,
    ids: &[AgentId],
    net: &DiSocialNetwork,
    seeds: &BTreeSet<AgentId>,
    params: &BaselineParams,
    on_round: &mut dyn FnMut(usize),
) -> BaselineResult {
    let mut rng = SimRng::from_seed(derive_seed(root, &[RNG_BASELINE_DYN]));
    let mut opinion: BTreeMap<AgentId, f64> = ids
        .iter()
        .map(|&v| (v, if seeds.contains(&v) { 1.0 } else { 0.0 }))
        .collect();
    let mut reached: BTreeSet<AgentId> = seeds.clone();
    let mut history = vec![opinion_metrics(
        0,
        &opinion,
        params.binary_threshold,
        reached.len(),
    )];
    let mut final_round = 0;
    let mut converged = false;

    for t in 1..=cfg.rounds {
        let mut next = opinion.clone();
        for &v in ids {
            let followees = net.out_neighbors(v);
            if followees.is_empty() {
                continue;
            }
            let pick = followees[rng.gen_range(0..followees.len())];
            let val = opinion[&pick];
            next.insert(v, val);
            if val >= params.binary_threshold {
                reached.insert(v);
            }
        }
        let changed = next != opinion;
        opinion = next;
        final_round = t;
        on_round(t);
        history.push(opinion_metrics(
            t,
            &opinion,
            params.binary_threshold,
            reached.len(),
        ));
        if !changed {
            converged = true;
            break;
        }
    }

    BaselineResult {
        model: BaselineModel::Voter,
        history,
        final_round,
        converged,
    }
}

// --------------------------------------------------------------------------- //
// DeGroot (連続合意)
// --------------------------------------------------------------------------- //

/// DeGroot: 各ノードの意見を `self_weight·自意見 + (1-self_weight)·フォロー先平均`
/// へ同期更新する線形合意モデル．種は意見 1，残りは 0 で開始する．
fn run_degroot(
    cfg: &Config,
    ids: &[AgentId],
    net: &DiSocialNetwork,
    seeds: &BTreeSet<AgentId>,
    params: &BaselineParams,
    on_round: &mut dyn FnMut(usize),
) -> BaselineResult {
    let w = params.degroot_self_weight.clamp(0.0, 1.0);
    let mut opinion: BTreeMap<AgentId, f64> = ids
        .iter()
        .map(|&v| (v, if seeds.contains(&v) { 1.0 } else { 0.0 }))
        .collect();
    let mut reached: BTreeSet<AgentId> = seeds.clone();
    let mut history = vec![opinion_metrics(
        0,
        &opinion,
        params.binary_threshold,
        reached.len(),
    )];
    let mut final_round = 0;
    let mut converged = false;

    for t in 1..=cfg.rounds {
        let mut next = opinion.clone();
        let mut max_delta = 0.0_f64;
        for &v in ids {
            let followees = net.out_neighbors(v);
            let new_val = if followees.is_empty() {
                opinion[&v]
            } else {
                let nbr_mean =
                    followees.iter().map(|u| opinion[u]).sum::<f64>() / followees.len() as f64;
                w * opinion[&v] + (1.0 - w) * nbr_mean
            };
            max_delta = max_delta.max((new_val - opinion[&v]).abs());
            if new_val >= params.binary_threshold {
                reached.insert(v);
            }
            next.insert(v, new_val);
        }
        opinion = next;
        final_round = t;
        on_round(t);
        history.push(opinion_metrics(
            t,
            &opinion,
            params.binary_threshold,
            reached.len(),
        ));
        if max_delta < cfg.tol {
            converged = true;
            break;
        }
    }

    BaselineResult {
        model: BaselineModel::DeGroot,
        history,
        final_round,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NetworkKind;

    fn cfg(network: NetworkKind, population: usize) -> Config {
        Config {
            network,
            population,
            rounds: 30,
            seed_posters: 3,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn parse_roundtrip() {
        for (s, m) in [
            ("lt", BaselineModel::LinearThreshold),
            ("ic", BaselineModel::IndependentCascade),
            ("voter", BaselineModel::Voter),
            ("degroot", BaselineModel::DeGroot),
        ] {
            assert_eq!(parse_baseline(s).unwrap(), m);
            assert_eq!(parse_baseline(m.label()).unwrap(), m);
        }
        assert!(parse_baseline("nope").is_err());
    }

    #[test]
    fn all_baselines_run_on_all_networks() {
        let params = BaselineParams::default();
        for net in [
            NetworkKind::ErdosRenyi,
            NetworkKind::WattsStrogatz,
            NetworkKind::BarabasiAlbert,
        ] {
            let mut c = cfg(net, 40);
            c.er_p = 0.1;
            for model in [
                BaselineModel::LinearThreshold,
                BaselineModel::IndependentCascade,
                BaselineModel::Voter,
                BaselineModel::DeGroot,
            ] {
                let r = run_baseline(&c, model, &params);
                assert_eq!(r.history[0].t, 0);
                assert!(!r.history.is_empty());
                for m in &r.history {
                    assert!((0.0..=1.0).contains(&m.active_frac));
                    assert!((0.0..=1.0).contains(&m.mean_opinion));
                }
            }
        }
    }

    #[test]
    fn progressive_models_have_monotone_reach() {
        let params = BaselineParams::default();
        let c = cfg(NetworkKind::BarabasiAlbert, 60);
        for model in [
            BaselineModel::LinearThreshold,
            BaselineModel::IndependentCascade,
        ] {
            let r = run_baseline(&c, model, &params);
            let reach: Vec<usize> = r.history.iter().map(|m| m.cumulative_reached).collect();
            for w in reach.windows(2) {
                assert!(w[1] >= w[0], "{model:?} reach must be monotone: {reach:?}");
            }
            // 種は round 0 に含まれる．
            assert_eq!(reach[0], c.seed_posters);
        }
    }

    #[test]
    fn baselines_are_bit_deterministic() {
        let params = BaselineParams::default();
        let c = cfg(NetworkKind::BarabasiAlbert, 50);
        for model in [
            BaselineModel::LinearThreshold,
            BaselineModel::IndependentCascade,
            BaselineModel::Voter,
            BaselineModel::DeGroot,
        ] {
            let a = run_baseline(&c, model, &params);
            let b = run_baseline(&c, model, &params);
            let af: Vec<f64> = a.history.iter().map(|m| m.active_frac).collect();
            let bf: Vec<f64> = b.history.iter().map(|m| m.active_frac).collect();
            assert_eq!(af, bf, "{model:?} は同一シードで bit 決定論的であるべき");
            assert_eq!(a.final_round, b.final_round);
        }
    }

    #[test]
    fn degroot_drives_mean_opinion_up_from_seeds() {
        let params = BaselineParams::default();
        let c = cfg(NetworkKind::BarabasiAlbert, 60);
        let r = run_baseline(&c, BaselineModel::DeGroot, &params);
        // 種が意見 1，残り 0 なので平均意見は 0 < m < 1 に収束する．
        let last = r.last();
        assert!(last.mean_opinion >= 0.0 && last.mean_opinion <= 1.0);
    }
}
