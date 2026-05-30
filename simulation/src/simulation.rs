//! 初期化と実行ドライバ (SimulationBuilder 配線 + 二層 LLM レイヤ)．
//!
//! 二層決定論を配線する:
//! - **下層 (決定論的 socsim コア)**: `derive_seed(root, &[0])` で網生成・属性/初期
//!   感情/初期態度割当の init RNG を，`derive_seed(root, &[1])` で engine RNG
//!   (= RandomActivationScheduler) を派生する．bit 単位で再現する．
//! - **上層 (非決定的 LLM レイヤ)**: [`crate::llm`] のキャッシュ付き Ollama→OpenAI
//!   フォールバッククライアントに閉じ込め，`temperature=0`/`seed` 固定 + プロンプト
//!   →応答キャッシュで擬似決定論化する．モデル・endpoint・温度・seed・cache-hit を
//!   `run_metadata.json` に記録する．
//!
//! # 有向フォローグラフの構築
//!
//! `socsim-net` は issue #28 で **有向生成器** を提供するようになったため，ネット
//! ワーク種別ごとに最も忠実な構築経路を用いる:
//!   - **BA** → [`DiSocialNetwork::barabasi_albert_directed`]．各新規ノードが `m`
//!     本の出弧 `new → target` を張り，target は in-degree 優先で選ばれる．辺
//!     `A → B` = 「A が B をフォロー」となり，in-degree(B) = B のフォロワ数となる．
//!   - **ER** → [`DiSocialNetwork::erdos_renyi_directed`]．順序対ごとに独立に弧を
//!     張る (非対称になりうる)．
//!   - **WS** → 有向生成器が無いので，無向 `watts_strogatz` を生成してから
//!     [`SocialNetwork::to_directed`]`(p_mutual, rng)` で方向を付与する．確率
//!     `p_mutual` で双方向 (相互フォロー)，残りは RNG で片方向に倒す．
//!
//! いずれの経路でも辺 `A → B` = 「A が B をフォロー」の規約は同一で，B の投稿は B の
//! フォロワ (= `in_neighbors(B)`) に届く ([`crate::mechanisms::NetworkInitMechanism`])．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use rand::Rng;
use serde::Serialize;

use socsim_core::{derive_seed, AgentId, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_net::{DiSocialNetwork, SocialNetwork};

use crate::config::{Config, NetworkKind};
use crate::llm::{build_live_client, S3Client};
use crate::mechanisms::{
    LLMPerceptionMechanism, NetworkInitMechanism, PopulationMetricsMechanism, SharedClient,
    SharedMetadata, SocialContagionMechanism,
};
use crate::metrics::Metrics;
use crate::world::{AgentState, Attitude, Behavior, Emotion, Profile, S3World};

/// 網生成・属性/初期感情/初期態度割当用 RNG ラベル．
const RNG_WORLD_INIT: u64 = 0;
/// socsim エンジン (= RandomActivationScheduler) 用 RNG ラベル．
const RNG_ENGINE: u64 = 1;

/// 既定トピック (合成実行の議論対象)．
pub const DEFAULT_TOPIC: &str = "nuclear_wastewater_release";

/// 人口統計属性のテンプレート (ラウンドロビン + RNG で割当; 決定論的)．
const GENDERS: [&str; 2] = ["man", "woman"];
const OCCUPATIONS: [&str; 6] = [
    "teacher",
    "engineer",
    "student",
    "nurse",
    "shop owner",
    "office worker",
];

/// シミュレーション全体の実行結果．
pub struct SimulationResult {
    /// 各 round (t=0 を含む) の集団メトリクス履歴．
    pub metrics_history: Vec<Metrics>,
    /// 収束したか (positive 態度割合が定常)．
    pub converged: bool,
    /// 収束 (または最終) round 番号．
    pub final_round: usize,
    /// LLM 呼び出しメタデータの集計．
    pub metadata: MetadataCollector,
    /// LLM モデル名 (run_metadata 用)．
    pub llm_model: String,
    /// LLM endpoint (run_metadata 用; primary)．
    pub llm_endpoint: String,
}

/// 指定モデルで **有向** フォローグラフ [`DiSocialNetwork`] を構築する．
///
/// 規約は全種別で同一: 辺 `A → B` = 「A が B をフォロー」．したがって B の投稿は
/// `in_neighbors(B)` (= B のフォロワ) に届く．
///   - **BA / ER** は `socsim-net` の有向生成器 (issue #28) を直接呼ぶ．BA は
///     in-degree 優先選択なので in-degree がフォロワ数の重い裾を持つ．
///   - **WS** は有向生成器が無いため，無向 `watts_strogatz` を生成してから
///     [`SocialNetwork::to_directed`]`(cfg.ws_p_mutual, rng)` で方向を付与する．
fn build_network(cfg: &Config, ids: &[AgentId], rng: &mut SimRng) -> DiSocialNetwork {
    match cfg.network {
        NetworkKind::ErdosRenyi => DiSocialNetwork::erdos_renyi_directed(ids, cfg.er_p, rng),
        NetworkKind::BarabasiAlbert => {
            DiSocialNetwork::barabasi_albert_directed(ids, cfg.ba_m, rng)
        }
        NetworkKind::WattsStrogatz => {
            let undirected = SocialNetwork::watts_strogatz(ids, cfg.ws_k, cfg.ws_beta, rng);
            undirected.to_directed(cfg.ws_p_mutual, rng)
        }
    }
}

/// [`build_network`] の公開ラッパ．古典ベースライン ([`crate::baseline`]) が S³ と
/// **同一の有向網** を同一シードストリームから再構築するために用いる (構築規約は
/// 上記 [`build_network`] と同一)．
pub fn build_network_pub(cfg: &Config, ids: &[AgentId], rng: &mut SimRng) -> DiSocialNetwork {
    build_network(cfg, ids, rng)
}

/// 世界状態を初期化する (有向網構築 + 属性/初期感情/初期態度割当)．
///
/// 属性・初期感情・初期態度は init RNG から決定論的に割り当てる (socsim コア層)．
/// 初期態度の LLM 決定 (論文) は拡張点であり，既定は規則ベース (RNG) を用いる．
/// `seed_posters` 体の発信源は round 0 で必ず投稿する (種まき; outbox へ直接積む)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> S3World {
    let ids: Vec<AgentId> = (0..cfg.population as u64).map(AgentId).collect();

    let net = build_network(cfg, &ids, rng);

    let mut agents: BTreeMap<AgentId, AgentState> = BTreeMap::new();
    for (idx, &id) in ids.iter().enumerate() {
        let gender = GENDERS[rng.gen_range(0..GENDERS.len())].to_string();
        let age = 18 + rng.gen_range(0u8..50);
        let occupation = OCCUPATIONS[idx % OCCUPATIONS.len()].to_string();
        let profile = Profile {
            gender,
            age,
            occupation,
        };
        // 初期感情・初期態度を一様抽選 (決定論的; init RNG ストリーム)．
        let emotion = match rng.gen_range(0u8..3) {
            0 => Emotion::Calm,
            1 => Emotion::Moderate,
            _ => Emotion::Intense,
        };
        let attitude = if rng.gen_bool(0.5) {
            Attitude::Positive
        } else {
            Attitude::Negative
        };
        agents.insert(id, AgentState::new(profile, emotion, attitude));
    }

    let mut world = S3World::new(net, agents, cfg.rounds as u64);

    // 発信源の種まき: 先頭 seed_posters 体を round 0 で必ず投稿させる (outbox へ積む)．
    // PreStep (round 1 相当の最初の tick) でフォロワに配送される．
    let topic_h = DEFAULT_TOPIC.replace('_', " ");
    for &id in ids.iter().take(cfg.seed_posters.min(cfg.population)) {
        if let Some(state) = world.agents.get_mut(&id) {
            state.behavior = Behavior::Post;
        }
        world.outbox.push(crate::world::Message {
            author: id,
            origin: id,
            content: format!("Sharing my view on {topic_h}."),
            round: 0,
            is_repost: false,
        });
        world.reached.insert(id);
    }

    world
}

/// シミュレーションを実行する (本番 LLM クライアントを構築して駆動)．
pub fn run(cfg: &Config) -> Result<SimulationResult, String> {
    let client =
        build_live_client(&cfg.llm).map_err(|e| format!("LLM クライアント構築に失敗: {e}"))?;
    run_with_client(cfg, client)
}

/// 与えられた [`S3Client`] でシミュレーションを実行する．
///
/// 本番は [`build_live_client`] の結果を，テストは [`crate::llm::wrap_client`] で
/// ラップした `mock::ScriptedClient` を渡す．
pub fn run_with_client(cfg: &Config, client: S3Client) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);

    // 初期世界 (root から派生した init RNG; 決定論的 socsim コア層)．
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    // LLM モデル/endpoint をメタデータ用に控える．
    let llm_model = client.inner().model().to_string();
    let llm_endpoint = client.inner().endpoint().to_string();

    // クライアント・メタデータを共有 (メカニズムと run ドライバで共用)．
    let shared_client: SharedClient = Rc::new(RefCell::new(client));
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));

    // Perception メカニズム: `--llm-perception` 有効時は SocialContagion と同一の
    // 共有クライアント・キャッシュで LLM 駆動選択を行う (既定は規則ベースで bit 決定論的)．
    let perception: LLMPerceptionMechanism = if cfg.llm_perception {
        LLMPerceptionMechanism::with_llm(
            cfg.top_k,
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            DEFAULT_TOPIC.to_string(),
        )
    } else {
        LLMPerceptionMechanism::new(cfg.top_k, false)
    };

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(NetworkInitMechanism))
        .add_mechanism(Box::new(perception))
        .add_mechanism(Box::new(SocialContagionMechanism::new(
            Rc::clone(&shared_client),
            Rc::clone(&shared_meta),
            cfg.llm.clone(),
            DEFAULT_TOPIC.to_string(),
        )))
        .add_mechanism(Box::new(PopulationMetricsMechanism::new(cfg.tol)))
        .build();

    let mut metrics_history: Vec<Metrics> = Vec::new();

    // 初期状態 (t=0) を記録．カスケード規模は種まきの reached 集合サイズ．
    {
        let w = sim.world();
        metrics_history.push(Metrics::compute(&w.agents, w.reached.len(), 0));
    }

    let mut converged = false;
    let mut final_round = 0usize;

    sim.run_observed(|report| {
        let t = report.t as usize;
        let w = report.world;
        metrics_history.push(Metrics::compute(&w.agents, w.reached.len(), t));
        converged = *report.scratch.get::<bool>("converged").unwrap_or(&false);
        final_round = t;
    })
    .map_err(|e| format!("シミュレーションの実行に失敗: {e}"))?;

    // キャッシュを保存 (cache_path 指定時; in-memory はスキップ)．
    if cfg.llm.cache_path.is_some() {
        let client = shared_client.borrow();
        client
            .cache()
            .save()
            .map_err(|e| format!("キャッシュ保存に失敗: {e}"))?;
    }

    let metadata = shared_meta.borrow().clone();

    Ok(SimulationResult {
        metrics_history,
        converged,
        final_round,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

/// メトリクス履歴を CSV に保存する．
///
/// 書き出し機構は `socsim_results::write_csv` に委譲する (各行を `serialize` し
/// 先頭行にヘッダを書く csv クレットの標準挙動; 従来の手書き writer とバイト等価)．
/// 行構造体 [`Metrics`] は repo 固有のままで，writer だけを共有化する．
pub fn save_metrics(metrics: &[Metrics], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    socsim_results::write_csv(metrics, &path).expect("metrics.csv の書き込みに失敗");
}

/// `run_metadata.json` の構造体 (LLM モデル・endpoint・温度・seed・cache 統計)．
#[derive(Serialize)]
pub struct RunMetadataJson {
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub determinism_note: &'static str,
}

/// `run_metadata.json` を保存する．
pub fn save_run_metadata(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = RunMetadataJson {
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (directed network, message delivery, \
                           activation order, metrics) is deterministic given the seed.",
    };
    // pretty-print JSON の書き出しは socsim_results::write_json に委譲する
    // (内部は serde_json::to_writer_pretty + flush; 従来の writer とバイト等価)．
    // model/endpoint/temperature/seed の値は従来どおり result / cfg から採り，
    // RunMetadataJson の構造 (フィールド名・順序・determinism_note) を保持する
    // (`MetadataCollector::summary()` は cache-hit 100% 再実行や呼び出し 0 件で
    // endpoint/model が変わりうるため，バイト等価のためここでは使わない)．
    let path = format!("{}/run_metadata.json", output_dir);
    socsim_results::write_json(&meta, &path).expect("run_metadata.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("出力ディレクトリの作成に失敗");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::wrap_client;
    use socsim_llm::mock::ScriptedClient;
    use socsim_llm::PromptCache;

    fn scripted_client() -> S3Client {
        let backend = ScriptedClient::new("mock-model", |_prompt: &str| {
            "EMOTION: moderate\nATTITUDE: positive\nBEHAVIOR: post\nCONTENT: I think this matters."
                .to_string()
        });
        wrap_client(backend, PromptCache::in_memory())
    }

    fn test_config() -> Config {
        Config {
            population: 12,
            rounds: 5,
            tol: 1e-12, // 収束で早期停止しないよう厳しめ
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn scripted_run_drives_attitudes_positive() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        let last = result.metrics_history.last().unwrap();
        // 全員 positive を採用するので positive 割合は 1 に向かう．
        assert!(last.attitude_positive_frac > 0.9);
        assert_eq!(result.metrics_history[0].t, 0);
    }

    #[test]
    fn core_is_deterministic_given_mock() {
        let cfg = test_config();
        let a = run_with_client(&cfg, scripted_client()).unwrap();
        let b = run_with_client(&cfg, scripted_client()).unwrap();
        let af: Vec<f64> = a
            .metrics_history
            .iter()
            .map(|m| m.attitude_positive_frac)
            .collect();
        let bf: Vec<f64> = b
            .metrics_history
            .iter()
            .map(|m| m.attitude_positive_frac)
            .collect();
        assert_eq!(af, bf);
        assert_eq!(a.final_round, b.final_round);
    }

    #[test]
    fn cascade_is_monotone_non_decreasing() {
        let cfg = test_config();
        let result = run_with_client(&cfg, scripted_client()).unwrap();
        let sizes: Vec<usize> = result
            .metrics_history
            .iter()
            .map(|m| m.info_cascade_size)
            .collect();
        for w in sizes.windows(2) {
            assert!(
                w[1] >= w[0],
                "info_cascade_size must be monotone: {sizes:?}"
            );
        }
    }
}
