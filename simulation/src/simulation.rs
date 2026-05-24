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
//! `socsim-net` の生成器 (`erdos_renyi` / `watts_strogatz` / `barabasi_albert`) は
//! **無向** `SocialNetwork` しか生成しない (issue #18 の有向 API は生成器を持た
//! ない)．そこで:
//!   1. 指定モデルで無向トポロジを生成する．
//!   2. 各無向辺 `{a, b}` に対し，決定論的に方向を付与して [`DiSocialNetwork`] の
//!      有向辺を張る．規約は «小さい id → 大きい id を基本に，init RNG のコイン投げ
//!      で約半数を逆向きに» とし，双方向 (相互フォロー) も混在させる．
//!
//! これにより辺 `A → B` = 「A が B をフォロー」となり，B の投稿は B のフォロワ
//! (= `in_neighbors(B)`) に届く ([`crate::mechanisms::NetworkInitMechanism`])．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
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

/// 指定モデルで無向トポロジを生成する．
fn generate_undirected(cfg: &Config, ids: &[AgentId], rng: &mut SimRng) -> SocialNetwork {
    match cfg.network {
        NetworkKind::ErdosRenyi => SocialNetwork::erdos_renyi(ids, cfg.er_p, rng),
        NetworkKind::WattsStrogatz => {
            SocialNetwork::watts_strogatz(ids, cfg.ws_k, cfg.ws_beta, rng)
        }
        NetworkKind::BarabasiAlbert => SocialNetwork::barabasi_albert(ids, cfg.ba_m, rng),
    }
}

/// 無向トポロジへフォロー方向を付与して [`DiSocialNetwork`] を構築する．
///
/// 各無向辺 `{a, b}` (a < b; `edges()` は a <= b で正規化) に対し，RNG のコイン投げで
/// 方向を決める: 25% `a→b` のみ，25% `b→a` のみ，50% 双方向 (相互フォロー)．これで
/// 一方向フォローと相互フォローが混在した有向グラフになる．
fn impose_follow_direction(
    undirected: &SocialNetwork,
    ids: &[AgentId],
    rng: &mut SimRng,
) -> DiSocialNetwork {
    let mut di = DiSocialNetwork::empty();
    for &id in ids {
        di.add_node(id);
    }
    for (a, b) in undirected.edges() {
        if a == b {
            continue; // 自己ループは無視．
        }
        // 0,1,2,3 → {a→b}, {b→a}, {双方向}, {双方向}
        match rng.gen_range(0u8..4) {
            0 => di.add_edge(a, b),
            1 => di.add_edge(b, a),
            _ => {
                di.add_edge(a, b);
                di.add_edge(b, a);
            }
        }
    }
    di
}

/// 世界状態を初期化する (有向網構築 + 属性/初期感情/初期態度割当)．
///
/// 属性・初期感情・初期態度は init RNG から決定論的に割り当てる (socsim コア層)．
/// 初期態度の LLM 決定 (論文) は拡張点であり，既定は規則ベース (RNG) を用いる．
/// `seed_posters` 体の発信源は round 0 で必ず投稿する (種まき; outbox へ直接積む)．
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> S3World {
    let ids: Vec<AgentId> = (0..cfg.population as u64).map(AgentId).collect();

    let undirected = generate_undirected(cfg, &ids, rng);
    let net = impose_follow_direction(&undirected, &ids, rng);

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

    let mut sim = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]))
        .add_mechanism(Box::new(NetworkInitMechanism))
        .add_mechanism(Box::new(LLMPerceptionMechanism::new(
            cfg.top_k,
            cfg.llm_perception,
        )))
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
pub fn save_metrics(metrics: &[Metrics], output_dir: &str) {
    let path = format!("{}/metrics.csv", output_dir);
    let file = File::create(&path).expect("metrics.csv の作成に失敗");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    for m in metrics {
        wtr.serialize(m).expect("メトリクス書き込みに失敗");
    }
    wtr.flush().expect("フラッシュに失敗");
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
    let path = format!("{}/run_metadata.json", output_dir);
    let file = File::create(&path).expect("run_metadata.json の作成に失敗");
    serde_json::to_writer_pretty(BufWriter::new(file), &meta)
        .expect("run_metadata.json の書き込みに失敗");
}

/// 出力ディレクトリを作成する．
pub fn ensure_output_dir(output_dir: &str) {
    fs::create_dir_all(output_dir).expect("出力ディレクトリの作成に失敗");
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
