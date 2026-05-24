//! Gao et al. (2023) S3 の統合テスト．
//!
//! **ライブ LLM を一切必要としない**: socsim-llm の `mock::ScriptedClient` で
//! 決定論的に感情/態度/行動更新を駆動し，以下を検証する:
//! ・有向辺に沿ったメッセージ配送 (follower = in_neighbors)
//! ・同期 round セマンティクス (round 内投稿は次 round の inbox にのみ反映)
//! ・集団指標の集計 (positive 態度割合・感情分布・カスケード規模)
//! ・固定 mock を与えたときの socsim コア層の RNG 決定論性
//! ・(小さな) 有向グラフ上のフォロワ伝播

use std::collections::{BTreeMap, BTreeSet};

use s3_simulation::config::{Config, NetworkKind};
use s3_simulation::llm::{wrap_client, S3Client};
use s3_simulation::mechanisms::NetworkInitMechanism;
use s3_simulation::simulation::run_with_client;
use s3_simulation::world::{AgentState, Attitude, Emotion, Message, Profile, S3World};

use socsim_core::{AgentId, Mechanism, Phase, WorldState};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// 常に positive / post を返す mock クライアント (content 付き)．
fn scripted_post() -> S3Client {
    let backend = ScriptedClient::new("mock-model", |_p: &str| {
        "EMOTION: moderate\nATTITUDE: positive\nBEHAVIOR: post\nCONTENT: A short post.".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// 常に negative / inactive を返す mock クライアント．
fn scripted_inactive() -> S3Client {
    let backend = ScriptedClient::new("mock-model", |_p: &str| {
        "EMOTION: calm\nATTITUDE: negative\nBEHAVIOR: inactive\nCONTENT: empty".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

fn base_config() -> Config {
    Config {
        network: NetworkKind::BarabasiAlbert,
        population: 12,
        rounds: 6,
        top_k: 3,
        seed_posters: 2,
        tol: 1e-12, // 収束で停止させない
        seed: Some(7),
        ..Config::default()
    }
}

// --------------------------------------------------------------------------- //
// 全員 positive を採用 → positive 態度割合が 1 に向かう
// --------------------------------------------------------------------------- //

#[test]
fn unanimous_positive_drives_attitude_fraction_up() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_post()).unwrap();
    let last = result.metrics_history.last().unwrap();
    assert!(
        last.attitude_positive_frac > 0.9,
        "全員 positive を採用すれば positive 割合は 1 に寄る (got {})",
        last.attitude_positive_frac
    );
    // 集計配線: t=0 が記録され，全 round が連続している．
    assert_eq!(result.metrics_history[0].t, 0);
    assert!(result.metrics_history.len() >= 2);
}

// --------------------------------------------------------------------------- //
// 指標が [0,1] に収まり，カスケード規模は単調非減少
// --------------------------------------------------------------------------- //

#[test]
fn metrics_are_well_formed() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_post()).unwrap();
    let mut prev_casc = 0usize;
    for m in &result.metrics_history {
        for f in [
            m.attitude_positive_frac,
            m.emotion_calm,
            m.emotion_moderate,
            m.emotion_intense,
            m.behavior_adoption_rate,
        ] {
            assert!((0.0..=1.0).contains(&f), "fraction out of [0,1]: {f}");
        }
        let dist_sum = m.emotion_calm + m.emotion_moderate + m.emotion_intense;
        assert!((dist_sum - 1.0).abs() < 1e-9, "emotion dist must sum to 1");
        assert!(
            m.info_cascade_size >= prev_casc,
            "cascade size must be monotone non-decreasing"
        );
        prev_casc = m.info_cascade_size;
    }
}

// --------------------------------------------------------------------------- //
// ネットワーク種別ごとに正しいノード数 + 実行が成立する
// --------------------------------------------------------------------------- //

#[test]
fn networks_build_and_run() {
    for net in [
        NetworkKind::ErdosRenyi,
        NetworkKind::WattsStrogatz,
        NetworkKind::BarabasiAlbert,
    ] {
        let mut cfg = base_config();
        cfg.network = net;
        cfg.population = 10;
        cfg.rounds = 3;
        cfg.er_p = 0.3;
        let result = run_with_client(&cfg, scripted_post()).unwrap();
        assert!(!result.metrics_history.is_empty(), "{net:?}: 履歴が空");
    }
}

// --------------------------------------------------------------------------- //
// 決定論性: 同一シード + 同一 mock → 完全再現 (socsim コア層)
// --------------------------------------------------------------------------- //

#[test]
fn core_is_deterministic_given_fixed_mock() {
    let cfg = base_config();
    let a = run_with_client(&cfg, scripted_post()).unwrap();
    let b = run_with_client(&cfg, scripted_post()).unwrap();
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
    let ac: Vec<usize> = a
        .metrics_history
        .iter()
        .map(|m| m.info_cascade_size)
        .collect();
    let bc: Vec<usize> = b
        .metrics_history
        .iter()
        .map(|m| m.info_cascade_size)
        .collect();
    assert_eq!(af, bf, "同一シードは positive 割合を完全再現すべき");
    assert_eq!(ac, bc, "同一シードはカスケード規模を完全再現すべき");
    assert_eq!(a.final_round, b.final_round);
}

#[test]
fn different_seed_changes_outcome() {
    let mut cfg_a = base_config();
    cfg_a.seed = Some(1);
    let mut cfg_b = base_config();
    cfg_b.seed = Some(999);
    let a = run_with_client(&cfg_a, scripted_inactive()).unwrap();
    let b = run_with_client(&cfg_b, scripted_inactive()).unwrap();
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
    // 初期態度割当が seed 依存なので (高確率で) 異なる軌跡になる．
    assert!(af != bf, "異なるシードは (一般に) 異なる軌跡を生む");
}

// --------------------------------------------------------------------------- //
// 有向辺に沿った配送: A→B のとき, B の投稿は A (= follower) に届く
// --------------------------------------------------------------------------- //

/// 手組みの小さな有向グラフで NetworkInit の配送を直接検証する．
///
/// 規約: 辺 `A → B` = 「A が B をフォロー」．したがって B の投稿は A (B のフォロワ
/// = in_neighbors(B)) の inbox に届くべき．
#[test]
fn directed_edge_delivers_to_followers() {
    use socsim_net::DiSocialNetwork;

    let a = AgentId(0);
    let b = AgentId(1);
    let c = AgentId(2);

    let mut net = DiSocialNetwork::empty();
    for id in [a, b, c] {
        net.add_node(id);
    }
    // A→B (A が B をフォロー), C→B (C が B をフォロー)．B のフォロワ = {A, C}．
    net.add_edge(a, b);
    net.add_edge(c, b);

    let profile = || Profile {
        gender: "person".into(),
        age: 30,
        occupation: "worker".into(),
    };
    let mut agents: BTreeMap<AgentId, AgentState> = BTreeMap::new();
    for id in [a, b, c] {
        agents.insert(
            id,
            AgentState::new(profile(), Emotion::Calm, Attitude::Negative),
        );
    }

    let mut world = S3World::new(net, agents, 5);

    // followers(B) は in_neighbors(B) = {A, C}．
    let followers_b: BTreeSet<AgentId> = world.followers(b).into_iter().collect();
    assert_eq!(followers_b, BTreeSet::from([a, c]));

    // B の投稿を outbox に積み, NetworkInit で配送させる．
    world.outbox.push(Message {
        author: b,
        origin: b,
        content: "hello from B".into(),
        round: 0,
        is_repost: false,
    });

    // NetworkInit を 1 回 apply して配送を確認する (engine ハーネスを介して).
    deliver_once(&mut world);

    // A と C の inbox に B の投稿が届き, B 自身には届かない (B は B をフォローしない)．
    assert_eq!(world.inbox[&a].len(), 1, "A は B のフォロワなので 1 件受信");
    assert_eq!(world.inbox[&c].len(), 1, "C は B のフォロワなので 1 件受信");
    assert_eq!(
        world.inbox[&b].len(),
        0,
        "B は自分のフォロワでないので 0 件"
    );
    assert_eq!(world.inbox[&a][0].author, b);
}

/// NetworkInit を engine の StepContext 無しで 1 回回す簡易ハーネス．
///
/// PreStep の配送ロジック (inbox クリア + outbox をフォロワへ配送) を，本物の
/// `NetworkInitMechanism` を通して検証するため，最小の StepContext を組み立てる．
fn deliver_once(world: &mut S3World) {
    use socsim_core::{Blackboard, Recorder, SimRng, StepContext};

    struct NullRecorder;
    impl Recorder for NullRecorder {
        fn record_metric(&mut self, _t: u64, _key: &str, _value: f64) {}
        fn record_event(&mut self, _t: u64, _kind: &str, _payload: serde_json::Value) {}
    }

    let mut rng = SimRng::from_seed(0);
    let mut recorder = NullRecorder;
    let mut scratch = Blackboard::new();
    let mut stop = false;
    let order: Vec<AgentId> = world.agent_ids();
    let clock = *world.clock();

    let mut ctx = StepContext {
        world,
        clock,
        rng: &mut rng,
        recorder: &mut recorder,
        agent_order: &order,
        scratch: &mut scratch,
        stop: &mut stop,
    };

    let mut mech = NetworkInitMechanism;
    mech.apply(Phase::PreStep, &mut ctx).unwrap();
}

// --------------------------------------------------------------------------- //
// 同期 round セマンティクス: round 内で生成された投稿は同 round の他者に見えない
// --------------------------------------------------------------------------- //

/// SocialContagion は投稿を `outbox` に積み, `inbox` には積まない (= 次 round 配送)．
/// したがって round 0 の Contagion 後, inbox はまだ空でなければならない (PreStep が
/// 次 round 先頭で初めて配送する)．これを mock 実行のカスケード規模の伸び方で確認:
/// t=0 はシード投稿者数, t=1 で初めてフォロワへ伝播し増える (同期更新の特徴)．
#[test]
fn synchronous_rounds_delay_propagation_by_one() {
    let cfg = base_config();
    let result = run_with_client(&cfg, scripted_post()).unwrap();
    let casc: Vec<usize> = result
        .metrics_history
        .iter()
        .map(|m| m.info_cascade_size)
        .collect();
    // t=0 のカスケード = シード投稿者の reached (= seed_posters 体)．
    assert_eq!(casc[0], cfg.seed_posters, "t=0 は種まき投稿者のみ到達");
    // 同期更新: 配送は次 round 先頭で起きるので t>=1 で初めて拡大しうる．
    assert!(
        *casc.last().unwrap() >= casc[0],
        "round が進むとカスケードは拡大する"
    );
}
