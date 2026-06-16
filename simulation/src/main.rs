//! Gao et al. (2023) "S3: Social-network Simulation System with Large Language
//! Model-Empowered Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`       : 単一設定で有向網上の LLM 駆動 感情/態度/行動伝播を実行する．
//! `sweep`     : ネットワーク種別 × 人口規模 を走査し，最終集団指標を
//!               `sweep_summary.csv` に集計する．
//! `reproduce` : S³ の headline 伝播を一括再現し，observed-vs-paper 照合
//!               (態度上昇・カスケード成長・行動採用・感情分布 MSED・態度時系列相関)
//!               と古典ベースライン (LT/IC/Voter/DeGroot) 比較を出力する．
//! `baseline`  : 古典的拡散・意見ダイナミクスのベースラインを単独で実行する．

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

use s3_simulation::baseline::{parse_baseline, run_baseline, BaselineParams};
use s3_simulation::config::{parse_network, Config, LlmSettings, NetworkKind};
use s3_simulation::llm::{build_live_client, wrap_client, S3Client};
use s3_simulation::reproduce::run_reproduce;
use s3_simulation::simulation::{ensure_output_dir, run, save_metrics, save_run_metadata};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

// ---------------------------------------------------------------------------
// CLI 定義
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "s3",
    about = "Gao et al. (2023) S3: Social-network Simulation System with LLM-Empowered Agents — 再現実験"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で有向網上の LLM 駆動 感情/態度/行動伝播を実行する．
    Run(RunArgs),
    /// ネットワーク種別 × 人口規模 を走査し，最終集団指標を集計する．
    Sweep(SweepArgs),
    /// S³ の headline 伝播を一括再現し，論文帯照合 + 古典ベースライン比較を出力する．
    Reproduce(ReproduceArgs),
    /// 古典的拡散・意見ダイナミクスのベースライン (lt/ic/voter/degroot) を単独実行する．
    Baseline(BaselineArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// ネットワーク種別 (er / ws / ba)．
    #[arg(long, default_value = "ba")]
    network: String,

    /// 人口規模 N (= ノード数)．
    #[arg(long, default_value_t = 100)]
    population: usize,

    /// ER の接続確率 p．
    #[arg(long, default_value_t = 0.05)]
    p: f64,

    /// WS の各ノードの初期次数 k (偶数)．
    #[arg(long, default_value_t = 4)]
    ws_k: usize,

    /// WS の再配線確率 β．
    #[arg(long, default_value_t = 0.1)]
    ws_beta: f64,

    /// WS の無向→有向変換で双方向 (相互フォロー) になる確率 p_mutual (ER/BA は不使用)．
    #[arg(long, default_value_t = 0.5)]
    ws_p_mutual: f64,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 3)]
    m: usize,

    /// 伝播ラウンド数 T．
    #[arg(long, default_value_t = 20)]
    rounds: usize,

    /// Perception で選ぶ重要メッセージ件数 K．
    #[arg(long, default_value_t = 3)]
    top_k: usize,

    /// LLM 駆動 Perception を使うか (既定 false = 規則ベース; 拡張スタブ)．
    #[arg(long, default_value_t = false)]
    llm_perception: bool,

    /// round 0 で必ず投稿する発信源エージェント数．
    #[arg(long, default_value_t = 3)]
    seed_posters: usize,

    /// 収束判定しきい値 (positive 態度割合の round 間変化)．
    #[arg(long, default_value_t = 1e-9)]
    tol: f64,

    /// 乱数シード (省略時はランダム; socsim コア層のみ支配)．
    #[arg(long)]
    seed: Option<u64>,

    /// LLM 生成温度 (既定 0.0; 再現性のため)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (バックエンドへ渡す)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (既定 .llm_cache/cache.json)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// カンマ区切りのネットワーク種別リスト．
    #[arg(long, default_value = "er,ws,ba")]
    network: String,

    /// カンマ区切りの人口規模リスト．
    #[arg(long, default_value = "50,100,200")]
    population_values: String,

    /// ER の接続確率 p．
    #[arg(long, default_value_t = 0.05)]
    p: f64,

    /// WS の各ノードの初期次数 k．
    #[arg(long, default_value_t = 4)]
    ws_k: usize,

    /// WS の再配線確率 β．
    #[arg(long, default_value_t = 0.1)]
    ws_beta: f64,

    /// WS の無向→有向変換で双方向 (相互フォロー) になる確率 p_mutual (ER/BA は不使用)．
    #[arg(long, default_value_t = 0.5)]
    ws_p_mutual: f64,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 3)]
    m: usize,

    /// 伝播ラウンド数 T．
    #[arg(long, default_value_t = 20)]
    rounds: usize,

    /// Perception で選ぶ重要メッセージ件数 K．
    #[arg(long, default_value_t = 3)]
    top_k: usize,

    /// round 0 で必ず投稿する発信源エージェント数．
    #[arg(long, default_value_t = 3)]
    seed_posters: usize,

    /// 各条件あたりの独立試行数．
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// 収束判定しきい値．
    #[arg(long, default_value_t = 1e-9)]
    tol: f64,

    /// 乱数シード基点 (各試行は derive により独立化する)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// LLM 生成温度．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (sweep 全体で共有しヒット率を高める)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// ネットワーク種別 (er / ws / ba)．
    #[arg(long, default_value = "ba")]
    network: String,

    /// 人口規模 N．
    #[arg(long, default_value_t = 200)]
    population: usize,

    /// ER の接続確率 p．
    #[arg(long, default_value_t = 0.05)]
    p: f64,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 3)]
    m: usize,

    /// 伝播ラウンド数 T．
    #[arg(long, default_value_t = 20)]
    rounds: usize,

    /// Perception で選ぶ重要メッセージ件数 K．
    #[arg(long, default_value_t = 3)]
    top_k: usize,

    /// round 0 で必ず投稿する発信源エージェント数．
    #[arg(long, default_value_t = 3)]
    seed_posters: usize,

    /// LLM 駆動 Perception を使うか (既定 false = 規則ベース)．
    #[arg(long, default_value_t = false)]
    llm_perception: bool,

    /// 乱数シード (socsim コア層を支配; 既定 42)．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// 収束判定しきい値 (態度割合の round 間変化; reproduce は早期停止させない既定)．
    #[arg(long, default_value_t = 1e-12)]
    tol: f64,

    /// LLM を使わず scripted mock で実行する (オフライン; live LLM 不要)．
    #[arg(long, default_value_t = false)]
    mock: bool,

    /// 短縮実行 (population/rounds を小さめに上書きしてスモークする)．
    #[arg(long, default_value_t = false)]
    quick: bool,

    /// LLM 生成温度 (live 時)．
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// LLM 生成シード (live 時)．
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,

    /// プロンプト→応答キャッシュの保存先 (live 時)．
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,

    /// LT の一様しきい値 θ．
    #[arg(long, default_value_t = 0.3)]
    lt_theta: f64,

    /// IC の 1 辺あたり感染確率 p．
    #[arg(long, default_value_t = 0.15)]
    ic_p: f64,

    /// DeGroot の自己重み．
    #[arg(long, default_value_t = 0.5)]
    degroot_self_weight: f64,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct BaselineArgs {
    /// ベースラインモデル (lt / ic / voter / degroot)．
    #[arg(long, default_value = "lt")]
    model: String,

    /// ネットワーク種別 (er / ws / ba)．
    #[arg(long, default_value = "ba")]
    network: String,

    /// 人口規模 N．
    #[arg(long, default_value_t = 200)]
    population: usize,

    /// ER の接続確率 p．
    #[arg(long, default_value_t = 0.05)]
    p: f64,

    /// BA の新規ノードあたりの結合数 m．
    #[arg(long, default_value_t = 3)]
    m: usize,

    /// 伝播ラウンド数 T．
    #[arg(long, default_value_t = 20)]
    rounds: usize,

    /// round 0 の種ノード数 (S³ の seed_posters と同集合)．
    #[arg(long, default_value_t = 3)]
    seed_posters: usize,

    /// 乱数シード．
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// 収束判定しきい値 (DeGroot の意見変化)．
    #[arg(long, default_value_t = 1e-9)]
    tol: f64,

    /// LT の一様しきい値 θ．
    #[arg(long, default_value_t = 0.3)]
    lt_theta: f64,

    /// IC の 1 辺あたり感染確率 p．
    #[arg(long, default_value_t = 0.15)]
    ic_p: f64,

    /// DeGroot の自己重み．
    #[arg(long, default_value_t = 0.5)]
    degroot_self_weight: f64,

    /// 結果出力ベースディレクトリ．
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

/// reproduce / mock 用の scripted クライアント (live LLM 不要のオフライン擬似挙動)．
///
/// - **Perception プロンプト** (`--llm-perception` 経路) には番号列を返す．
/// - **Contagion プロンプト** には，受信があれば肯定的態度・投稿に寄せて伝播曲線に
///   変化を出す．感情はプロフィール (年齢) に応じて calm/moderate/intense へ
///   分散させ，論文の定性的な感情分布の形状 (中〜強感情が支配的) を代理する
///   (全員 intense へ潰れず MSED 整合が安定する)．
///
/// 決定論的 (同一プロンプト→同一応答) なので二層決定論を保つ．
fn scripted_mock_client() -> S3Client {
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        // Perception プロンプト: 先頭候補を選ぶ番号列を返す．
        if prompt.contains("Answer with ONLY the numbers") {
            return "1, 2, 3".to_string();
        }
        let received = prompt.matches("user ").count();
        // プロフィールの年齢で感情を分散させる ("a 27-year-old ..." の数字を拾う)．
        let age: u32 = prompt
            .split("-year-old")
            .next()
            .and_then(|s| {
                s.rsplit(|c: char| !c.is_ascii_digit())
                    .find(|t| !t.is_empty())
            })
            .and_then(|s| s.parse().ok())
            .unwrap_or(30);
        let emotion = if received == 0 {
            "calm"
        } else {
            match age % 3 {
                0 => "moderate",
                1 => "intense",
                _ => "moderate",
            }
        };
        let (attitude, behavior, content) = if received >= 1 {
            ("positive", "post", "I agree, this is important to share.")
        } else {
            ("negative", "inactive", "")
        };
        format!(
            "EMOTION: {emotion}\nATTITUDE: {attitude}\nBEHAVIOR: {behavior}\nCONTENT: {content}"
        )
    });
    wrap_client(backend, PromptCache::in_memory())
}

/// `sweep_summary.csv` の 1 行．
#[derive(serde::Serialize)]
struct SweepRow {
    network: String,
    population: usize,
    run: usize,
    seed: u64,
    converged: bool,
    final_round: usize,
    final_attitude_positive_frac: f64,
    final_emotion_calm: f64,
    final_emotion_moderate: f64,
    final_emotion_intense: f64,
    final_behavior_adoption_rate: f64,
    final_info_cascade_size: usize,
    cache_hit_rate: f64,
}

/// `sweep_config.json` の構造体．
#[derive(serde::Serialize)]
struct SweepConfigJson {
    command: &'static str,
    network_values: Vec<String>,
    population_values: Vec<usize>,
    rounds: usize,
    top_k: usize,
    seed_posters: usize,
    runs: usize,
    tol: f64,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

/// 派生シードのラベルに使う文字列ハッシュ (explicit identity)．
fn label_hash(label: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in label.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// カンマ区切り文字列を trim 済みの非空リストへ．
fn split_csv(s: &str) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(args: RunArgs) {
    let network = parse_network(&args.network).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    let cfg = Config {
        network,
        population: args.population,
        er_p: args.p,
        ws_k: args.ws_k,
        ws_beta: args.ws_beta,
        ws_p_mutual: args.ws_p_mutual,
        ba_m: args.m,
        rounds: args.rounds,
        top_k: args.top_k,
        llm_perception: args.llm_perception,
        seed_posters: args.seed_posters,
        tol: args.tol,
        seed: args.seed,
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            cache_path: Some(args.cache_path.clone()),
        },
        output_dir: output_dir.clone(),
    };

    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    ensure_output_dir(&cfg.output_dir);

    println!("=== Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 再現実験 ===");
    println!(
        "network: {} | population: {} | rounds: {} | top_k: {} | seed_posters: {}",
        cfg.network.label(),
        cfg.population,
        cfg.rounds,
        cfg.top_k,
        cfg.seed_posters,
    );
    println!(
        "seed: {:?} | LLM: temp={} llm_seed={} cache={}",
        cfg.seed, cfg.llm.temperature, cfg.llm.seed, args.cache_path
    );
    println!("出力先: {}", cfg.output_dir);
    println!("-----------------------------------------------------------------");

    let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));

    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json (pretty-print JSON; socsim_results::write_json に委譲)．
    {
        let path = format!("{}/config.json", cfg.output_dir);
        write_json(&cfg.to_run_config_json(), &path).expect("config.json の書き込みに失敗");
    }

    // latest シンボリックリンクを再作成する (best-effort; 従来同様エラーは無視)．
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

    let last = result.metrics_history.last().unwrap();
    println!(
        "収束: {} | round: {}",
        if result.converged { "Yes" } else { "No" },
        result.final_round
    );
    println!(
        "最終 positive 態度割合: {:.3} | 感情 (calm/mod/int): {:.2}/{:.2}/{:.2}",
        last.attitude_positive_frac, last.emotion_calm, last.emotion_moderate, last.emotion_intense,
    );
    println!(
        "行動採用率: {:.3} | 情報カスケード規模: {}",
        last.behavior_adoption_rate, last.info_cascade_size
    );
    println!(
        "LLM 呼び出し: {} 回 | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("メトリクス → {}/metrics.csv", cfg.output_dir);
    println!("LLM メタ   → {}/run_metadata.json", cfg.output_dir);
    println!("設定       → {}/config.json", cfg.output_dir);
}

// ---------------------------------------------------------------------------
// sweep
// ---------------------------------------------------------------------------

fn cmd_sweep(args: SweepArgs) {
    let networks: Vec<NetworkKind> = split_csv(&args.network)
        .iter()
        .map(|s| parse_network(s).unwrap_or_else(|e| panic!("{}", e)))
        .collect();
    let populations: Vec<usize> = split_csv(&args.population_values)
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .unwrap_or_else(|_| panic!("不正な人口規模: {s}"))
        })
        .collect();

    let timestamp = timestamp();
    let sweep_dir = format!("{}/{}_sweep", args.output_dir, timestamp);
    fs::create_dir_all(&sweep_dir).expect("sweep ディレクトリの作成に失敗");
    if let Some(parent) = Path::new(&args.cache_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    let n_total = networks.len() * populations.len() * args.runs;

    println!("=== Gao et al. (2023) S3 パラメータスイープ (network × population) ===");
    println!(
        "network: {} 種 | population: {} 種 | 試行: {} | 合計: {} 実行",
        networks.len(),
        populations.len(),
        args.runs,
        n_total,
    );
    println!("出力先: {}", sweep_dir);
    println!("-----------------------------------------------------------------");

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);
    let mut done = 0usize;

    for &network in &networks {
        for &population in &populations {
            for run_idx in 0..args.runs {
                let seed = socsim_core::derive_seed(
                    args.seed,
                    &[
                        label_hash(network.label()),
                        population as u64,
                        run_idx as u64,
                    ],
                );

                let cfg = Config {
                    network,
                    population,
                    er_p: args.p,
                    ws_k: args.ws_k,
                    ws_beta: args.ws_beta,
                    ws_p_mutual: args.ws_p_mutual,
                    ba_m: args.m,
                    rounds: args.rounds,
                    top_k: args.top_k,
                    llm_perception: false,
                    seed_posters: args.seed_posters,
                    tol: args.tol,
                    seed: Some(seed),
                    llm: LlmSettings {
                        temperature: args.temperature,
                        seed: args.llm_seed,
                        cache_path: Some(args.cache_path.clone()),
                    },
                    output_dir: sweep_dir.clone(),
                };

                let result = run(&cfg).unwrap_or_else(|e| panic!("実行に失敗: {}", e));
                let last = result.metrics_history.last().unwrap();

                summary_rows.push(SweepRow {
                    network: network.label().to_string(),
                    population,
                    run: run_idx,
                    seed,
                    converged: result.converged,
                    final_round: result.final_round,
                    final_attitude_positive_frac: last.attitude_positive_frac,
                    final_emotion_calm: last.emotion_calm,
                    final_emotion_moderate: last.emotion_moderate,
                    final_emotion_intense: last.emotion_intense,
                    final_behavior_adoption_rate: last.behavior_adoption_rate,
                    final_info_cascade_size: last.info_cascade_size,
                    cache_hit_rate: result.metadata.cache_hit_rate(),
                });

                done += 1;
            }
            println!(
                "[{}/{}] network={} population={} 完了 ({} 試行)",
                done,
                n_total,
                network.label(),
                population,
                args.runs,
            );
        }
    }

    // sweep_summary.csv (各行を serialize; socsim_results::write_csv に委譲)．
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        write_csv(&summary_rows, &path).expect("sweep_summary.csv の書き込みに失敗");
    }

    // sweep_config.json
    {
        let config_json = SweepConfigJson {
            command: "sweep",
            network_values: split_csv(&args.network),
            population_values: populations.clone(),
            rounds: args.rounds,
            top_k: args.top_k,
            seed_posters: args.seed_posters,
            runs: args.runs,
            tol: args.tol,
            seed: args.seed,
            llm_temperature: args.temperature,
            llm_seed: args.llm_seed,
        };
        let path = format!("{}/sweep_config.json", sweep_dir);
        write_json(&config_json, &path).expect("sweep_config.json の書き込みに失敗");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &format!("{}_sweep", timestamp));

    println!("=================================================================");
    println!("スイープ完了: {} 実行", n_total);
    println!("-----------------------------------------------------------------");
    println!("ネットワーク別の平均 positive 態度割合 / 情報カスケード規模:");
    for &network in &networks {
        let rows: Vec<&SweepRow> = summary_rows
            .iter()
            .filter(|r| r.network == network.label())
            .collect();
        if rows.is_empty() {
            continue;
        }
        let avg_pos = rows
            .iter()
            .map(|r| r.final_attitude_positive_frac)
            .sum::<f64>()
            / rows.len() as f64;
        let avg_casc = rows
            .iter()
            .map(|r| r.final_info_cascade_size)
            .sum::<usize>() as f64
            / rows.len() as f64;
        println!(
            "  {:<3} → positivē = {:.3} | cascadē = {:.1}",
            network.label(),
            avg_pos,
            avg_casc
        );
    }
    println!("-----------------------------------------------------------------");
    println!("サマリ → {}/sweep_summary.csv", sweep_dir);
    println!("設定   → {}/sweep_config.json", sweep_dir);
}

// ---------------------------------------------------------------------------
// reproduce
// ---------------------------------------------------------------------------

fn cmd_reproduce(args: ReproduceArgs) {
    let network = parse_network(&args.network).unwrap_or_else(|e| panic!("{}", e));

    // --quick はスモーク用に規模を絞る．
    let (population, rounds) = if args.quick {
        (args.population.min(40), args.rounds.min(8))
    } else {
        (args.population, args.rounds)
    };

    let timestamp = timestamp();
    let output_dir = format!("{}/reproduce_{}", args.output_dir, timestamp);
    let figures_dir = format!("{}/figures", output_dir);
    fs::create_dir_all(&figures_dir).expect("reproduce ディレクトリの作成に失敗");

    let cfg = Config {
        network,
        population,
        er_p: args.p,
        ba_m: args.m,
        rounds,
        top_k: args.top_k,
        llm_perception: args.llm_perception,
        seed_posters: args.seed_posters,
        tol: args.tol,
        seed: Some(args.seed),
        llm: LlmSettings {
            temperature: args.temperature,
            seed: args.llm_seed,
            // mock はキャッシュ永続化しない (in-memory)．live のみ cache_path を持つ．
            cache_path: if args.mock {
                None
            } else {
                Some(args.cache_path.clone())
            },
        },
        output_dir: output_dir.clone(),
        ..Config::default()
    };

    let params = BaselineParams {
        lt_theta: args.lt_theta,
        ic_p: args.ic_p,
        degroot_self_weight: args.degroot_self_weight,
        ..BaselineParams::default()
    };

    println!("=== Gao et al. (2023) S³ 一括再現 (reproduce) ===");
    println!(
        "network: {} | population: {} | rounds: {} | seed: {} | mock: {} | quick: {} | llm_perception: {}",
        cfg.network.label(),
        cfg.population,
        cfg.rounds,
        args.seed,
        args.mock,
        args.quick,
        cfg.llm_perception,
    );
    println!("出力先: {}", output_dir);
    println!("-----------------------------------------------------------------");

    // mock (オフライン) なら scripted client，live なら Ollama→OpenAI フォールバック．
    let client = if args.mock {
        scripted_mock_client()
    } else {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        build_live_client(&cfg.llm).unwrap_or_else(|e| panic!("LLM クライアント構築に失敗: {e}"))
    };

    let out = run_reproduce(&cfg, client, args.mock, &params)
        .unwrap_or_else(|e| panic!("reproduce 失敗: {e}"));

    // S³ の round 別集団指標 (visualize 互換: metrics.csv 名でも置く)．
    save_metrics(&out.s3.metrics_history, &output_dir);
    write_csv(
        &out.s3.metrics_history,
        format!("{}/s3_metrics.csv", output_dir),
    )
    .expect("s3_metrics.csv の書き込みに失敗");

    // 各ベースラインの round 別指標．
    for (label, history) in &out.baseline_histories {
        let path = format!("{}/baseline_{}.csv", output_dir, label);
        write_csv(history, &path).expect("baseline CSV の書き込みに失敗");
    }

    // config.json (visualize のネットワーク描画用)．
    write_json(
        &cfg.to_run_config_json(),
        format!("{}/config.json", output_dir),
    )
    .expect("config.json の書き込みに失敗");

    // reproduce_summary.json (observed-vs-paper + ベースライン比較)．
    write_json(
        &out.summary,
        format!("{}/reproduce_summary.json", output_dir),
    )
    .expect("reproduce_summary.json の書き込みに失敗");

    let _ = refresh_latest_symlink(&args.output_dir, &format!("reproduce_{}", timestamp));

    // --- 観測-参照-PASS の表示 ---
    println!("observed-vs-paper (S³ headline 伝播):");
    for c in &out.summary.checks {
        println!(
            "  {:<24} = {:>8.3}   ({} {:.2}: {})",
            c.indicator,
            c.observed,
            c.direction,
            c.paper,
            if c.pass { "PASS" } else { "off" },
        );
    }
    println!(
        "  → {}/{} PASS ({})",
        out.summary.passed,
        out.summary.total,
        if out.summary.all_pass {
            "all PASS"
        } else {
            "review"
        },
    );
    println!("-----------------------------------------------------------------");
    println!(
        "古典ベースライン比較 (同一網・同一シード; S³ 最終 active={:.3} / 到達割合={:.3}):",
        out.summary.s3_final_active_frac, out.summary.s3_final_reached_frac,
    );
    for b in &out.summary.baselines {
        println!(
            "  {:<8} 最終 active={:.3} | 平均意見={:.3} | 到達={:>4} | round={}",
            b.model, b.final_active_frac, b.final_mean_opinion, b.final_reached, b.final_round,
        );
    }
    println!("-----------------------------------------------------------------");
    println!(
        "LLM: model={} cache-hit={:.1}%",
        out.summary.llm_model,
        out.summary.cache_hit_rate * 100.0
    );
    println!("サマリ → {}/reproduce_summary.json", output_dir);
    println!(
        "可視化 → uv run s3-tools reproduce --results-dir {}",
        output_dir
    );
}

// ---------------------------------------------------------------------------
// baseline
// ---------------------------------------------------------------------------

fn cmd_baseline(args: BaselineArgs) {
    let network = parse_network(&args.network).unwrap_or_else(|e| panic!("{}", e));
    let model = parse_baseline(&args.model).unwrap_or_else(|e| panic!("{}", e));

    let timestamp = timestamp();
    let output_dir = format!("{}/baseline_{}_{}", args.output_dir, args.model, timestamp);
    fs::create_dir_all(&output_dir).expect("baseline ディレクトリの作成に失敗");

    let cfg = Config {
        network,
        population: args.population,
        er_p: args.p,
        ba_m: args.m,
        rounds: args.rounds,
        seed_posters: args.seed_posters,
        tol: args.tol,
        seed: Some(args.seed),
        output_dir: output_dir.clone(),
        ..Config::default()
    };
    let params = BaselineParams {
        lt_theta: args.lt_theta,
        ic_p: args.ic_p,
        degroot_self_weight: args.degroot_self_weight,
        ..BaselineParams::default()
    };

    println!(
        "=== Gao et al. (2023) 古典ベースライン ({}) ===",
        model.label()
    );
    println!(
        "network: {} | population: {} | rounds: {} | seed_posters: {} | seed: {}",
        cfg.network.label(),
        cfg.population,
        cfg.rounds,
        cfg.seed_posters,
        args.seed,
    );
    println!("出力先: {}", output_dir);
    println!("-----------------------------------------------------------------");

    let result = run_baseline(&cfg, model, &params);

    write_csv(
        &result.history,
        format!("{}/baseline_{}.csv", output_dir, model.label()),
    )
    .expect("baseline CSV の書き込みに失敗");
    write_json(
        &cfg.to_run_config_json(),
        format!("{}/config.json", output_dir),
    )
    .expect("config.json の書き込みに失敗");

    let _ = refresh_latest_symlink(
        &args.output_dir,
        &format!("baseline_{}_{}", args.model, timestamp),
    );

    let last = result.last();
    println!(
        "最終 active 割合: {:.3} | 平均意見: {:.3} | 累積到達: {} | round: {} | 収束: {}",
        last.active_frac,
        last.mean_opinion,
        last.cumulative_reached,
        result.final_round,
        if result.converged { "Yes" } else { "No" },
    );
    println!("メトリクス → {}/baseline_{}.csv", output_dir, model.label());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
        Commands::Baseline(args) => cmd_baseline(args),
    }
}
