//! Gao et al. (2023) "S3: Social-network Simulation System with Large Language
//! Model-Empowered Agents" — 再現実験の CLI エントリポイント．
//!
//! `run`   : 単一設定で有向網上の LLM 駆動 感情/態度/行動伝播を実行する．
//! `sweep` : ネットワーク種別 × 人口規模 を走査し，最終集団指標を
//!           `sweep_summary.csv` に集計する．
//!
//! Phase 3 の `reproduce` (実データ MSED/Cor 整合・LT/IC/Voter/DeGroot ベース
//! ライン比較) は未実装 (拡張点)．

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use chrono::Local;
use clap::{Parser, Subcommand};
use csv::Writer;

use s3_simulation::config::{parse_network, Config, LlmSettings, NetworkKind};
use s3_simulation::simulation::{ensure_output_dir, run, save_metrics, save_run_metadata};

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
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// 単一設定で有向網上の LLM 駆動 感情/態度/行動伝播を実行する．
    Run(RunArgs),
    /// ネットワーク種別 × 人口規模 を走査し，最終集団指標を集計する．
    Sweep(SweepArgs),
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

// ---------------------------------------------------------------------------
// 補助
// ---------------------------------------------------------------------------

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

/// latest シンボリックリンクを (再) 作成する．
fn refresh_latest(output_dir: &str, target: &str) {
    let symlink_path = Path::new(output_dir).join("latest");
    if symlink_path.is_symlink() {
        let _ = fs::remove_file(&symlink_path);
    }
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(target, &symlink_path);
    }
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

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);

    let cfg = Config {
        network,
        population: args.population,
        er_p: args.p,
        ws_k: args.ws_k,
        ws_beta: args.ws_beta,
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

    // config.json
    {
        let path = format!("{}/config.json", cfg.output_dir);
        let file = File::create(&path).expect("config.json の作成に失敗");
        serde_json::to_writer_pretty(BufWriter::new(file), &cfg.to_run_config_json())
            .expect("config.json の書き込みに失敗");
    }

    refresh_latest(&args.output_dir, &timestamp);

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

    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
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

    // sweep_summary.csv
    {
        let path = format!("{}/sweep_summary.csv", sweep_dir);
        let file = File::create(&path).expect("sweep_summary.csv の作成に失敗");
        let mut wtr = Writer::from_writer(BufWriter::new(file));
        for row in &summary_rows {
            wtr.serialize(row).expect("サマリ行の書き込みに失敗");
        }
        wtr.flush().expect("フラッシュに失敗");
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
        let file = File::create(&path).expect("sweep_config.json の作成に失敗");
        serde_json::to_writer_pretty(BufWriter::new(file), &config_json)
            .expect("sweep_config.json の書き込みに失敗");
    }

    refresh_latest(&args.output_dir, &format!("{}_sweep", timestamp));

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
// main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
    }
}
