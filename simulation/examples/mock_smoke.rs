//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (metrics.csv / run_metadata.json) と Python 可視化を検証する
//! ための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に感情/態度/
//! 行動更新を駆動し，本番 `run` と同じ writer で結果を書き出す．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;
use std::fs;

use chrono::Local;

use s3_simulation::config::Config;
use s3_simulation::llm::wrap_client;
use s3_simulation::simulation::{
    ensure_output_dir, run_with_client, save_metrics, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        network: s3_simulation::config::NetworkKind::BarabasiAlbert,
        population: 20,
        rounds: 6,
        top_k: 3,
        seed_posters: 3,
        tol: 1e-12, // 収束で早期停止させない
        seed: Some(42),
        output_dir: output_dir.clone(),
        ..Config::default()
    };

    // 受信メッセージ数に応じて感情を強め，肯定的に投稿する擬似挙動．
    // メモリ/受信に語が多いほど intense へ寄せ，伝播曲線に変化を出す．
    let backend = ScriptedClient::new("mock-llama3.2", |prompt: &str| {
        // 受信メッセージが多い (= "posted"/"reposted" を多く含む) ほど感情を強める．
        let received = prompt.matches("user ").count();
        let emotion = if received >= 3 {
            "intense"
        } else if received >= 1 {
            "moderate"
        } else {
            "calm"
        };
        // 受信があれば肯定的態度・投稿，無ければ非活動気味に．
        let (attitude, behavior, content) = if received >= 1 {
            ("positive", "post", "I agree, this is important to share.")
        } else {
            ("negative", "inactive", "")
        };
        format!(
            "EMOTION: {emotion}\nATTITUDE: {attitude}\nBEHAVIOR: {behavior}\nCONTENT: {content}"
        )
    });
    let client = wrap_client(backend, PromptCache::in_memory());

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, client).expect("mock run failed");
    save_metrics(&result.metrics_history, &cfg.output_dir);
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    let f = fs::File::create(&cfg_path).unwrap();
    serde_json::to_writer_pretty(f, &cfg.to_run_config_json()).unwrap();

    // latest symlink
    let link = format!("{base}/latest");
    let _ = fs::remove_file(&link);
    #[cfg(unix)]
    let _ = std::os::unix::fs::symlink(&timestamp, &link);

    let last = result.metrics_history.last().unwrap();
    println!("mock smoke wrote: {output_dir}");
    println!(
        "final positive_frac={:.3} emotion(calm/mod/int)={:.2}/{:.2}/{:.2} adoption={:.3} cascade={} rounds={}",
        last.attitude_positive_frac,
        last.emotion_calm,
        last.emotion_moderate,
        last.emotion_intense,
        last.behavior_adoption_rate,
        last.info_cascade_size,
        result.final_round
    );
}
