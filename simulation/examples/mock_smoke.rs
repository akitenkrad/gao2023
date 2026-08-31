//! Mock 駆動のスモーク実行 (ライブ LLM 不要)．
//!
//! ライブ Ollama/OpenAI が使えない環境 (CI・ネットワーク遮断サンドボックス) で
//! 出力パイプライン (runvault の run ディレクトリ) と Python 可視化を検証する
//! ための補助バイナリ．`socsim-llm::mock::ScriptedClient` で決定論的に感情/態度/
//! 行動更新を駆動し，本番 `run` と同じ経路で結果を記録する．
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};

use s3_simulation::config::Config;
use s3_simulation::llm::wrap_client;
use s3_simulation::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use s3_simulation::simulation::run_with_client;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::{LlmClient, PromptCache};

/// シードは固定 (スモークの目的は «同じ入力で同じ出力» の確認)．
const SEED: u64 = 42;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());

    let cfg = Config {
        network: s3_simulation::config::NetworkKind::BarabasiAlbert,
        population: 20,
        rounds: 6,
        top_k: 3,
        seed_posters: 3,
        tol: 1e-12, // 収束で早期停止させない
        seed: Some(SEED),
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
    let llm = record::llm_block(
        client.inner().model(),
        client.inner().endpoint(),
        cfg.llm.temperature,
    );

    let parameters = cfg.to_run_config_json();
    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "run")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&base)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(SEED)
            .llm(llm)
            .replication(record::replication()),
    )
    .expect("runvault: run の開始に失敗");

    let result = run_with_client(&cfg, client).expect("mock run failed");
    record::log_simulation(&mut rv, &result);

    let last = result.metrics_history.last().unwrap();
    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
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
