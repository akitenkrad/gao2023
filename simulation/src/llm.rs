//! LLM クライアント層 (Ollama 第一 → OpenAI フォールバック + キャッシュ)．
//!
//! 本モジュールは `socsim-llm` の合成 API に対する薄いビルダである．二層
//! アーキテクチャの **上層 (非決定的 LLM レイヤ)** をここに閉じ込め，下層の
//! 決定論的 socsim コアからは [`S3Client`] 型エイリアス経由でのみ触れる．
//!
//! # 合成 (Ollama 第一 → OpenAI フォールバック → キャッシュ)
//!
//! ```text
//! CachingClient< Box<dyn LlmClient> >
//!   └─ cache: PromptCache (prompt → response; 擬似決定論の本体)
//!      └─ Box<dyn LlmClient> = FallbackClient< OllamaClient, OpenAiClient >
//!         primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!         secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! 設計 §4.2/§7 の `reqwest` は本層 (socsim-llm) で置換される．`FallbackClient` は
//! socsim-llm が提供する (自前実装しない)．「Ollama を試行 → 任意のエラーで OpenAI
//! へフォールバック」を担う．`CachingClient` はその上にプロンプト→応答キャッシュを
//! 被せ，`temperature=0` / `seed` 固定と合わせて再実行を擬似決定論化する．
//!
//! テストでは `socsim-llm::mock::ScriptedClient` を `Box<dyn LlmClient>` として
//! 同じ [`S3Client`] に流し込める．`socsim-llm` が `Box<dyn LlmClient>` に対する
//! [`LlmClient`] の転送実装を提供する (issue #26) ため，専用 newtype は不要．

use std::path::Path;

use socsim_llm::{CachingClient, LlmClient, LlmConfig, LlmError, PromptCache};

use crate::config::LlmSettings;

/// 本シミュレーションが用いるキャッシュ付きクライアント型．
///
/// バックエンドは `Box<dyn LlmClient>` に型消去してあり，本番は
/// `FallbackClient<OllamaClient, OpenAiClient>`，テストは `ScriptedClient` を
/// 注入できる．`socsim-llm` の `impl LlmClient for Box<T>` (issue #26) により
/// 専用 newtype なしで `CachingClient` の `C: LlmClient` 境界を満たす．
pub type S3Client = CachingClient<Box<dyn LlmClient>>;

/// 本番用の «Ollama 第一 → OpenAI フォールバック + キャッシュ» クライアントを
/// 環境変数から構築する．
///
/// - Ollama: `OLLAMA_HOST` (既定 `http://localhost:11434`) / `OLLAMA_MODEL`
///   (既定 `llama3.2:latest`)．
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (既定 `gpt-4o-mini`)．未設定なら
///   空キーのフォールバックを置く (Ollama が成功すれば呼ばれない; 両方失敗時のみ
///   設定エラーになる)．
/// - キャッシュ: `settings.cache_path` があればその JSON ファイル，なければ
///   in-memory．
pub fn build_live_client(settings: &LlmSettings) -> Result<S3Client, LlmError> {
    // «Ollama 第一 → OpenAI フォールバック → 型消去 → キャッシュ» の組み立ては
    // socsim-llm の `build_live_client` に委譲する (挙動は従来の手書き実装と等価)．
    // 本ラッパは replication 固有の `LlmSettings` (cache_path) を受け取る薄い層．
    socsim_llm::build_live_client(settings.cache_path.as_deref().map(Path::new))
}

/// 任意の [`LlmClient`] (例: `mock::ScriptedClient`) をキャッシュで包んだ
/// [`S3Client`] を作る (主にテスト用)．
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> S3Client {
    let boxed: Box<dyn LlmClient> = Box::new(backend);
    CachingClient::new(boxed, cache)
}

/// [`LlmSettings`] から socsim-llm の [`LlmConfig`] を組み立てる．
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
}

/// 既定の OLLAMA モデル名 (環境変数未設定時; 設計の規約)．
pub const DEFAULT_OLLAMA_MODEL: &str = "llama3.2:latest";
