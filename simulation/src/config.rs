//! シミュレーション設定．
//!
//! Gao et al. (2023) "S3" のコアモデル (有向ソーシャルネットワーク上の LLM 駆動
//! 感情・態度・行動伝播) と感度分析パラメータを保持する [`Config`] と，その JSON
//! シリアライズ表現を定義する．ネットワーク種別・LLM 設定などの列挙型もここに集約
//! する．

use serde::Serialize;

// --------------------------------------------------------------------------- //
// ネットワーク種別
// --------------------------------------------------------------------------- //

/// 合成ソーシャルネットワークの生成モデル．
///
/// 論文はフォロー関係に基づく **有向** グラフを用いる．`socsim-net` は issue #28 で
/// **有向生成器** を提供するようになったため，ER/BA は有向生成器を直接呼ぶ:
/// `erdos_renyi_directed` / `barabasi_albert_directed` ([`socsim_net::DiSocialNetwork`])．
/// WS には有向生成器が無いので，無向の `watts_strogatz` を生成してから
/// `to_directed(p_mutual, rng)` で方向を付与する (構築規約は
/// [`crate::simulation::init_world`] を参照)．
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkKind {
    /// Erdős–Rényi 有向 G(n,p) (`erdos_renyi_directed`; 各順序対ごとに独立な弧)．
    ErdosRenyi,
    /// Watts–Strogatz 小世界網を無向生成 → `to_directed(p_mutual)` で方向付与．
    WattsStrogatz,
    /// Barabási–Albert 有向版 (`barabasi_albert_directed`; in-degree 優先選択)．
    BarabasiAlbert,
}

impl NetworkKind {
    /// 短い識別ラベル (CLI / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            NetworkKind::ErdosRenyi => "er",
            NetworkKind::WattsStrogatz => "ws",
            NetworkKind::BarabasiAlbert => "ba",
        }
    }
}

/// 文字列から [`NetworkKind`] をパースする．
pub fn parse_network(s: &str) -> Result<NetworkKind, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "er" | "erdos_renyi" | "erdos-renyi" => Ok(NetworkKind::ErdosRenyi),
        "ws" | "watts_strogatz" | "watts-strogatz" => Ok(NetworkKind::WattsStrogatz),
        "ba" | "barabasi_albert" | "barabasi-albert" => Ok(NetworkKind::BarabasiAlbert),
        _ => Err(format!("不正なネットワーク種別: \"{}\" (er / ws / ba)", s)),
    }
}

// --------------------------------------------------------------------------- //
// LLM 設定
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// 単一実行の設定．
#[derive(Debug, Clone)]
pub struct Config {
    /// ネットワーク種別 (er / ws / ba)．
    pub network: NetworkKind,
    /// 人口規模 (= ノード数)．
    pub population: usize,
    /// ER の接続確率 p．
    pub er_p: f64,
    /// WS の各ノードの初期次数 k (偶数)．
    pub ws_k: usize,
    /// WS の再配線確率 β．
    pub ws_beta: f64,
    /// WS の無向→有向変換 (`to_directed`) で双方向 (相互フォロー) になる確率
    /// `p_mutual`．残りの辺は RNG で片方向に倒す (既定 0.5)．ER/BA は有向生成器を
    /// 直接使うため本値は影響しない．
    pub ws_p_mutual: f64,
    /// BA の新規ノードあたりの結合数 m．
    pub ba_m: usize,
    /// 伝播ラウンド数 (= engine tick 数)．
    pub rounds: usize,
    /// LLM Perception で選択する重要メッセージ件数 K．
    pub top_k: usize,
    /// LLM 駆動 Perception を使うか (既定 false = 規則ベース; 拡張スタブ)．
    pub llm_perception: bool,
    /// 初期に種をまく "発信源" エージェント数 (round 0 で必ず投稿する)．
    pub seed_posters: usize,
    /// 集団指標の収束しきい値 (positive 態度割合の round 間変化 < tol で停止)．
    pub tol: f64,
    /// 乱数シード (None の場合はランダム; socsim コア層のみ支配)．
    pub seed: Option<u64>,
    /// LLM レイヤ設定．
    pub llm: LlmSettings,
}

impl Default for Config {
    /// 標準設定 (BA, N=100, 20 round, top-k=3)．
    fn default() -> Self {
        Config {
            network: NetworkKind::BarabasiAlbert,
            population: 100,
            er_p: 0.05,
            ws_k: 4,
            ws_beta: 0.1,
            ws_p_mutual: 0.5,
            ba_m: 3,
            rounds: 20,
            top_k: 3,
            llm_perception: false,
            seed_posters: 3,
            tol: 1e-9,
            seed: Some(42),
            llm: LlmSettings::default(),
        }
    }
}

/// `config.json` の `parameters` に載る S³ の実験条件．
///
/// 実行の同一性 (どのサブコマンドか・どこへ出力したか) は runvault の `run.json` が
/// 持つので，ここには条件だけを置く (`command` / `output_dir` は持たない)．
#[derive(Serialize)]
pub struct RunConfigJson {
    pub network: String,
    pub population: usize,
    pub er_p: f64,
    pub ws_k: usize,
    pub ws_beta: f64,
    pub ws_p_mutual: f64,
    pub ba_m: usize,
    pub rounds: usize,
    pub top_k: usize,
    pub llm_perception: bool,
    pub seed_posters: usize,
    pub tol: f64,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
}

impl Config {
    /// `config.json` の `parameters` 用の表現を組み立てる．
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            network: self.network.label().to_string(),
            population: self.population,
            er_p: self.er_p,
            ws_k: self.ws_k,
            ws_beta: self.ws_beta,
            ws_p_mutual: self.ws_p_mutual,
            ba_m: self.ba_m,
            rounds: self.rounds,
            top_k: self.top_k,
            llm_perception: self.llm_perception,
            seed_posters: self.seed_posters,
            tol: self.tol,
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
        }
    }
}
