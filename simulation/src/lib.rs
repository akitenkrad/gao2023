//! Gao et al. (2023) "S3: Social-network Simulation System with Large Language
//! Model-Empowered Agents" の再現実装ライブラリ．
//!
//! socsim フレームワーク上に構築した，**有向** ソーシャルネットワーク (フォロー
//! 関係) 上の LLM 駆動の感情・態度・行動伝播の公開 API を提供する．設定
//! (`config`)・世界状態 (`world`)・LLM クライアント層 (`llm`)・プロンプト生成
//! (`prompts`)・応答パース (`parse`)・知覚スコア (`perception`)・更新メカニズム
//! (`mechanisms`)・実行ドライバ (`simulation`)・集計メトリクス (`metrics`) を
//! モジュールとして公開し，バイナリ (`s3`) と統合テストの双方から利用する．
//!
//! # 二層決定論
//!
//! socsim コア層 (有向網生成・メッセージ配送・活性化順序・メトリクス) は seed から
//! bit 単位で決定論的である．LLM レイヤ (感情/態度/行動の同時更新 + コンテンツ
//! 生成) は socsim の bit 再現性の **外側** にあり，`socsim-llm` のキャッシュ +
//! `temperature=0` + `seed` 固定で擬似決定論化する．詳細は `crate::llm` を参照．

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod parse;
pub mod perception;
pub mod prompts;
pub mod simulation;
pub mod world;
