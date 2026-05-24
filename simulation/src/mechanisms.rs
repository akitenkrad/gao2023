//! socsim フレームワーク上の S³ メカニズム (感情・態度・行動の LLM 駆動伝播)．
//!
//! 二層アーキテクチャの **境界** がここにある．下層 (決定論的 socsim コア) は
//! メッセージ配送・活性化順序を `ctx.rng` (ChaCha20) と有向グラフ構造で行い，上層
//! (非決定的 LLM レイヤ) は [`S3Client`] (キャッシュ付き Ollama→OpenAI フォール
//! バック) 越しの社会的伝染決定 (感情/態度/行動 + コンテンツ) を行う．
//!
//! # 同期更新 (synchronous rounds)
//!
//! 1 engine tick = 1 シミュレーション round = 全エージェント 1 回更新．round 内で
//! 生成された投稿/転送は `world.outbox` に積まれ，**次** round の先頭 (PreStep) で
//! inbox へ配送される．mid-round の状態変化が同 round の他者更新へ波及しない．
//!
//! # Mechanism × Phase
//!
//! | Mechanism | Phase | 役割 |
//! |-----------|-------|------|
//! | [`NetworkInitMechanism`]      | PreStep     | inbox クリア + 前 round の投稿/転送を著者のフォロワへ配送 |
//! | [`LLMPerceptionMechanism`]    | Decision    | inbox/memory から影響スコア上位 K を選択 (既定 規則ベース) |
//! | [`SocialContagionMechanism`]  | Interaction | **中核 LLM 呼び出し**: 感情/態度/行動を同時更新 + 投稿生成 |
//! | [`PopulationMetricsMechanism`]| PostStep    | 集団指標集計 + 収束/最大 round で request_stop |
//!
//! LLM クライアントと呼び出しメタデータは `Rc<RefCell<…>>` で共有し，run ドライバ
//! が実行後にキャッシュ保存・メタデータ集計に使う (engine はメカニズムを所有する
//! ため，共有参照で取り出す)．

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::config::LlmSettings;
use crate::llm::{llm_config, S3Client};
use crate::metrics::attitude_positive_frac;
use crate::parse::parse_contagion;
use crate::perception::select_top_k;
use crate::prompts::contagion_prompt;
use crate::world::{Message, S3World};

/// 共有 LLM クライアント (run ドライバとメカニズムで共有)．
pub type SharedClient = Rc<RefCell<S3Client>>;
/// 共有メタデータコレクタ (cache-hit 率などを run 後に集計)．
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

/// scratch キー: 選択された重要メッセージ (Perception → Contagion 受け渡し)．
const SCRATCH_SELECTED: &str = "selected_messages";
/// scratch キー: positive 態度割合 (収束判定 / ドライバ用)．
const SCRATCH_POS_FRAC: &str = "attitude_positive_frac";
/// scratch キー: 収束フラグ．
const SCRATCH_CONVERGED: &str = "converged";

// --------------------------------------------------------------------------- //
// NetworkInitMechanism (PreStep)
// --------------------------------------------------------------------------- //

/// round 先頭でのメッセージ配送 (`PreStep`)．
///
/// inbox をクリアし，前 round に積まれた `outbox` の各メッセージを **著者の
/// フォロワ** (有向辺 `follower → author`; = `in_neighbors(author)`) の inbox へ
/// 配送する．配送したノードを `world.reached` に積み，情報カスケード規模を更新する．
pub struct NetworkInitMechanism;

impl Mechanism<S3World> for NetworkInitMechanism {
    fn name(&self) -> &str {
        "network_init"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PreStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, S3World>) -> Result<()> {
        // inbox を空にする (各 round はその round に届いたものだけを見る)．
        for v in ctx.world.inbox.values_mut() {
            v.clear();
        }

        // 前 round の outbox を著者のフォロワへ配送する．
        let outbox = std::mem::take(&mut ctx.world.outbox);
        for msg in outbox {
            let followers = ctx.world.followers(msg.author);
            for f in followers {
                if let Some(inbox) = ctx.world.inbox.get_mut(&f) {
                    inbox.push(msg.clone());
                    ctx.world.reached.insert(f);
                }
            }
        }
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// LLMPerceptionMechanism (Decision)
// --------------------------------------------------------------------------- //

/// 重要メッセージ選択 (`Decision`)．
///
/// 各エージェントが inbox/memory から影響スコア (時間減衰 + 関連性 + 真正性) 上位
/// K 件を選び，scratch (`SCRATCH_SELECTED`) に格納する．既定は **規則ベース**
/// ([`crate::perception::select_top_k`])．`llm_perception` フラグは LLM 駆動選択の
/// 拡張スタブ (現状は規則ベースにフォールバック)．
pub struct LLMPerceptionMechanism {
    top_k: usize,
    /// LLM 駆動 Perception を使うか (拡張スタブ; 現状は規則ベース)．
    llm_perception: bool,
}

impl LLMPerceptionMechanism {
    /// 選択件数 K と LLM-perception フラグから作る．
    pub fn new(top_k: usize, llm_perception: bool) -> Self {
        LLMPerceptionMechanism {
            top_k: top_k.max(1),
            llm_perception,
        }
    }
}

impl Mechanism<S3World> for LLMPerceptionMechanism {
    fn name(&self) -> &str {
        "llm_perception"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Decision]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, S3World>) -> Result<()> {
        // LLM 駆動 Perception の拡張点: 現状は規則ベースに委ねる (将来 LLM へ差替)．
        let _ = self.llm_perception;
        let now = ctx.clock.t();
        let mut selected: BTreeMap<AgentId, Vec<Message>> = BTreeMap::new();
        for &id in ctx.world.agents.keys() {
            let agent = &ctx.world.agents[&id];
            let inbox = ctx
                .world
                .inbox
                .get(&id)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            let top = select_top_k(id, agent, inbox, now, self.top_k);
            selected.insert(id, top);
        }
        ctx.scratch.insert(SCRATCH_SELECTED, selected);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// SocialContagionMechanism (Interaction)
// --------------------------------------------------------------------------- //

/// 社会的伝染 (`Interaction`) — **本モデルの中核 LLM 呼び出し**．
///
/// `ctx.agent_order` の各エージェントについて，選択済み重要メッセージ・属性・現状態
/// を条件に LLM を **1 回** 呼び，感情・態度・行動を同時更新する．投稿/転送
/// (post/repost) 時はコンテンツを生成し，`world.outbox` に積む (= 次 round の配送
/// 対象)．repost で content が空のときは選択メッセージ先頭を転送する．
pub struct SocialContagionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    topic: String,
}

impl SocialContagionMechanism {
    /// 共有クライアント・メタデータ・LLM 設定・トピックから作る．
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        topic: String,
    ) -> Self {
        SocialContagionMechanism {
            client,
            metadata,
            settings,
            topic,
        }
    }
}

impl Mechanism<S3World> for SocialContagionMechanism {
    fn name(&self) -> &str {
        "social_contagion"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, S3World>) -> Result<()> {
        let now = ctx.clock.t();
        // Perception が選んだ重要メッセージを取り出す (なければ空)．
        let selected: BTreeMap<AgentId, Vec<Message>> = ctx
            .scratch
            .get::<BTreeMap<AgentId, Vec<Message>>>(SCRATCH_SELECTED)
            .cloned()
            .unwrap_or_default();

        // 活性化順序 (RandomActivationScheduler) で各エージェントを更新する．
        // 同期更新: 生成された投稿は outbox に積み，このラウンドの他者には見せない．
        let order: Vec<AgentId> = ctx.agent_order.to_vec();
        let mut new_outbox: Vec<Message> = Vec::new();

        for id in order {
            let sel = selected.get(&id).cloned().unwrap_or_default();
            let agent = match ctx.world.agents.get(&id) {
                Some(a) => a.clone(),
                None => continue,
            };

            // --- 中核 LLM 呼び出し (感情/態度/行動 + コンテンツの同時決定) ---
            let prompt = contagion_prompt(&agent, &self.topic, &sel);
            let text = {
                let mut client = self.client.borrow_mut();
                let resp = client
                    .complete(&prompt, &llm_config(&self.settings))
                    .map_err(|e| {
                        SocsimError::Mechanism(format!("contagion LLM call failed: {e}"))
                    })?;
                self.metadata.borrow_mut().record(resp.metadata.clone());
                resp.text
            };
            let decision = parse_contagion(&text);

            // --- world 更新 (読めなかった量は現状態維持) ---
            if let Some(state) = ctx.world.agents.get_mut(&id) {
                if let Some(e) = decision.emotion {
                    state.emotion = e;
                }
                if let Some(a) = decision.attitude {
                    state.attitude = a;
                }
                let behavior = decision.behavior.unwrap_or(state.behavior);
                state.behavior = behavior;
                // 選択した重要メッセージをメモリプールへ蓄積する．
                for m in &sel {
                    state.memory.push(m.clone());
                }

                // --- 投稿/転送なら outbox へ (次 round の配送対象) ---
                if behavior.is_active() {
                    let is_repost = behavior == crate::world::Behavior::Repost;
                    if is_repost {
                        // 転送: 選択メッセージ先頭を origin を継承して転送する．
                        if let Some(src) = sel.first() {
                            new_outbox.push(Message {
                                author: id,
                                origin: src.origin,
                                content: decision
                                    .content
                                    .clone()
                                    .unwrap_or_else(|| src.content.clone()),
                                round: now,
                                is_repost: true,
                            });
                        } else if let Some(content) = decision.content.clone() {
                            // 転送元が無いが repost を選んだ場合は新規投稿扱い．
                            new_outbox.push(Message {
                                author: id,
                                origin: id,
                                content,
                                round: now,
                                is_repost: false,
                            });
                        }
                    } else if let Some(content) = decision.content.clone() {
                        // 新規投稿．
                        new_outbox.push(Message {
                            author: id,
                            origin: id,
                            content,
                            round: now,
                            is_repost: false,
                        });
                        // 自分の投稿は自分も「到達済み」とみなす (cascade の起点)．
                        ctx.world.reached.insert(id);
                    }
                }
            }
        }

        // 次 round 用に積む (PreStep で配送される)．
        ctx.world.outbox = new_outbox;
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// PopulationMetricsMechanism (PostStep)
// --------------------------------------------------------------------------- //

/// 集団指標集計 + 収束判定 (`PostStep`)．
///
/// positive 態度割合を scratch に積み，前 round からの変化が `tol` 未満なら
/// `request_stop` で収束停止を要求する．集団指標自体の CSV 化は run ドライバが
/// 行う (engine の `run_observed` でスナップショットを観測する)．
pub struct PopulationMetricsMechanism {
    /// 収束判定しきい値 (positive 態度割合の round 間変化)．
    pub tol: f64,
    /// 前 round の positive 態度割合 (収束判定用)．
    prev_pos_frac: RefCell<Option<f64>>,
}

impl PopulationMetricsMechanism {
    /// 収束しきい値から作る．
    pub fn new(tol: f64) -> Self {
        PopulationMetricsMechanism {
            tol,
            prev_pos_frac: RefCell::new(None),
        }
    }
}

impl Mechanism<S3World> for PopulationMetricsMechanism {
    fn name(&self) -> &str {
        "population_metrics"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, S3World>) -> Result<()> {
        let pos_frac = attitude_positive_frac(&ctx.world.agents);
        ctx.scratch.insert(SCRATCH_POS_FRAC, pos_frac);

        let converged = match *self.prev_pos_frac.borrow() {
            Some(prev) => (pos_frac - prev).abs() < self.tol,
            None => false,
        };
        *self.prev_pos_frac.borrow_mut() = Some(pos_frac);

        ctx.scratch.insert(SCRATCH_CONVERGED, converged);
        if converged {
            ctx.request_stop();
        }
        Ok(())
    }
}
