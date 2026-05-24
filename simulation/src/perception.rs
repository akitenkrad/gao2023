//! メモリプールの影響因子スコア (論文 §3 個人レベル: 上位 K メッセージ選択)．
//!
//! 影響スコア
//!
//! ```text
//! influence(m) = w1 * τ(m) + w2 * rel(d, m) + w3 * auth(m)
//! ```
//!
//! - `τ(m)`  : 忘却関数による時間スコア (新しいほど高い; 指数減衰)．
//! - `rel`   : 属性とメッセージ内容の関連性 (本実装では語彙重なりによる簡易代理)．
//! - `auth`  : 情報源スコア (自己投稿 < 一方向 < 相互フォロー / repost vs post)．
//!
//! 既定は **規則ベース** (LLM 非依存)．LLM 駆動 Perception は拡張スタブ
//! ([`crate::config::Config::llm_perception`]) として残す．コサイン類似度は埋め込み
//! 器が必要なため，ここでは決定論的かつ軽量な語彙重なり (Jaccard) で代理する．

use std::collections::BTreeSet;

use crate::world::{AgentState, Message};

/// 時間スコアの重み．
pub const W_TIME: f64 = 0.5;
/// 関連性スコアの重み．
pub const W_REL: f64 = 0.3;
/// 真正性 (情報源) スコアの重み．
pub const W_AUTH: f64 = 0.2;

/// 時間減衰の半減期 (round)．
const TIME_HALF_LIFE: f64 = 3.0;

/// 忘却関数による時間スコア τ(m) ∈ (0,1]．現 round に近いほど 1 に近い．
pub fn time_score(msg_round: u64, now: u64) -> f64 {
    let age = now.saturating_sub(msg_round) as f64;
    0.5_f64.powf(age / TIME_HALF_LIFE)
}

/// テキストを小文字 ASCII 単語集合へ正規化する．
fn tokens(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_ascii_lowercase())
        .collect()
}

/// 属性とメッセージ内容の関連性 rel(d, m) ∈ [0,1] (Jaccard 係数による簡易代理)．
pub fn relevance(agent: &AgentState, msg: &Message) -> f64 {
    let profile_text = agent.profile.describe();
    let a = tokens(&profile_text);
    let b = tokens(&msg.content);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// 情報源スコア auth(m) ∈ [0,1]．
///
/// 自己投稿は低く，新規投稿 (post) は転送 (repost) よりやや高い情報量を持つと
/// みなす (論文の «一方向/相互/推薦/自己» 序列の簡易代理)．
pub fn authenticity(agent_self: socsim_core::AgentId, msg: &Message) -> f64 {
    if msg.author == agent_self {
        0.1
    } else if msg.is_repost {
        0.6
    } else {
        0.8
    }
}

/// メッセージ 1 件の影響スコア．
pub fn influence(
    agent_self: socsim_core::AgentId,
    agent: &AgentState,
    msg: &Message,
    now: u64,
) -> f64 {
    W_TIME * time_score(msg.round, now)
        + W_REL * relevance(agent, msg)
        + W_AUTH * authenticity(agent_self, msg)
}

/// inbox + memory から影響スコア上位 K 件を選ぶ (規則ベース Perception)．
///
/// 同点は (round 降順, author 昇順) で安定ソートし決定論を保つ．
pub fn select_top_k(
    agent_self: socsim_core::AgentId,
    agent: &AgentState,
    inbox: &[Message],
    now: u64,
    k: usize,
) -> Vec<Message> {
    // inbox (新着) と memory (過去に選んだもの) を併せて候補にする．
    let mut scored: Vec<(f64, &Message)> = inbox
        .iter()
        .chain(agent.memory.iter())
        .map(|m| (influence(agent_self, agent, m, now), m))
        .collect();
    // スコア降順 → round 降順 → author 昇順 で安定化．
    scored.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.round.cmp(&a.1.round))
            .then(a.1.author.0.cmp(&b.1.author.0))
    });
    scored.into_iter().take(k).map(|(_, m)| m.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{AgentState, Attitude, Emotion, Profile};
    use socsim_core::AgentId;

    fn msg(author: u64, content: &str, round: u64, is_repost: bool) -> Message {
        Message {
            author: AgentId(author),
            origin: AgentId(author),
            content: content.into(),
            round,
            is_repost,
        }
    }

    fn agent() -> AgentState {
        AgentState::new(
            Profile {
                gender: "person".into(),
                age: 30,
                occupation: "engineer".into(),
            },
            Emotion::Calm,
            Attitude::Negative,
        )
    }

    #[test]
    fn newer_messages_score_higher_in_time() {
        assert!(time_score(10, 10) > time_score(7, 10));
        assert!(time_score(10, 10) <= 1.0);
    }

    #[test]
    fn top_k_limits_and_orders() {
        let a = agent();
        let inbox = vec![
            msg(1, "old news", 1, false),
            msg(2, "fresh engineer topic", 5, false),
            msg(3, "another", 5, true),
        ];
        let sel = select_top_k(AgentId(0), &a, &inbox, 5, 2);
        assert_eq!(sel.len(), 2);
        // 新しく関連性も高い message 2 が先頭に来る．
        assert_eq!(sel[0].author, AgentId(2));
    }
}
