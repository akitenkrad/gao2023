//! LLM プロンプト生成 (社会的伝染 = 感情/態度/行動の同時更新 + コンテンツ生成)．
//!
//! 論文 §3 の個人レベルシミュレーション (感情・態度・行動の更新規則) に対応する
//! プロンプトを組み立てる．プロンプトはキャッシュキー (`hash(prompt + model)`) の
//! 素材になるため，同一状態からは同一プロンプト = 同一応答 (擬似決定論) になるよう
//! 決定論的に構築する．

use crate::world::{AgentState, Message};

/// メモリ/受信メッセージ (直近数件) をプロンプト用の短い文字列に畳む．
pub fn messages_digest(msgs: &[Message], max_items: usize) -> String {
    if msgs.is_empty() {
        return "(no messages received)".to_string();
    }
    let start = msgs.len().saturating_sub(max_items);
    msgs[start..]
        .iter()
        .map(|m| {
            let kind = if m.is_repost { "reposted" } else { "posted" };
            // 改行を畳んでプロンプトを安定化する．
            let body = m.content.replace(['\n', '\r'], " ");
            format!("- (user {} {kind}) \"{}\"", m.author.0, body)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 社会的伝染プロンプト: 受信メッセージ・属性・現状態を条件に，感情・態度・行動を
/// **同時に** 更新させる (1 エージェント = 1 LLM 呼び出し)．
///
/// 末尾で厳密な出力フォーマット (`EMOTION: ... / ATTITUDE: ... / BEHAVIOR: ... /
/// CONTENT: ...`) を要求し，[`crate::parse`] の規則ベースパースを成立させる．
pub fn contagion_prompt(agent: &AgentState, topic: &str, selected: &[Message]) -> String {
    let topic_h = topic.replace('_', " ");
    format!(
        "You are a social-media user. Profile: {profile}.\n\
         Topic under discussion: {topic_h}.\n\
         Your current emotion is {emotion}, your current attitude is {attitude}, \
         and your last action was {behavior}.\n\
         Recent messages you received from people you follow:\n{messages}\n\n\
         Based on your profile, your current state and the messages above, decide how you \
         react this round. Update your EMOTION (one of: calm, moderate, intense), your \
         ATTITUDE toward the topic (one of: negative, positive), and your BEHAVIOR \
         (one of: repost, post, inactive). If you choose post or repost, also write a \
         short message (max 30 words, no hashtags, no quotation marks). If inactive, leave \
         CONTENT empty.\n\n\
         Answer in EXACTLY this format and nothing else:\n\
         EMOTION: <calm|moderate|intense>\n\
         ATTITUDE: <negative|positive>\n\
         BEHAVIOR: <repost|post|inactive>\n\
         CONTENT: <your message or empty>",
        profile = agent.profile.describe(),
        topic_h = topic_h,
        emotion = agent.emotion.label(),
        attitude = agent.attitude.label(),
        behavior = agent.behavior.label(),
        messages = messages_digest(selected, 5),
    )
}

/// 初期態度の決定プロンプト (論文: 初期態度も LLM が決める)．
///
/// 末尾で «negative / positive の 1 語» の出力を促す．拡張点であり，既定の
/// 初期化は規則ベース ([`crate::simulation::init_world`]) を用いる．
pub fn initial_attitude_prompt(agent: &AgentState, topic: &str) -> String {
    let topic_h = topic.replace('_', " ");
    format!(
        "You are a social-media user. Profile: {profile}.\n\
         Topic: {topic_h}.\n\
         Before seeing any posts, what is your initial attitude toward this topic? \
         Answer with a SINGLE word: negative or positive.",
        profile = agent.profile.describe(),
        topic_h = topic_h,
    )
}
