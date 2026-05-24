//! 集団レベルの評価指標 (論文 §3 集団レベル伝播)．
//!
//! 各 round の世界状態から，態度割合・感情分布・行動採用率・情報カスケード規模を
//! 計算する．`MSED` / `Cor` のような実世界参照系列との整合 (論文 Table 4/5) は
//! Phase 3 (reproduce) に委ね，ここでは単一実行ごとの集団指標を扱う．

use std::collections::BTreeMap;

use serde::Serialize;
use socsim_core::AgentId;

use crate::world::{AgentState, Attitude, Emotion};

/// positive 態度を保持するエージェント割合 ∈ [0,1]．
pub fn attitude_positive_frac(agents: &BTreeMap<AgentId, AgentState>) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let pos = agents
        .values()
        .filter(|a| a.attitude == Attitude::Positive)
        .count();
    pos as f64 / agents.len() as f64
}

/// 感情分布 (calm / moderate / intense) の割合 ∈ [0,1]^3 (合計 1)．
pub fn emotion_dist(agents: &BTreeMap<AgentId, AgentState>) -> (f64, f64, f64) {
    let n = agents.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut calm = 0usize;
    let mut moderate = 0usize;
    let mut intense = 0usize;
    for a in agents.values() {
        match a.emotion {
            Emotion::Calm => calm += 1,
            Emotion::Moderate => moderate += 1,
            Emotion::Intense => intense += 1,
        }
    }
    let n = n as f64;
    (calm as f64 / n, moderate as f64 / n, intense as f64 / n)
}

/// 行動採用率: 当該 round に repost/post (= 情報発信) を行ったエージェント割合 ∈ [0,1]．
pub fn behavior_adoption_rate(agents: &BTreeMap<AgentId, AgentState>) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    let active = agents.values().filter(|a| a.behavior.is_active()).count();
    active as f64 / agents.len() as f64
}

/// 1 round 分の集団指標 (metrics.csv の 1 行)．
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    /// round 番号 t．
    pub t: usize,
    /// positive 態度割合 ∈ [0,1]．
    pub attitude_positive_frac: f64,
    /// calm 感情割合 ∈ [0,1]．
    pub emotion_calm: f64,
    /// moderate 感情割合 ∈ [0,1]．
    pub emotion_moderate: f64,
    /// intense 感情割合 ∈ [0,1]．
    pub emotion_intense: f64,
    /// 行動採用率 (repost/post 割合) ∈ [0,1]．
    pub behavior_adoption_rate: f64,
    /// 情報カスケード規模 (累積到達ノード数; 単調非減少)．
    pub info_cascade_size: usize,
}

impl Metrics {
    /// 世界状態とカスケード規模から集団指標を計算する．
    pub fn compute(
        agents: &BTreeMap<AgentId, AgentState>,
        info_cascade_size: usize,
        t: usize,
    ) -> Self {
        let (calm, moderate, intense) = emotion_dist(agents);
        Metrics {
            t,
            attitude_positive_frac: attitude_positive_frac(agents),
            emotion_calm: calm,
            emotion_moderate: moderate,
            emotion_intense: intense,
            behavior_adoption_rate: behavior_adoption_rate(agents),
            info_cascade_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{AgentState, Behavior, Profile};

    fn agent(emotion: Emotion, attitude: Attitude, behavior: Behavior) -> AgentState {
        let mut a = AgentState::new(
            Profile {
                gender: "person".into(),
                age: 30,
                occupation: "worker".into(),
            },
            emotion,
            attitude,
        );
        a.behavior = behavior;
        a
    }

    fn world(states: Vec<AgentState>) -> BTreeMap<AgentId, AgentState> {
        states
            .into_iter()
            .enumerate()
            .map(|(i, s)| (AgentId(i as u64), s))
            .collect()
    }

    #[test]
    fn fractions_in_unit_interval() {
        let w = world(vec![
            agent(Emotion::Calm, Attitude::Positive, Behavior::Post),
            agent(Emotion::Intense, Attitude::Negative, Behavior::Inactive),
            agent(Emotion::Moderate, Attitude::Positive, Behavior::Repost),
            agent(Emotion::Calm, Attitude::Negative, Behavior::Inactive),
        ]);
        assert!((attitude_positive_frac(&w) - 0.5).abs() < 1e-12);
        let (c, m, i) = emotion_dist(&w);
        assert!((c + m + i - 1.0).abs() < 1e-12);
        assert!((c - 0.5).abs() < 1e-12);
        assert!((behavior_adoption_rate(&w) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empty_world_is_zero() {
        let w: BTreeMap<AgentId, AgentState> = BTreeMap::new();
        assert_eq!(attitude_positive_frac(&w), 0.0);
        assert_eq!(emotion_dist(&w), (0.0, 0.0, 0.0));
        assert_eq!(behavior_adoption_rate(&w), 0.0);
    }
}
