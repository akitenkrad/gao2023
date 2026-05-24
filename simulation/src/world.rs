//! socsim フレームワーク上の S³ (Gao et al. 2023) の世界状態．
//!
//! エージェントは **固定ノード** であり移動せず，内部状態 (感情・態度・行動・
//! メモリプール) だけが変化する．空間は **有向** ソーシャルネットワーク
//! [`socsim_net::DiSocialNetwork`] (フォロー関係) を採用する (issue #18 の有向対応;
//! 設計 §4.3 の UNCONFIRMED note を解消)．
//!
//! # フォロー方向の規約
//!
//! 有向辺 `A → B` は「**A が B をフォローする**」を表す (Twitter/X 流)．したがって
//! B の投稿は B の **フォロワ** に届く．B のフォロワとは `* → B` の辺を持つノード，
//! すなわち B の **in-neighbours** (`in_neighbors(B)`) である．配送はこの規約で行う
//! ([`crate::mechanisms::NetworkInitMechanism`])．
//!
//! `#[derive(Clone)]` でスナップショット (save/resume) と比較実験に対応する．
//! `agent_ids()` は `BTreeMap` のソート済みキーを返し決定論を保証する (socsim コア層)．

use std::collections::BTreeMap;

use socsim_core::{AgentId, SimClock, WorldState};
use socsim_net::DiSocialNetwork;

// --------------------------------------------------------------------------- //
// 伝播 3 量 (感情・態度・行動)
// --------------------------------------------------------------------------- //

/// 感情状態 (3 段階マルコフ): calm / moderate / intense．
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Emotion {
    /// 平静．
    Calm,
    /// 中程度．
    Moderate,
    /// 強い感情．
    Intense,
}

impl Emotion {
    /// 短いラベル (プロンプト / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            Emotion::Calm => "calm",
            Emotion::Moderate => "moderate",
            Emotion::Intense => "intense",
        }
    }

    /// 文字列 (LLM 応答トークン) から感情をパースする．未知語は `None`．
    pub fn parse(s: &str) -> Option<Emotion> {
        match s.trim().to_ascii_lowercase().as_str() {
            "calm" => Some(Emotion::Calm),
            "moderate" => Some(Emotion::Moderate),
            "intense" => Some(Emotion::Intense),
            _ => None,
        }
    }
}

/// 態度状態 (2 値マルコフ): negative / positive．
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Attitude {
    /// 否定的態度．
    Negative,
    /// 肯定的態度．
    Positive,
}

impl Attitude {
    /// 短いラベル (プロンプト / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            Attitude::Negative => "negative",
            Attitude::Positive => "positive",
        }
    }

    /// 文字列 (LLM 応答トークン) から態度をパースする．未知語は `None`．
    pub fn parse(s: &str) -> Option<Attitude> {
        match s.trim().to_ascii_lowercase().as_str() {
            "negative" | "neg" => Some(Attitude::Negative),
            "positive" | "pos" => Some(Attitude::Positive),
            _ => None,
        }
    }
}

/// 相互作用行動: 転送 / 新規投稿 / 非活動．
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Behavior {
    /// 受信メッセージを転送する (repost)．
    Repost,
    /// 新規にコンテンツを投稿する (post)．
    Post,
    /// 行動しない (inactive)．
    Inactive,
}

impl Behavior {
    /// 短いラベル (プロンプト / 出力用)．
    pub fn label(&self) -> &'static str {
        match self {
            Behavior::Repost => "repost",
            Behavior::Post => "post",
            Behavior::Inactive => "inactive",
        }
    }

    /// 文字列 (LLM 応答トークン) から行動をパースする．未知語は `None`．
    pub fn parse(s: &str) -> Option<Behavior> {
        match s.trim().to_ascii_lowercase().as_str() {
            "repost" | "share" | "retweet" => Some(Behavior::Repost),
            "post" | "tweet" => Some(Behavior::Post),
            "inactive" | "none" | "idle" => Some(Behavior::Inactive),
            _ => None,
        }
    }

    /// 投稿/転送なら `true` (= 情報を発信する行動)．
    pub fn is_active(&self) -> bool {
        matches!(self, Behavior::Repost | Behavior::Post)
    }
}

// --------------------------------------------------------------------------- //
// 人口統計属性
// --------------------------------------------------------------------------- //

/// 人口統計属性 (性別・年齢・職業)．init 時に固定される．
#[derive(Clone, Debug)]
pub struct Profile {
    /// 性別 (テキスト)．
    pub gender: String,
    /// 年齢．
    pub age: u8,
    /// 職業 (テキスト)．
    pub occupation: String,
}

impl Profile {
    /// プロンプト埋め込み用の短い記述を返す．
    pub fn describe(&self) -> String {
        format!(
            "a {age}-year-old {gender} {occupation}",
            age = self.age,
            gender = self.gender,
            occupation = self.occupation,
        )
    }
}

// --------------------------------------------------------------------------- //
// メッセージ
// --------------------------------------------------------------------------- //

/// ネットワークを流れる 1 メッセージ (投稿/転送)．
#[derive(Clone, Debug)]
pub struct Message {
    /// 発信元エージェント．
    pub author: AgentId,
    /// このメッセージの起点となった発信源 (info-cascade 追跡用; repost で継承)．
    pub origin: AgentId,
    /// 本文．
    pub content: String,
    /// 生成された round (時間減衰スコアの基準)．
    pub round: u64,
    /// 転送 (repost) か新規投稿 (post) か．
    pub is_repost: bool,
}

// --------------------------------------------------------------------------- //
// エージェント状態
// --------------------------------------------------------------------------- //

/// 1 エージェントの個別状態 (固定ノード; 移動しない)．
#[derive(Clone, Debug)]
pub struct AgentState {
    /// 人口統計属性 (init 時に固定)．
    pub profile: Profile,
    /// 感情 (3 段階マルコフ)．
    pub emotion: Emotion,
    /// 態度 (2 値マルコフ)．
    pub attitude: Attitude,
    /// 相互作用行動．
    pub behavior: Behavior,
    /// メモリプール (過去に受信/選択したメッセージ)．
    pub memory: Vec<Message>,
}

impl AgentState {
    /// 初期属性・初期 3 量からエージェント状態を作る．
    pub fn new(profile: Profile, emotion: Emotion, attitude: Attitude) -> Self {
        AgentState {
            profile,
            emotion,
            attitude,
            behavior: Behavior::Inactive,
            memory: Vec::new(),
        }
    }
}

// --------------------------------------------------------------------------- //
// 世界状態
// --------------------------------------------------------------------------- //

/// S³ の世界状態．
#[derive(Clone)]
pub struct S3World {
    /// シミュレーションクロック．
    pub clock: SimClock,
    /// 有向ソーシャルネットワーク (辺 `A → B` = 「A が B をフォロー」)．
    pub net: DiSocialNetwork,
    /// 各エージェントの状態 (ソート済みキー)．
    pub agents: BTreeMap<AgentId, AgentState>,
    /// 当該 round に届いたメッセージ (NetworkInit で配送; 同期更新の入口)．
    pub inbox: BTreeMap<AgentId, Vec<Message>>,
    /// **次** round に配送するメッセージ (SocialContagion で投稿/転送時に充填)．
    /// 同期更新: round 内で生成された投稿は次 round の inbox にのみ反映する．
    pub outbox: Vec<Message>,
    /// info-cascade 追跡: これまでに (累積で) いずれかのメッセージに到達した
    /// ノード集合 (origin に依らず「情報拡散の累積到達ノード数」を測る)．
    pub reached: std::collections::BTreeSet<AgentId>,
}

impl S3World {
    /// 構成済みフィールドから世界状態を組み立てる (網生成・初期化は
    /// [`crate::simulation::init_world`])．
    pub fn new(net: DiSocialNetwork, agents: BTreeMap<AgentId, AgentState>, rounds: u64) -> Self {
        let inbox = agents.keys().map(|&id| (id, Vec::new())).collect();
        S3World {
            clock: SimClock::new(rounds),
            net,
            agents,
            inbox,
            outbox: Vec::new(),
            reached: std::collections::BTreeSet::new(),
        }
    }

    /// 人口規模 N．
    pub fn n(&self) -> usize {
        self.agents.len()
    }

    /// B のフォロワ (= B の投稿が届く相手) を返す．
    ///
    /// フォロー規約 `A → B` = 「A が B をフォロー」より，B のフォロワは `* → B` の
    /// 辺を持つノード = B の in-neighbours．
    pub fn followers(&self, author: AgentId) -> Vec<AgentId> {
        self.net.in_neighbors(author)
    }
}

impl WorldState for S3World {
    fn agent_ids(&self) -> Vec<AgentId> {
        // BTreeMap のキーはソート済み．契約 (sorted) を明示する．
        self.agents.keys().copied().collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}
