//! 社会的伝染 LLM 応答のパース (感情・態度・行動・コンテンツ)．
//!
//! [`crate::prompts::contagion_prompt`] が要求する厳密フォーマット
//!
//! ```text
//! EMOTION: <calm|moderate|intense>
//! ATTITUDE: <negative|positive>
//! BEHAVIOR: <repost|post|inactive>
//! CONTENT: <...>
//! ```
//!
//! を規則ベースで読み取る．自由文に崩れても，行内のキーワードを走査して頑健に
//! 抽出し，読めない量は **現状態を維持** する (呼び出し側で fallback)．LLM 追加
//! 呼び出しは不要 (キャッシュ消費ゼロ)．

use crate::world::{Attitude, Behavior, Emotion};

/// パース済みの社会的伝染決定．読めなかった量は `None` (= 現状態維持)．
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ContagionDecision {
    /// 更新後の感情 (読めなければ `None`)．
    pub emotion: Option<Emotion>,
    /// 更新後の態度 (読めなければ `None`)．
    pub attitude: Option<Attitude>,
    /// 更新後の行動 (読めなければ `None`)．
    pub behavior: Option<Behavior>,
    /// 投稿/転送コンテンツ (空なら `None`)．
    pub content: Option<String>,
}

/// `KEY: value` 形式の行から `key` (大文字小文字無視) の値を取り出す．
fn field_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    for line in text.lines() {
        let line = line.trim();
        if let Some((lhs, rhs)) = line.split_once(':') {
            if lhs.trim().eq_ignore_ascii_case(key) {
                return Some(rhs.trim());
            }
        }
    }
    None
}

/// 行全体から最初に出現する既知のキーワードを探す (フォーマット崩れ時の保険)．
fn scan_keyword<T>(text: &str, parse: impl Fn(&str) -> Option<T>) -> Option<T> {
    for tok in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        if let Some(v) = parse(tok) {
            return Some(v);
        }
    }
    None
}

/// 社会的伝染 LLM 応答をパースする．
pub fn parse_contagion(text: &str) -> ContagionDecision {
    let emotion = field_value(text, "EMOTION")
        .and_then(Emotion::parse)
        .or_else(|| scan_keyword(text, Emotion::parse));
    let attitude = field_value(text, "ATTITUDE")
        .and_then(Attitude::parse)
        .or_else(|| scan_keyword(text, Attitude::parse));
    let behavior = field_value(text, "BEHAVIOR")
        .and_then(Behavior::parse)
        .or_else(|| scan_keyword(text, Behavior::parse));

    let content = field_value(text, "CONTENT")
        .map(|s| s.trim().trim_matches('"').trim().to_string())
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("empty"))
        .map(|s| s.replace(['\n', '\r'], " "));

    ContagionDecision {
        emotion,
        attitude,
        behavior,
        content,
    }
}

/// LLM 駆動 Perception 応答 (`crate::prompts::perception_prompt`) から **1 始まりの
/// 番号列** を抽出し，0 始まりの候補インデックスへ変換する．
///
/// 重複・範囲外 (`>= n_candidates`) は捨て，先頭から最大 `k` 件を順序保存で返す．
/// 番号が 1 つも読めなければ空 (呼び出し側で規則ベースへフォールバック)．
pub fn parse_perception_ranking(text: &str, n_candidates: usize, k: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for tok in text.split(|c: char| !c.is_ascii_digit()) {
        if tok.is_empty() {
            continue;
        }
        if let Ok(one_based) = tok.parse::<usize>() {
            if one_based == 0 || one_based > n_candidates {
                continue;
            }
            let idx = one_based - 1;
            if !out.contains(&idx) {
                out.push(idx);
                if out.len() >= k {
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_perception_ranking() {
        // 1 始まり番号を 0 始まりへ; 重複・範囲外は捨て，最大 k 件．
        assert_eq!(parse_perception_ranking("3, 1, 5", 5, 3), vec![2, 0, 4]);
        assert_eq!(parse_perception_ranking("2 2 9 4", 5, 3), vec![1, 3]);
        assert_eq!(
            parse_perception_ranking("no numbers here", 5, 3),
            Vec::<usize>::new()
        );
        assert_eq!(parse_perception_ranking("1,2,3,4", 5, 2), vec![0, 1]);
    }

    #[test]
    fn parses_strict_format() {
        let resp = "EMOTION: intense\nATTITUDE: positive\nBEHAVIOR: post\nCONTENT: I support this!";
        let d = parse_contagion(resp);
        assert_eq!(d.emotion, Some(Emotion::Intense));
        assert_eq!(d.attitude, Some(Attitude::Positive));
        assert_eq!(d.behavior, Some(Behavior::Post));
        assert_eq!(d.content.as_deref(), Some("I support this!"));
    }

    #[test]
    fn empty_content_is_none() {
        let resp = "EMOTION: calm\nATTITUDE: negative\nBEHAVIOR: inactive\nCONTENT: empty";
        let d = parse_contagion(resp);
        assert_eq!(d.behavior, Some(Behavior::Inactive));
        assert_eq!(d.content, None);
    }

    #[test]
    fn missing_fields_are_none() {
        let d = parse_contagion("I am not sure how I feel.");
        assert_eq!(d.emotion, None);
        assert_eq!(d.attitude, None);
        assert_eq!(d.behavior, None);
        assert_eq!(d.content, None);
    }

    #[test]
    fn scans_keywords_on_format_break() {
        let d = parse_contagion("My mood is calm and I am positive; I will repost this.");
        assert_eq!(d.emotion, Some(Emotion::Calm));
        assert_eq!(d.attitude, Some(Attitude::Positive));
        assert_eq!(d.behavior, Some(Behavior::Repost));
    }
}
