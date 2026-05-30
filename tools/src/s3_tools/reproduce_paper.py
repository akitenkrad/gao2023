#!/usr/bin/env python3
"""
reproduce_paper.py — Gao et al. (2023) S3 一括再現の集計・可視化スクリプト

Rust の `s3 reproduce` が書き出した再現ディレクトリ (reproduce_<ts>/) を読み，

  1. reproduce_summary.json の observed-vs-paper チェックを PASS / off で表示する
     (態度上昇・カスケード成長・行動採用・感情分布 MSED・態度時系列相関)，
  2. s3_metrics.csv (S³) と baseline_<model>.csv (LT/IC/Voter/DeGroot) を重ねて，
     LLM 駆動 S³ と古典拡散ベースラインの伝播曲線を比較する図を生成する．

Rust 側で帯照合・JSON は確定済みなので，本ツールは図の生成と要約の再表示に専念する
(再計算しない)．

Usage:
    uv run s3-tools reproduce
    uv run s3-tools reproduce --results_dir results/reproduce_20260530_120000
    uv run s3-tools reproduce --output_dir out

Outputs:
    output_dir/
    ├── reproduce_propagation.png  ← S³ の集団レベル伝播 (態度・感情・行動・カスケード)
    └── reproduce_comparison.png   ← S³ vs LT/IC/Voter/DeGroot の到達/active 割合比較
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_S3 = "#F44336"
EMOTION_COLORS = {"calm": "#90CAF9", "moderate": "#FFB74D", "intense": "#E53935"}
BASELINE_COLORS = {
    "lt": "#2196F3",
    "ic": "#9C27B0",
    "voter": "#4CAF50",
    "degroot": "#FF9800",
}
BASELINE_LABELS = {
    "lt": "Linear Threshold",
    "ic": "Independent Cascade",
    "voter": "Voter",
    "degroot": "DeGroot",
}
BASELINE_ORDER = ["lt", "ic", "voter", "degroot"]


def load_summary(results_dir: str) -> dict:
    path = os.path.join(results_dir, "reproduce_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"reproduce_summary.json が見つかりません: {path}\n"
            f"  まず `cargo run --release -- reproduce --mock` を実行してください．"
        )
    with open(path) as f:
        return json.load(f)


def _read_csv(results_dir: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(results_dir, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def save_propagation(s3: pd.DataFrame, out_path: str) -> None:
    """S³ の集団レベル伝播 (態度・感情・行動・カスケード) を 4 パネルで保存する．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=COLOR_BG)
    fig.suptitle("Gao et al. (2023) S3 — 一括再現: 集団レベル伝播", fontsize=14)
    t = s3["t"]

    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, s3["attitude_positive_frac"], color=COLOR_S3, lw=2, marker="o", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("positive 態度割合")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("態度伝播")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    ax.stackplot(
        t,
        s3["emotion_calm"],
        s3["emotion_moderate"],
        s3["emotion_intense"],
        labels=["calm", "moderate", "intense"],
        colors=[EMOTION_COLORS["calm"], EMOTION_COLORS["moderate"], EMOTION_COLORS["intense"]],
        alpha=0.9,
    )
    ax.set_xlabel("round t")
    ax.set_ylabel("感情分布")
    ax.set_ylim(0, 1)
    ax.set_title("感情伝播")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, s3["behavior_adoption_rate"], color="#2196F3", lw=2, marker="s", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("行動採用率")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("行動採用曲線")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, s3["info_cascade_size"], color="#4CAF50", lw=2, marker="^", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("累積到達ノード数")
    ax.set_title("情報カスケード規模")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_comparison(
    s3: pd.DataFrame, baselines: dict[str, pd.DataFrame], population: int, out_path: str
) -> None:
    """S³ vs 古典ベースラインの伝播比較 (active 割合・累積到達割合) を保存する．"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Gao et al. (2023) S3 — LLM 駆動伝播 vs 古典拡散ベースライン", fontsize=14
    )

    # (1) active/positive 割合の時系列．
    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(
        s3["t"], s3["attitude_positive_frac"], color=COLOR_S3, lw=2.5,
        marker="o", ms=3, label="S3 (LLM)",
    )
    for m in BASELINE_ORDER:
        df = baselines.get(m)
        if df is None:
            continue
        ax.plot(
            df["t"], df["active_frac"], color=BASELINE_COLORS[m], lw=1.8,
            ls="--", label=BASELINE_LABELS[m],
        )
    ax.set_xlabel("round t")
    ax.set_ylabel("active / positive 割合")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("active 割合の伝播")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) 累積到達割合の時系列．
    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    n = max(population, 1)
    ax.plot(
        s3["t"], s3["info_cascade_size"] / n, color=COLOR_S3, lw=2.5,
        marker="o", ms=3, label="S3 (LLM)",
    )
    for m in BASELINE_ORDER:
        df = baselines.get(m)
        if df is None:
            continue
        ax.plot(
            df["t"], df["cumulative_reached"] / n, color=BASELINE_COLORS[m],
            lw=1.8, ls="--", label=BASELINE_LABELS[m],
        )
    ax.set_xlabel("round t")
    ax.set_ylabel("累積到達割合")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("情報カスケードの到達割合")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def print_summary(summary: dict) -> None:
    print("=" * 70)
    print("Gao et al. (2023) S3 — 一括再現 (observed-vs-paper)")
    print("=" * 70)
    print(f"setup: {summary['setup']}")
    print(f"LLM: model={summary['llm_model']} mock={summary['mock']} "
          f"cache-hit={summary['cache_hit_rate'] * 100:.1f}%\n")
    print("[1] headline 伝播チェック:")
    for c in summary["checks"]:
        verdict = "PASS" if c["pass"] else "off"
        print(
            f"  {c['indicator']:<24} = {c['observed']:>8.3f}   "
            f"({c['direction']} {c['paper']:.2f}: {verdict})"
        )
    print(
        f"  → {summary['passed']}/{summary['total']} PASS "
        f"({'all PASS' if summary['all_pass'] else 'review'})\n"
    )
    print("[2] 古典ベースライン比較 (同一網・同一シード):")
    print(
        f"  S3 (LLM)  最終 active={summary['s3_final_active_frac']:.3f} | "
        f"到達割合={summary['s3_final_reached_frac']:.3f}"
    )
    for b in summary["baselines"]:
        print(
            f"  {b['model']:<8}  最終 active={b['final_active_frac']:.3f} | "
            f"平均意見={b['final_mean_opinion']:.3f} | 到達={b['final_reached']} | "
            f"round={b['final_round']}"
        )
    print("=" * 70)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s3-tools reproduce",
        description="Gao et al. (2023) S3 一括再現の集計・可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="`s3 reproduce` の出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先 (default: {results_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir
    out_dir = args.output_dir or os.path.join(results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    try:
        summary = load_summary(results_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print_summary(summary)

    s3 = _read_csv(results_dir, "s3_metrics.csv")
    if s3 is None:
        s3 = _read_csv(results_dir, "metrics.csv")
    if s3 is None:
        print(
            f"warning: s3_metrics.csv が無いため図を生成しません ({results_dir})",
            file=sys.stderr,
        )
        return 0

    baselines = {
        m: _read_csv(results_dir, f"baseline_{m}.csv") for m in BASELINE_ORDER
    }
    baselines = {m: df for m, df in baselines.items() if df is not None}

    # population は config.json から (図の正規化用)．
    population = 1
    cfg_path = os.path.join(results_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            population = int(json.load(f).get("population", 1))

    print("\n図を生成中 ...")
    save_propagation(s3, os.path.join(out_dir, "reproduce_propagation.png"))
    if baselines:
        save_comparison(
            s3, baselines, population, os.path.join(out_dir, "reproduce_comparison.png")
        )
    else:
        print("  (baseline_*.csv が無いため比較図はスキップ)")

    print("-" * 70)
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
