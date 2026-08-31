#!/usr/bin/env python3
"""
reproduce_paper.py — Gao et al. (2023) S3 一括再現の集計・可視化スクリプト

Rust の `s3 reproduce` が書いた run ディレクトリを読み，

  1. events.jsonl の帯照合 (`x.gao2023.check`) を PASS / off で表示する
     (態度上昇・カスケード成長・行動採用・感情分布 MSED・態度時系列相関)，
  2. metrics.csv の S³ 系列と `baseline_<model>_*` 系列 (LT/IC/Voter/DeGroot) を
     重ねて，LLM 駆動 S³ と古典拡散ベースラインの伝播曲線を比較する図を生成する．

Rust 側で帯照合は確定済みなので，本ツールは図の生成と要約の再表示に専念する
(再計算しない)．

--results_dir を省略すると
`runvault path --experiment s3 --latest --subcommand reproduce`
が返す run ディレクトリを対象にする (`runvault` が PATH にある必要がある)．

Usage:
    uv run s3-tools reproduce
    uv run s3-tools reproduce --results_dir "$(runvault path --experiment s3 --latest --subcommand reproduce)"
    uv run s3-tools reproduce --output_dir out

Outputs:
    output_dir/
    ├── reproduce_propagation.png  ← S³ の集団レベル伝播 (態度・感情・行動・カスケード)
    └── reproduce_comparison.png   ← S³ vs LT/IC/Voter/DeGroot の到達/active 割合比較
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from runvault.read import (
    config_parameters,
    events_table,
    figures_dir,
    load_run_meta,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

# --------------------------------------------------------------------------- #
# runvault 側の名前 (Rust 側 record.rs と揃える)
# --------------------------------------------------------------------------- #
EXPERIMENT = "s3"
CHECK_EVENT = "x.gao2023.check"

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

# S³ 本体の round 別系列 (`baseline_*` はベースライン側)．
S3_SERIES = [
    "attitude_positive_frac",
    "emotion_calm",
    "emotion_moderate",
    "emotion_intense",
    "behavior_adoption_rate",
    "info_cascade_size",
]
# ベースライン 1 本の round 別系列．
BASELINE_SERIES = ["active_frac", "mean_opinion", "cumulative_reached"]


def _series(wide: pd.DataFrame, columns: dict[str, str]) -> pd.DataFrame | None:
    """long → wide に開いた表から系列だけを切り出す．

    S³ とベースラインは 1 つの run に同居し，停止するラウンドが違う．pivot は
    足りない側を NaN で埋めるので，切り出した後に落とす — 描かない点と «値が 0» を
    取り違えないため．
    """
    if not set(columns) <= set(wide.columns):
        return None
    df = (
        wide[["step", *columns]]
        .rename(columns={"step": "t", **columns})
        .dropna()
        .reset_index(drop=True)
    )
    return df if not df.empty else None


def s3_series(wide: pd.DataFrame) -> pd.DataFrame | None:
    """S³ 本体の round 別系列．"""
    return _series(wide, {name: name for name in S3_SERIES})


def baseline_series(wide: pd.DataFrame, model: str) -> pd.DataFrame | None:
    """ベースライン 1 本の round 別系列 (接頭辞を外した列名にして返す)．"""
    return _series(wide, {f"baseline_{model}_{name}": name for name in BASELINE_SERIES})


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


def print_summary(
    params: dict,
    meta: dict,
    scoped: dict[str, float],
    checks: pd.DataFrame,
    s3: pd.DataFrame | None,
    baselines: dict[str, pd.DataFrame],
) -> None:
    print("=" * 70)
    print("Gao et al. (2023) S3 — 一括再現 (observed-vs-paper)")
    print("=" * 70)
    setup = " ".join(
        f"{key}={params.get(key)}"
        for key in (
            "network",
            "population",
            "rounds",
            "top_k",
            "seed_posters",
            "seed",
            "llm_perception",
        )
    )
    print(f"setup: {setup}")
    model = (meta.get("llm") or {}).get("model_snapshot", "-")
    print(
        f"LLM: model={model} mock={params.get('mock')} "
        f"cache-hit={scoped.get('llm_cache_hit_rate', 0.0) * 100:.1f}%\n"
    )
    print("[1] headline 伝播チェック:")
    # `pass` は予約語なので itertuples では列名が潰れる．行は dict で取る．
    for _, c in checks.iterrows():
        verdict = "PASS" if c["pass"] else "off"
        print(
            f"  {c['indicator']:<24} = {c['observed']:>8.3f}   "
            f"({c['direction']} {c['paper']:.2f}: {verdict})"
        )
    passed = int(scoped.get("checks_passed", 0))
    total = int(scoped.get("checks_total", len(checks)))
    print(
        f"  → {passed}/{total} PASS "
        f"({'all PASS' if passed == total else 'review'})\n"
    )
    print("[2] 古典ベースライン比較 (同一網・同一シード):")
    if s3 is not None:
        n = max(int(params.get("population", 1)), 1)
        last = s3.iloc[-1]
        print(
            f"  S3 (LLM)  最終 active={last['attitude_positive_frac']:.3f} | "
            f"到達割合={last['info_cascade_size'] / n:.3f}"
        )
    for model_label in BASELINE_ORDER:
        df = baselines.get(model_label)
        if df is None:
            continue
        last = df.iloc[-1]
        final_round = int(scoped.get(f"baseline_{model_label}_final_round", last["t"]))
        print(
            f"  {model_label:<8}  最終 active={last['active_frac']:.3f} | "
            f"平均意見={last['mean_opinion']:.3f} | "
            f"到達={int(last['cumulative_reached'])} | round={final_round}"
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
        default=None,
        help="`s3 reproduce` の run ディレクトリ (省略時は runvault path --latest --subcommand reproduce)",
    )
    p.add_argument(
        "--results_root",
        "--results-root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    p.add_argument(
        "--experiment",
        default=EXPERIMENT,
        help=f"runvault の experiment 名 (default: {EXPERIMENT})",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先 (default: <experiment>/figures/<run_slug>/)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = runvault_path(
            args.experiment, args.results_root, subcommand="reproduce"
        )

    try:
        checks = events_table(results_dir, kind=CHECK_EVENT)
        params = config_parameters(results_dir)
        meta = load_run_meta(results_dir)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        print(
            "  まず `cargo run --release -- reproduce --mock` を実行してください．",
            file=sys.stderr,
        )
        return 1

    scoped = run_scope_metrics(results_dir)
    wide = metrics_wide(os.path.join(results_dir, "metrics.csv"))
    s3 = s3_series(wide)
    baselines = {m: baseline_series(wide, m) for m in BASELINE_ORDER}
    baselines = {m: df for m, df in baselines.items() if df is not None}

    print_summary(params, meta, scoped, checks, s3, baselines)

    if s3 is None:
        print(
            f"warning: S³ の round 別系列が無いため図を生成しません ({results_dir})",
            file=sys.stderr,
        )
        return 0

    # 図は run が終わった後に作るものなので run ディレクトリの外に置く．
    out_dir = args.output_dir or figures_dir(results_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("\n図を生成中 ...")
    save_propagation(s3, os.path.join(out_dir, "reproduce_propagation.png"))
    if baselines:
        population = max(int(params.get("population", 1)), 1)
        save_comparison(
            s3, baselines, population, os.path.join(out_dir, "reproduce_comparison.png")
        )
    else:
        print("  (ベースラインの系列が無いため比較図はスキップ)")

    print("-" * 70)
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
