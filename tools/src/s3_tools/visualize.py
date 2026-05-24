#!/usr/bin/env python3
"""
visualize.py — Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 可視化スクリプト

results/latest (または --results_dir 指定先) の metrics.csv を読み，集団レベル伝播の
時系列図を生成する:
(1) 態度割合の時系列 (positive 態度を保持するエージェント割合)
(2) 感情分布の時系列 (calm / moderate / intense の積み上げ)
(3) 行動採用曲線 (repost/post 割合)
(4) 情報カスケード規模 (累積到達ノード数)
config.json があれば，network/population に応じた合成有向グラフのスナップショットも
描く (networkx; --no-graph で抑止)．

Usage:
    uv run s3-tools visualize
    uv run s3-tools visualize --results_dir results/20260524_153000
    uv run s3-tools visualize --output_dir out --no-graph

Outputs:
    output_dir/
    ├── propagation_timeseries.png ← 態度割合・感情分布・行動採用・カスケード
    └── network_snapshot.png       ← 合成有向グラフのスナップショット (任意)
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 日本語フォント設定
# --------------------------------------------------------------------------- #
plt.rcParams["font.family"] = "Hiragino Sans"

# --------------------------------------------------------------------------- #
# カラー設定
# --------------------------------------------------------------------------- #
COLOR_BG = "#FAFAF8"
COLOR_ATT = "#F44336"
COLOR_ADOPT = "#2196F3"
COLOR_CASC = "#4CAF50"
EMOTION_COLORS = {
    "calm": "#90CAF9",
    "moderate": "#FFB74D",
    "intense": "#E53935",
}


def load_metrics(path: str) -> pd.DataFrame:
    """metrics.csv を読み込む．"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"metrics.csv が見つかりません: {path}")
    return pd.read_csv(path)


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_propagation_timeseries(df: pd.DataFrame, out_path: str) -> None:
    """集団レベル伝播の時系列図 (4 パネル) を保存する．"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor=COLOR_BG)
    fig.suptitle("Gao et al. (2023) S3 — 集団レベル伝播の時系列", fontsize=14)
    t = df["t"]

    # (1) 態度割合
    ax = axes[0, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["attitude_positive_frac"], color=COLOR_ATT, lw=2, marker="o", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("positive 態度割合")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("態度伝播 (positive 態度割合)")
    ax.grid(True, alpha=0.3)

    # (2) 感情分布 (積み上げ)
    ax = axes[0, 1]
    ax.set_facecolor(COLOR_BG)
    ax.stackplot(
        t,
        df["emotion_calm"],
        df["emotion_moderate"],
        df["emotion_intense"],
        labels=["calm", "moderate", "intense"],
        colors=[EMOTION_COLORS["calm"], EMOTION_COLORS["moderate"], EMOTION_COLORS["intense"]],
        alpha=0.9,
    )
    ax.set_xlabel("round t")
    ax.set_ylabel("感情分布")
    ax.set_ylim(0, 1)
    ax.set_title("感情伝播 (calm / moderate / intense)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) 行動採用曲線
    ax = axes[1, 0]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["behavior_adoption_rate"], color=COLOR_ADOPT, lw=2, marker="s", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("行動採用率 (repost/post)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("行動採用曲線")
    ax.grid(True, alpha=0.3)

    # (4) 情報カスケード規模
    ax = axes[1, 1]
    ax.set_facecolor(COLOR_BG)
    ax.plot(t, df["info_cascade_size"], color=COLOR_CASC, lw=2, marker="^", ms=3)
    ax.set_xlabel("round t")
    ax.set_ylabel("累積到達ノード数")
    ax.set_title("情報カスケード規模")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_network_snapshot(cfg: dict, out_path: str, seed: int = 0) -> None:
    """config から合成有向グラフを再構成しスナップショットを描く (近似)．

    Rust 側の生成器と完全一致はしない (RNG ストリームが別) が，トポロジ種別・規模の
    定性的な把握用に networkx で同種の有向グラフを生成して描画する．
    """
    import networkx as nx

    net = cfg.get("network", "ba")
    n = int(cfg.get("population", 50))
    n = min(n, 200)  # 描画は最大 200 ノードに制限

    if net == "er":
        g = nx.gnp_random_graph(n, float(cfg.get("er_p", 0.05)), seed=seed, directed=True)
    elif net == "ws":
        und = nx.watts_strogatz_graph(n, max(2, int(cfg.get("ws_k", 4))),
                                      float(cfg.get("ws_beta", 0.1)), seed=seed)
        g = und.to_directed()
    else:  # ba
        und = nx.barabasi_albert_graph(n, max(1, int(cfg.get("ba_m", 3))), seed=seed)
        g = und.to_directed()

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    pos = nx.spring_layout(g, seed=seed)
    degrees = np.array([d for _, d in g.degree()])
    sizes = 20 + 8 * degrees
    nx.draw_networkx_nodes(g, pos, node_size=sizes, node_color="#2196F3", alpha=0.8, ax=ax)
    nx.draw_networkx_edges(g, pos, alpha=0.12, arrows=True, arrowsize=6,
                           edge_color="#555555", ax=ax)
    ax.set_title(
        f"合成有向ネットワーク (近似): {net.upper()}, N={cfg.get('population')}",
        fontsize=12,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s3-tools visualize",
        description="Gao et al. (2023) S3 集団レベル伝播 可視化スクリプト",
    )
    p.add_argument(
        "--results_dir",
        "--results-dir",
        default="results/latest",
        help="Rust シミュレーションの出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {results_dir}/figures)",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="ネットワークスナップショット描画を抑止する．",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    out_dir = args.output_dir if args.output_dir else os.path.join(args.results_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Gao et al. (2023) S3 集団レベル伝播 可視化 ===")
    print(f"メトリクス: {metrics_path}")
    print(f"出力先:     {out_dir}")
    print("-----------------------------------------")

    print("[1/2] メトリクス時系列を保存中 ...")
    df = load_metrics(metrics_path)
    print(f"      {len(df)} round")
    save_propagation_timeseries(df, os.path.join(out_dir, "propagation_timeseries.png"))

    if not args.no_graph:
        cfg = load_config(args.results_dir)
        if cfg is not None:
            print("[2/2] ネットワークスナップショットを保存中 ...")
            save_network_snapshot(cfg, os.path.join(out_dir, "network_snapshot.png"))
        else:
            print("[2/2] config.json が無いためネットワーク描画をスキップ．")
    else:
        print("[2/2] --no-graph 指定によりネットワーク描画をスキップ．")

    print("-----------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
