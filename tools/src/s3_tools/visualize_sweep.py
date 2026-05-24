#!/usr/bin/env python3
"""
visualize_sweep.py — Gao et al. (2023) S3 スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
network × population の格子について最終集団指標 (positive 態度割合・行動採用率・
情報カスケード規模・感情強度) を集計し，ヒートマップと折れ線で可視化する．

Usage:
    uv run s3-tools visualize-sweep
    uv run s3-tools visualize-sweep --sweep_dir results/20260524_160000_sweep

Outputs:
    output_dir/
    ├── sweep_attitude_heatmap.png ← positive 態度割合 (network × population)
    ├── sweep_cascade_heatmap.png  ← 情報カスケード規模 (network × population)
    └── sweep_metrics_vs_population.png ← 指標 vs 人口規模 (network 別折れ線)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
NETWORK_ORDER = ["er", "ws", "ba"]
NETWORK_COLORS = {"er": "#2196F3", "ws": "#FF9800", "ba": "#4CAF50"}


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def _ordered_networks(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["network"].tolist()))
    ordered = [n for n in NETWORK_ORDER if n in present]
    ordered += [n for n in present if n not in ordered]
    return ordered


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(network, population) ごとに metric の試行平均をピボットする．"""
    agg = df.groupby(["network", "population"])[metric].mean().reset_index()
    table = agg.pivot(index="network", columns="population", values=metric)
    nets = [n for n in _ordered_networks(df) if n in table.index]
    return table.loc[nets]


def save_heatmap(table: pd.DataFrame, title: str, out_path: str, cmap: str) -> None:
    """network × population のヒートマップを保存する．"""
    fig, ax = plt.subplots(
        figsize=(2.0 + 1.3 * table.shape[1], 1.6 + 0.9 * table.shape[0]),
        facecolor=COLOR_BG,
    )
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([str(n).upper() for n in table.index])
    ax.set_xlabel("人口規模 N")
    ax.set_ylabel("ネットワーク種別")
    ax.set_title(title, fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_metrics_vs_population(df: pd.DataFrame, out_path: str) -> None:
    """人口規模に対する指標を network 別の折れ線で比較する．"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=COLOR_BG)
    metrics = [
        ("final_attitude_positive_frac", "positive 態度割合"),
        ("final_behavior_adoption_rate", "行動採用率"),
        ("final_info_cascade_size", "情報カスケード規模"),
    ]
    nets = _ordered_networks(df)
    for ax, (col, label) in zip(axes, metrics):
        ax.set_facecolor(COLOR_BG)
        for net in nets:
            sub = df[df["network"] == net]
            agg = sub.groupby("population")[col].mean().reset_index()
            ax.plot(
                agg["population"],
                agg[col],
                marker="o",
                lw=2,
                label=net.upper(),
                color=NETWORK_COLORS.get(net, "#888888"),
            )
        ax.set_xlabel("人口規模 N")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs 人口規模")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="s3-tools visualize-sweep",
        description="Gao et al. (2023) S3 スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Gao et al. (2023) S3 スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/3] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(f"      network {df['network'].nunique()} 種 × population {df['population'].nunique()} 種")

    print("[2/3] ヒートマップを保存中 ...")
    save_heatmap(
        pivot_metric(df, "final_attitude_positive_frac"),
        "最終 positive 態度割合 (network × population)",
        os.path.join(out_dir, "sweep_attitude_heatmap.png"),
        cmap="RdYlGn",
    )
    save_heatmap(
        pivot_metric(df, "final_info_cascade_size"),
        "最終 情報カスケード規模 (network × population)",
        os.path.join(out_dir, "sweep_cascade_heatmap.png"),
        cmap="YlGnBu",
    )

    print("[3/3] 指標 vs 人口規模 折れ線を保存中 ...")
    save_metrics_vs_population(df, os.path.join(out_dir, "sweep_metrics_vs_population.png"))

    print("-------------------------------------------------")
    print("ネットワーク別の平均 positive 態度割合:")
    for net in _ordered_networks(df):
        v = df[df["network"] == net]["final_attitude_positive_frac"].mean()
        print(f"  {net.upper():<3} → positivē = {v:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
