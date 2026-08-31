#!/usr/bin/env python3
"""スイープの «1 行 1 試行» の表．

run ディレクトリの読み方そのものは `runvault.read` にある．ここに残るのは S³ 固有の
部分だけ — どの列を持つ表なのか (`network` / `population` / `final_emotion_*` …) で
ある．モデルの話であって run ディレクトリの読み方ではないので，共通部品には置かない．
"""
from __future__ import annotations

import json
import os

import pandas as pd
from runvault.read import (
    config_parameters,
    metrics_wide,
    run_scope_metrics,
    sweep_children,
)

__all__ = ["sweep_summary_table"]

#: 最終ラウンドの値から作る列 (列名 → metrics.csv の指標名)．
_FINAL_COLUMNS = {
    "final_attitude_positive_frac": "attitude_positive_frac",
    "final_emotion_calm": "emotion_calm",
    "final_emotion_moderate": "emotion_moderate",
    "final_emotion_intense": "emotion_intense",
    "final_behavior_adoption_rate": "behavior_adoption_rate",
    "final_info_cascade_size": "info_cascade_size",
}


def sweep_summary_table(sweep_dir: str | os.PathLike) -> pd.DataFrame:
    """1 行 1 試行のサマリ表を用意する．

    runvault ではこの表はファイルとして存在しない．sweep 親の子 run
    (`lineage.parent_run_uid` が親の `run_uid`) を集め，各子の `config.json` の
    `parameters` と `metrics.csv` の最終ラウンド・run スコープ指標から組み直す．
    legacy のスイープには `sweep_summary.csv` があるのでそれを読む．

    どちらの経路でも `run_dir` 列を付けるので，呼び出し側は条件からディレクトリ名を
    組み立てなくてよい．
    """
    sweep_dir = str(sweep_dir)
    legacy = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(legacy):
        df = pd.read_csv(legacy)
        df["run_dir"] = sweep_dir
        return df

    children = sweep_children(sweep_dir)
    if not children:
        raise SystemExit(
            f"エラー: この sweep 親に紐づく子 run が見つかりません: {sweep_dir}\n"
            "  子 run は lineage.parent_run_uid で親を指します．"
            "親と子が同じ results ルートにあるか確認してください．"
        )

    rows: list[dict] = []
    for child in children:
        params = config_parameters(child) or {}
        with open(os.path.join(child, "run.json")) as f:
            meta = json.load(f)
        scoped = run_scope_metrics(child)
        last = metrics_wide(os.path.join(child, "metrics.csv")).iloc[-1]
        row = {
            "network": params.get("network"),
            "population": params.get("population"),
            # 同一条件の何本目かは runvault の rng.replicate_index が持つ．
            "run": (meta.get("rng") or {}).get("replicate_index"),
            "seed": params.get("seed"),
            "converged": bool(scoped.get("converged", 0.0)),
            "final_round": int(scoped.get("final_round", last["step"])),
        }
        row.update(
            {column: float(last[name]) for column, name in _FINAL_COLUMNS.items()}
        )
        row["final_info_cascade_size"] = int(row["final_info_cascade_size"])
        row["cache_hit_rate"] = float(scoped.get("llm_cache_hit_rate", 0.0))
        row["run_dir"] = child
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["network", "population", "run"])
        .reset_index(drop=True)
    )
