"""s3-tools show-experiment-settings — 実行結果の設定表示．

results/{timestamp}/config.json (run) または
results/{timestamp}_sweep/sweep_config.json (sweep) を読み，実行時に使われた全
パラメータを整形表示する．存在すれば run_metadata.json の LLM 情報
(モデル・endpoint・温度・seed・cache-hit 率) も併せて表示する．
`results/latest` も解決される．

I/O (results-dir 解決・config/run_metadata ロード) と LLM メタデータブロックは
共有ヘルパ `socsim_tools` に委譲する (出力はバイト等価)．run/sweep の設定テーブルは
`WS k / β` のような複合行を含み S3 固有なので本モジュールに残す．`--json` の `kind`
フィールドも S3 固有．

Usage:
    s3-tools show-experiment-settings
    s3-tools show-experiment-settings --results-dir results/20260524_153000
    s3-tools show-experiment-settings --results-dir results/latest --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_metadata


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """config.json (run) か sweep_config.json (sweep) を探す．"""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (run) または sweep_config.json (sweep)"
    )


def render_run_config(cfg: dict, source: Path) -> str:
    """run 設定テーブルを整形する (S3 固有; `WS k / β` の複合行を含む)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (run)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"ネットワーク種別 : {cfg.get('network', '-')}")
    lines.append(f"人口規模 N       : {cfg.get('population', '-')}")
    lines.append(f"ER p             : {cfg.get('er_p', '-')}")
    lines.append(f"WS k / β         : {cfg.get('ws_k', '-')} / {cfg.get('ws_beta', '-')}")
    lines.append(f"BA m             : {cfg.get('ba_m', '-')}")
    lines.append(f"伝播ラウンド T   : {cfg.get('rounds', '-')}")
    lines.append(f"top_k            : {cfg.get('top_k', '-')}")
    lines.append(f"LLM Perception   : {cfg.get('llm_perception', '-')}")
    lines.append(f"発信源 seed数     : {cfg.get('seed_posters', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"出力先           : {cfg.get('output_dir', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_config(cfg: dict, source: Path) -> str:
    """sweep 設定テーブルを整形する (S3 固有; リスト項目を `, ` 連結する)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"ネットワーク種別 : {', '.join(cfg.get('network_values', []))}")
    pops = cfg.get("population_values", [])
    lines.append(f"人口規模         : {', '.join(str(x) for x in pops)}")
    lines.append(f"伝播ラウンド T   : {cfg.get('rounds', '-')}")
    lines.append(f"top_k            : {cfg.get('top_k', '-')}")
    lines.append(f"発信源 seed数     : {cfg.get('seed_posters', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="s3-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="実行結果ディレクトリ (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open() as f:
        cfg = json.load(f)
    meta = load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
