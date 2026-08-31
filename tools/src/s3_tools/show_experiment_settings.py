"""s3-tools show-experiment-settings — 実行結果の設定表示．

runvault の run ディレクトリの config.json (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．run か sweep かは run.json の
`subcommand` で判別する．LLM 情報 (モデル・provider・温度) は run.json の `llm`
ブロック，呼び出し数と cache-hit 率は metrics.csv の run スコープ指標から採る．
legacy の flat な config.json / sweep_config.json も読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment s3 --latest --subcommand run --standalone
    runvault path --experiment s3 --latest --subcommand sweep

run/sweep の設定テーブルは `WS k / β` のような複合行を含み S3 固有なので本モジュール
に残す．`--json` の `kind` フィールドも S3 固有．

Usage:
    s3-tools show-experiment-settings
    s3-tools show-experiment-settings --results-dir "$(runvault path --experiment s3 --latest --subcommand run --standalone)"
    s3-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runvault.read import (
    config_parameters,
    load_run_meta,
    run_scope_metrics,
    runvault_path,
)

# runvault の experiment 名 (Rust 側 record::EXPERIMENT と揃える)．
EXPERIMENT = "s3"


def _load_config(results_dir: Path) -> tuple[dict, Path, str]:
    """run ディレクトリの実験条件と，それがどのサブコマンドのものかを返す．

    runvault の run では config.json は封筒で，条件は `parameters` の下にある．
    どのサブコマンドかは run.json の `subcommand` が答える (`sweep_config.json` は
    もう書かれない)．legacy の flat な config.json / sweep_config.json も読む．
    """
    # 設定が無いことは «まだ sweep_config.json の方かもしれない» という意味なので，
    # ここでは欠落を失敗として扱わない (下で sweep_config.json を見る)．
    params = config_parameters(results_dir, required=False)
    if params is not None:
        meta = load_run_meta(results_dir, required=False)
        if meta is not None:
            kind = meta["subcommand"]
        else:
            # legacy: 自前で書いていた config.json は "command" を持つ
            kind = "sweep" if params.get("command") == "sweep" else "run"
        return params, results_dir / "config.json", kind

    sweep_cfg = results_dir / "sweep_config.json"
    if sweep_cfg.exists():
        with sweep_cfg.open() as f:
            return json.load(f), sweep_cfg, "sweep"

    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def render_run_config(cfg: dict, source: Path, kind: str) -> str:
    """run / reproduce / baseline の設定テーブルを整形する (S3 固有)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"実行設定 ({kind})")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"ネットワーク種別 : {cfg.get('network', '-')}")
    lines.append(f"人口規模 N       : {cfg.get('population', '-')}")
    lines.append(f"ER p             : {cfg.get('er_p', '-')}")
    lines.append(f"WS k / β         : {cfg.get('ws_k', '-')} / {cfg.get('ws_beta', '-')}")
    lines.append(f"BA m             : {cfg.get('ba_m', '-')}")
    lines.append(f"伝播ラウンド T   : {cfg.get('rounds', '-')}")
    # LLM を呼ばない baseline の条件には Perception / LLM 設定が無い．
    if cfg.get("top_k") is not None:
        lines.append(f"top_k            : {cfg['top_k']}")
        lines.append(f"LLM Perception   : {cfg.get('llm_perception', '-')}")
    lines.append(f"発信源 seed数     : {cfg.get('seed_posters', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード (コア)    : {cfg.get('seed', '-')}")
    if cfg.get("llm_temperature") is not None:
        lines.append(f"LLM 温度         : {cfg['llm_temperature']}")
        lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    # reproduce / baseline だけが持つ条件．
    if cfg.get("model") is not None:
        lines.append(f"ベースライン     : {cfg['model']}")
    if cfg.get("mock") is not None:
        lines.append(f"mock / quick     : {cfg['mock']} / {cfg.get('quick', '-')}")
    if cfg.get("lt_theta") is not None:
        lines.append(f"LT θ / IC p      : {cfg['lt_theta']} / {cfg.get('ic_p', '-')}")
        lines.append(f"DeGroot 自己重み : {cfg.get('degroot_self_weight', '-')}")
    # 出力先は run ディレクトリそのものなので条件には含まれない (legacy のみ持つ)．
    if cfg.get("output_dir") is not None:
        lines.append(f"出力先           : {cfg['output_dir']}")
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


def llm_block(meta: dict | None, scoped: dict[str, float]) -> dict | None:
    """LLM の同一性 (run.json) と呼び出しの内訳 (metrics.csv) をまとめる．

    LLM を 1 度も呼ばない `baseline` の run には `llm` ブロックが無い．無いものは
    無いまま返し，呼び出し側で «表示しない» を選べるようにする．
    """
    llm = (meta or {}).get("llm")
    if llm is None:
        return None
    return {
        **llm,
        "calls": int(scoped.get("llm_calls", 0)),
        "cache_hits": int(scoped.get("llm_cache_hits", 0)),
        "cache_hit_rate": scoped.get("llm_cache_hit_rate", 0.0),
    }


def render_llm(llm: dict) -> str:
    lines: list[str] = []
    lines.append("LLM")
    lines.append("-" * 70)
    lines.append(f"provider         : {llm.get('provider', '-')}")
    lines.append(f"model            : {llm.get('model_snapshot', '-')}")
    lines.append(f"温度             : {llm.get('temperature', '-')}")
    lines.append(f"呼び出し         : {llm['calls']} 回")
    lines.append(
        f"cache-hit        : {llm['cache_hits']} ({llm['cache_hit_rate'] * 100:.1f}%)"
    )
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
        default=None,
        help="run ディレクトリ (省略時は runvault path --latest)",
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="runvault の results ルート (default: results)",
    )
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT,
        help=f"runvault の experiment 名 (default: {EXPERIMENT})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        results_dir = Path(runvault_path(args.experiment, args.results_root))
    else:
        results_dir = Path(os.path.realpath(args.results_dir))
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, cfg_path, kind = _load_config(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    # 指標を読むのは runvault の run だけ．legacy の metrics.csv は wide なので
    # run スコープの行という概念が無く，long 形式を前提とした読み取りは落ちる．
    meta = load_run_meta(results_dir, required=False)
    scoped = run_scope_metrics(results_dir) if meta is not None else {}
    llm = llm_block(meta, scoped)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "llm": llm}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "sweep":
            print(render_sweep_config(cfg, cfg_path))
        else:
            print(render_run_config(cfg, cfg_path, kind))
        if llm is not None:
            print(render_llm(llm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
