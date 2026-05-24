"""s3-tools — Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 ツール統合 CLI．

Usage:
    s3-tools visualize [...]
    s3-tools visualize-sweep [...]
    s3-tools show-experiment-settings [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` (論文 Table 4/5 の一括再現・実データ MSED/Cor 整合) は Phase 3 で
実装予定 (未提供)．
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="s3-tools",
        description="Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 可視化・分析ツール",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="単一実行結果 (態度割合・感情分布・行動採用・カスケード) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="スイープ結果 (network × population の集団指標) の可視化",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from s3_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from s3_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from s3_tools.show_experiment_settings import main as run_main

        run_main(rest)
    else:
        # 未知のコマンドは argparse のエラーメッセージに委ねる
        parser.parse_args(argv)


if __name__ == "__main__":
    main()
