"""s3-tools — Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 ツール統合 CLI．

Usage:
    s3-tools visualize [...]
    s3-tools visualize-sweep [...]
    s3-tools show-experiment-settings [...]
    s3-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

`reproduce` は Rust の `s3 reproduce` が書き出した再現ディレクトリ
(reproduce_summary.json + s3_metrics.csv + baseline_*.csv) を読み，observed-vs-paper
の照合結果と，S³ vs 古典ベースライン (LT/IC/Voter/DeGroot) の伝播比較図を生成する．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示の
実体 (visualize / visualize_sweep / show_experiment_settings) は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="s3-tools",
    description="Gao et al. (2023) S3 LLM ソーシャルネットワーク伝播 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (態度割合・感情分布・行動採用・カスケード) の可視化",
            "s3_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (network × population の集団指標) の可視化",
            "s3_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "s3_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "一括再現の observed-vs-paper 照合表示 + S³ vs 古典ベースライン伝播比較図",
            "s3_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
