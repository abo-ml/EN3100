#!/usr/bin/env bash

set -euo pipefail

python -m compileall src
python -m src.data.align_data --help
python -m src.experiments.per_asset_evaluation --help
python -m src.experiments.download_equity_universe --help
python -m src.experiments.per_asset_equity_evaluation --help
python -m src.experiments.run_pipeline --help
