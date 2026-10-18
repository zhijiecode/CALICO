#!/usr/bin/env bash
# Produce the official CALICO test submission files.
#
#   PY=python GPU=0 HP=out SUB=submit ./run_submit.sh
#
# HP points at a directory holding the validation json written by run_eval.sh
# (used for --hparams); SUB is where the submission files are written.
# See README section 7.
set -euo pipefail

PY=${PY:-python}
GPU=${GPU:-0}
HP=${HP:-out}
SUB=${SUB:-submit}
mkdir -p "$SUB"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# CIRR test1 (K=12) -> two files: <name>_cirr_...json and subset_<name>_cirr_...json
$PY calico_eval.py --dataset cirr_test1 --cqcc --csls 12 --csls_lam 0.5 \
    --submit --hparams "$HP/cirr_val.json" --sub-dir "$SUB" --gpu "$GPU"

# CIRCO test (K=20) -> one flat {"<query id>": [top-50 ids]} file
$PY calico_eval.py --dataset circo_test --cqcc --csls 20 --csls_lam 0.5 \
    --submit --hparams "$HP/circo_val.json" --sub-dir "$SUB" --gpu "$GPU"

# FashionIQ: submit the three category rankings through the official
# Fashion-IQ evaluation repository json format (see README section 7).
