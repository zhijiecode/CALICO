#!/usr/bin/env bash
# Run the frozen CALICO method on the public validation splits.
#
#   PY=python GPU=0 OUT=out ./run_eval.sh
#
# K = 12 on CIRR, K = 20 on CIRCO / FashionIQ.
set -euo pipefail

PY=${PY:-python}
GPU=${GPU:-0}
OUT=${OUT:-out}
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

$PY calico_eval.py --dataset cirr_val  --cqcc --csls 12 --csls_lam 0.5 \
    --gpu "$GPU" --out "$OUT/cirr_val.json"

$PY calico_eval.py --dataset circo_val --cqcc --csls 20 --csls_lam 0.5 \
    --gpu "$GPU" --out "$OUT/circo_val.json"

for c in dress shirt toptee; do
    $PY calico_eval.py --dataset "fiq_${c}_val" --cqcc --csls 20 --csls_lam 0.5 \
        --gpu "$GPU" --out "$OUT/fiq_${c}_val.json"
done
