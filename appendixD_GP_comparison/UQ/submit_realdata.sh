#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPR_JOBS="${GPR_JOBS:-1}"

for dataset in kin8nm house_8L puma8NH BNG_stock bank8FM
do
    for seed in $(seq 1 20)
    do
        echo "$dataset seed $seed"
        "$PYTHON_BIN" real_data_calibration.py \
            --dataset "$dataset" \
            --seed "$seed" \
            --gpr-jobs "$GPR_JOBS"
    done
done
