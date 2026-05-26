#!/bin/bash

for dataset in kin8nm house_8L puma8NH BNG_stock cmc
do
    for seed in $(seq 1 20)
    do
        echo "$dataset seed $seed"
        python real_data_calibration.py --dataset "$dataset" --seed "$seed"
    done
done
