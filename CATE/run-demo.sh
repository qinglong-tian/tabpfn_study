#!/bin/bash

# Use AutoML
python CATE_comparison.py --ss 500 --seed 1 --setup A --var 0.5

# Use TabPFN
python CATE_comparison.py --ss 500 --seed 1 --setup A --var 0.5 --tabpfn



