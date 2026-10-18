#!/bin/bash

# Use AutoML
python CATE_comparison.py --ss 500 --seed 1 --setup A --d 6 --noise-variance 0.5

# Use TabPFN
python CATE_comparison.py --ss 500 --seed 1 --setup A --d 6 --noise-variance 0.5 --tabpfn



