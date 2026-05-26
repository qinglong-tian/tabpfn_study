#!/bin/bash

# python acic2017.py --setup group_corr --scenario 111 --method dr

for method in s
do
  for set in iid nonadditive group_corr
  do
    for mode in 000 001 010 011 100 101 110 111
    do 
      python acic2017.py --setup $set --scenario $mode --method $method
      done
    done
done