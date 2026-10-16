# Airfoil input

`airfoil_self_noise.dat` is the unmodified UCI Airfoil Self-Noise input downloaded
from https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat.
Dataset information and licensing: https://archive.ics.uci.edu/dataset/291/airfoil+self+noise.

It has 1,503 rows, five raw features, and one response column. The experiment
checks its dimensions, requires finite values, and records its SHA-256 in the
run manifest. The original notebook fetched this file at runtime; it is retained
here so the corrected rerun has a stable, inspectable input.
