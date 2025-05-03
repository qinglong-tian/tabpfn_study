# TabPFN: One Model to Rule Them All?
This is the official implementation for paper entitled "TabPFN: One Model to Rule Them All?".
All implementations are done in Python.
In the semi-supervised learning experiment, to compare with other baselines, we also use their R code.


# Instructions
Please make sure that you have the [TabPFN](https://github.com/PriorLabs/TabPFN) package installed.
Our code assumes that you can use the package online in the sense that the pretrained TabPFN can be automatically downloaded.
For offline usage, please refere to the instruction for TabPFN. You also need to specify the `model_path` wherever `TabPFNClassifier()` or `TabPFNRegressor()` is called.


# Reproduce experimental results
We provide the code for all case studies as well as the TabPFN inductive bias analysis.
Detailed instruction is provided for each one below.

## Case studies 

### Inference under semisupervised learning setting


### Prediction under covariate shift



### CATE estimation
All code for this experiment is located in the `CATE` directory.
First, navigate to the project directory:
```
cd CATE
```
The directory contains 5 scripts with main script `CATE_comparison.py`. Running this script will help you reproduce the results in Figure 3 and Table 4.
The `run-demo.sh` shows you an example of how to run the script in bash.
The rest of the files are helpers.
The `cate.py` defines all CATE estimators, `data_gen.py` can generate the dataset, `utils.py` are utility functions used.


## TabPFN inductive bias: 

### :signal_strength: Interpolation vs :rocket: extrapolation
All code for this experiment is located in the `extrapolation` directory.
First, navigate to the project directory:
```
cd extrpolation
```
The directory contains two Jupyter notebooks: 

- `extrapolation_1D.ipynb`: Handles 1D interpolation and extrapolation experiments. You can reproduce Figures 4 and 8 using this file.
- `extrapolation_2D.ipynb`: Manages 2D interpolation and extrapolation experiments. You can reproduce Figures 9 using this file.


### :chart_with_upwards_trend: :chart_with_downwards_trend: Comparison with LASSO regression

All code for this experiment is located in the `LASSO` directory.
First, navigate to the project directory:
```
cd LASSO
```
The directory contains three files:
- The `comparisons.py` contains the code for reproducing the results in Figure 5 and Table 2.

- The `run-demo.sh` shows you an example how to run the experiment in bash script. The example in this base is when training set is of size $n=50$ and $100$ features with sparsity $s=1$ under random seed $1$.
- The `ALE_plot.ipynb` provides the code to reproduce Figure 10 in the paper.


### :shield: :muscle: Robustness-efficiency trade-offs in classification
All code for this experiment is located in the `label_noise` directory.
First, navigate to the project directory:
```
cd label_noise
```
The directory contains two main scripts: `knn.py` and `comparison_LDA.py`.
 - The `knn.py` contains the code for kNN (clean) and kNN (noisy).
- The `comparison_LDA.py` contains the code for LDA and TabPFN.

Execute either script with default parameters (all noise levels + 1000 repetitions): e.g.,
```
python knn.py
```
What the scripts do automatically: 1) Iterate through all configured noise levels 2)Run 1000 repetitions per configuration 3)Output all results.

You can post-process the results to recreate Figures 6 and 7 from the paper.


## License Info
This code is offered under the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).