## 
This is the official implementation for paper entitled "TabPFN: One Model to Rule Them All?".

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


## TabPFN inductive bias: 

### Interpolation vs extrapolation


### :chart_with_upwards_trend: :chart_with_downwards_trend: Comparison with LASSO regression

### :shield: :muscle: :gear: Robustness-efficiency trade-offs in classification
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
What the Scripts Do Automatically: 1) Iterate through all configured noise levels 2)Run 1000 repetitions per configuration 3)Output all results.

You can post-process the results to recreate Figures 6 and 7 from the paper.


## License Info
This code is offered under the [Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).