# common to all
seed = 20
size = 0.2

# k fold
k = 10

# classical cox
_alphas = [100, 10, 1, 0.1, 0.01, 1e-03, 1e-04, 1e-05]
_l1_ratios = [0, 0.001, 0.01, 0.1, 0.5]

# cox net
_alphas_cn = [100, 10, 1, 0.1, 0.01, 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
_l1_ratios_cn = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.01, 0.001]

# RSF
split = [2, 4, 6, 8]
leaf = [2, 8, 32, 64, 128]
n_estimators = [500, 1000]
max_features = ["auto", "sqrt"]
n_jobs = [-1]
random_state_rsf = [seed]

# neural network
estimates = 5
epochs = 512

# hyperopt
evals = 25