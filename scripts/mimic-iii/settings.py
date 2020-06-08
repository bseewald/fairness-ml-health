# common to all
seed = [3538733006, 1740342141, 2034474633, 1427421568, 2892461105, 3895719990, 2754073253, 2888635065, 1033490874,
        1353268974, 745472848, 2294877073, 3158590767, 2658318247, 2901899183, 2417472731, 3906059074, 3159426436,
        127000746, 896325390, 519400573, 530206849, 1982971951, 3324488161, 1436837179, 3862800656, 224796801,
        2556449784, 4057789202, 2934705183, 3006649751, 777972479, 1311426398, 915183445, 587257814, 480461828,
        3346154147, 3472978784, 1680114710, 3249445872, 3549818389, 3095040986, 3467813425, 567013934, 2231759630,
        513264208, 3071193438, 2186561917, 1429000735, 1744536459]

seed_fixed = 42

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