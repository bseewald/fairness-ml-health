import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cohort.get_cohort as cohort
import settings
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder


def main():

    ############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # CoxnetSurvivalAnalysis
    #
    #     """Cox's proportional hazard's model with elastic net penalty.
    #
    #     References
    #     ----------
    #     .. [1] Simon N, Friedman J, Hastie T, Tibshirani R.
    #            Regularization paths for Cox’s proportional hazards model via coordinate descent.
    #            Journal of statistical software. 2011 Mar;39(5):1.
    #
    ############################################################

    old_score = 0

    # Open file
    _file = open("files/cox-net.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")

    # Cohort
    cohort_X, cohort_y = cohort.cox()

    # OneHot
    cohort_Xt = OneHotEncoder().fit_transform(cohort_X)

    # Train / test samples
    X_train, X_test, y_train, y_test = train_test_split(cohort_Xt, cohort_y)
    # X_train, X_test, y_train, y_test = train_test_split(cohort_Xt, cohort_y, test_size=settings.size, random_state=settings.seed)

    # KFold
    cv = KFold(n_splits=settings.k, shuffle=True, random_state=settings.seed)

    _file.write("Tuning hyper-parameters\n\n")
    _file.write("Alphas: " + str(settings._alphas_cn) + "\n\n")

    # Training Model
    for ratio in settings._l1_ratios_cn:
        coxnet = CoxnetSurvivalAnalysis(alphas=settings._alphas_cn, l1_ratio=ratio).fit(cohort_Xt, cohort_y)

        _file.write("\nL1 Ratio: " + str(ratio) + "\n")

        # GridSearchCV
        gcv = GridSearchCV(coxnet, {"alphas": [[v] for v in coxnet.alphas_]}, cv=cv)

        # Fit
        gcv_fit = gcv.fit(X_train, y_train)

        # Score
        gcv_score = gcv.score(X_test, y_test)

        _file.write("gcv_score: " + str(gcv_score) + " old_score: " + str(old_score) + "\n")
        if gcv_score > old_score:

            old_score = gcv_score

            # Results
            results = pd.DataFrame(gcv_fit.cv_results_)

            alphas = results.param_alphas.map(lambda x: x[0])
            mean = results.mean_test_score
            std = results.std_test_score

            # Plot
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(alphas, mean)
            ax.fill_between(alphas, mean - std, mean + std, alpha=.15)
            ax.set_xscale("log")
            ax.set_ylabel("c-index")
            ax.set_xlabel("alpha")
            ax.axvline(gcv.best_params_['alphas'][0], c='C1')
            ax.grid(True)
            fig.savefig("img/coxnet.png", format="png", bbox_inches="tight")

            # Best Parameters
            _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

            # C-Index
            _file.write("C-Index: " + str(gcv_score) + "\n")

            # Coef
            coef = pd.Series(gcv_fit.best_estimator_.coef_[:, 0], index=cohort_Xt.columns)
            _file.write("Coeficients:\n" + str(coef[coef != 0]) + "\n\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()
