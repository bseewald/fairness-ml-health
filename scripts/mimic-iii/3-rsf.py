import time

import numpy as np
import pandas as pd
import cohort.get_cohort as cohort
import settings
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv


def main():

    #############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # Random Survival Forest
    #
    #############################################################

    old_score = 0

    # Open file
    _file = open("files/cox-rsf.txt", "a")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("########## Init: " + time_string + "\n\n")
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))

    cohort_X, cohort_y = cohort.cox()

    # Transformation
    Xt = OneHotEncoder().fit_transform(cohort_X)
    Xt = np.column_stack(Xt.values)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(Xt.transpose(), cohort_y)
    # X_train, X_test, y_train, y_test = train_test_split(Xt.transpose(), cohort_y, test_size=settings.size, random_state=settings.seed)

    # KFold
    cv = KFold(n_splits=settings.k, shuffle=True, random_state=settings.seed)

    # Params
    params = {'n_estimators': settings.n_estimators, 'min_samples_split': settings.split, 'min_samples_leaf': settings.leaf,
              'max_features': settings.max_features, 'n_jobs': settings.n_jobs, 'random_state': settings.random_state_rsf}

    # Train model
    rsf = RandomSurvivalForest()
    gcv = GridSearchCV(rsf, param_grid=params, cv=cv)
    gcv_fit = gcv.fit(X_train, y_train)

    # C-index score
    gcv_score = gcv.score(X_test, y_test)

    _file.write("gcv_score: " + str(gcv_score) + " old_score: " + str(old_score) + "\n")
    if gcv_score > old_score:

        old_score = gcv_score

        # Best Parameters
        _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

        # C-Index
        _file.write("C-Index: " + str(gcv_score) + "\n")

    time_string = time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime())
    _file.write("\n########## Final: " + time_string + "\n")
    print(time.strftime("%d/%m/%Y, %H:%M:%S", time.localtime()))

    _file.write("\n*** The last one is the best configuration! ***\n\n")

    # Close file
    _file.close()


if __name__ == "__main__":
    main()
