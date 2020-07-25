import sys
from time import localtime, strftime

import cohort.get_cohort as cohort
import numpy as np
import settings
import ram
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv


def main(seed):

    #############################################################
    # Scikit-Survival Library
    # https://github.com/sebp/scikit-survival
    #
    # Random Survival Forest
    #
    #############################################################

    # Open file
    _file = open("files/cox-rsf.txt", "a")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("########## Init: " + time_string + "\n\n")
    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))

    cohort_x, cohort_y = cohort.cox()

    # Train / validation / test datasets
    train_size, x_train, x_val, x_test, y_train, y_val, y_test = cohort.train_test_split(seed, settings.size, cohort_x, cohort_y)

    # Transformation
    x_train_t = OneHotEncoder().fit_transform(x_train)
    x_train_t = np.column_stack(x_train_t.values)
    x_train_t = x_train_t.transpose()

    x_val_t = OneHotEncoder().fit_transform(x_val)
    x_val_t = np.column_stack(x_val_t.values)
    x_val_t = x_val_t.transpose()

    x_test_t = OneHotEncoder().fit_transform(x_test)
    x_test_t = np.column_stack(x_test_t.values)
    x_test_t = x_test_t.transpose()

    y_train = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_train)
    y_val = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_val)
    y_test = Surv.from_dataframe("hospital_expire_flag", "los_hospital", y_test)

    # KFold
    # cv = KFold(n_splits=settings.k, shuffle=True, random_state=seed)

    fold = [-1 for _ in range(train_size)] + [0 for _ in range(x_train.shape[0] - train_size)]
    cv = PredefinedSplit(test_fold=fold)

    # Params
    params = {'n_estimators': settings.n_estimators, 'max_depth': settings.max_depth,
              'min_samples_split': settings.split, 'min_samples_leaf': settings.leaf,
              'max_features': settings.max_features, 'n_jobs': settings.n_jobs, 'random_state': [seed]}

    # Train model
    rsf = RandomSurvivalForest()
    gcv = GridSearchCV(rsf, param_grid=params, cv=cv)
    gcv_fit = gcv.fit(x_train_t, y_train)

    # C-index score
    gcv_score_val = gcv.score(x_val_t, y_val)
    gcv_score_test = gcv.score(x_test_t, y_test)

    # Best Parameters
    _file.write("Best Parameters: " + str(gcv_fit.best_params_) + "\n")

    # C-Index
    _file.write("C-Index Validation: " + str(gcv_score_val) + "\n")
    _file.write("C-Index Test: " + str(gcv_score_test) + "\n")

    time_string = strftime("%d/%m/%Y, %H:%M:%S", localtime())
    _file.write("\n########## Final: " + time_string + "\n")

    print(strftime("%d/%m/%Y, %H:%M:%S", localtime()))

    # Close file
    _file.close()


if __name__ == "__main__":
    for seed in settings.seed:
        main(seed)
    # Only Unix systems
    # ram.memory_limit()
    # try:
    #     for seed in settings.seed:
    #         main(seed)
    # except MemoryError:
    #     sys.stderr.write('\n\nERROR: Memory Exception\n')
    #     sys.exit(1)
