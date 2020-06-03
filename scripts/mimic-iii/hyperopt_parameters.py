import settings
from hyperopt import hp, fmin, tpe, Trials

# ---------------------
# Hyperparameter values
# ---------------------
# Layers                           {2, 4}
# Nodes per layer                  {64, 128, 256, 512}
# Dropout                          {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}
# Weigh decay                      {0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0}
# Batch size                       {64, 128, 256, 512, 1024}
# λ(penalty to the loss function)  {0.1, 0.01, 0.001, 0} - CoxCC(net, optimizer, shrink)


def hyperopt(experiment, parameters=None):
    space = {'num_nodes': hp.choice('num_nodes', [[64, 64], [128, 128], [256, 256], [512, 512],
                                                  [64, 64, 64, 64], [128, 128, 128, 128],
                                                  [256, 256, 256, 256], [512, 512, 512, 512]]),
             'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
             'weight_decay': hp.choice('weight_decay', [0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0]),
             'batch': hp.choice('batch', [64, 128, 256, 512, 1024]),
             'lr': hp.choice('lr', [0.01, 0.001, 0.0001])}

    if parameters == "cc":
            space['shrink'] = hp.choice('shrink', [0.1, 0.01, 0.001, 0])
    elif parameters == "deephit":
            space['alpha'] = hp.uniform('alpha', 0, 1)
            space['sigma'] = hp.choice('sigma', [0.1, 0.25, 0.5, 1, 2.5, 5, 10, 100])
            space['num_durations'] = hp.choice('num_durations', [50, 100, 200, 400])

    trials = Trials()

    # Tree of Parzen Estimators (TPE)
    best = fmin(experiment, space, algo=tpe.suggest, max_evals=settings.evals, trials=trials)

    return trials, best
