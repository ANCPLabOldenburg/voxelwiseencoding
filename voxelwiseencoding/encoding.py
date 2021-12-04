import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV, MultiTaskLassoCV, LinearRegression, SGDRegressor
import joblib
import copy
from sklearn.preprocessing import StandardScaler
from scipy.special import btdtr


def product_moment_corr(x, y):
    """ Product-moment correlation for two ndarrays x, y """
    r, n = _product_moment_corr(x, y)
    # From scipy.stats.pearsonr:
    # As explained in the docstring, the p-value can be computed as
    #     p = 2*dist.cdf(-abs(r))
    # where dist is the beta distribution on [-1, 1] with shape parameters
    # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
    # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
    # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
    # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
    # to avoid a TypeError raised by btdtr when r is higher precision.)
    ab = n / 2 - 1
    prob = 2 * btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))
    return r, prob


def _product_moment_corr(x, y):
    x = StandardScaler().fit_transform(x)
    y = StandardScaler().fit_transform(y)
    n = x.shape[0]
    r = (1/(n-1)) * (x*y).sum(axis=0)
    return r, n


def get_model_plus_scores(X, y, estimator=None, cv=None, scorer=None,
                          run_start_indices=None, model_dump_path=None,
                          save_permutation_path=None, permutation_params=None,
                          **estimator_params):
    """ Returns multiple estimator trained in a cross-validation of
        the data and scores on the left-out folds.

    Parameters
        X : ndarray of shape (samples, features)
        y : ndarray of shape (samples, targets)
        estimator : None or estimator object that implements fit and predict.
                    If None, uses RidgeCV per default.
                    Can also be one of {'RidgeCV', 'Ridge', 'LinearRegression', 'MultiTaskLassoCV', 'SGDRegressor'}
                    to use the corresponding sklearn estimator.
                    Arguments may be passed to the estimator via **estimator_params.
        cv : int, None, 'leave-one-run-out', or a cross-validation object that
             implements a split method, default is None, optional.
             int specifies the number of cross-validation splits of a KFold cross validation.
             None defaults to a scikit-learn KFold cross-validation with default settings.
             'leave-one-run-out' specifies leave-one-run-out cross-validation.
             A scikit-learn-like cross-validation object needs to implement a
             split method for X and y.
        scorer : None or any scoring function that returns (score, pvalue), optional
                 default uses product moment correlation
        run_start_indices: list of int, optional, default None
                     Start index of each run which is used to group data into
                     cross-validation folds. If provided and cv is 'leave-one-run-out',
                     a leave-one-run-out cross-validation splitter will be used.
        model_dump_path: Path where to save dumps of model coefficients.
                         Default None, None means no saving
        save_permutation_path: If not None, permutation test is performed and results
                               are saved to this folder
        permutation_params: Dict, ignored if save_permutation_path is None. Contains:
            n_permutations: Number of permutations to perform.
            permutation_start_seed: Seeds which are used to shuffle targets start at this number and are
                                    incremented with each permutation. Useful to restart the permutation
                                    test at a specific seed.
            n_jobs: Number of cores to use for parallel processing of the permutation test.
        **estimator_params : additional arguments that will be passed to the estimator

    Returns
        tuple of n_splits estimators trained on training folds or single estimator
        if validation is False and scores for all concatenated out-of-fold predictions
    """
    if scorer is None:
        scorer = product_moment_corr
    if cv is None:
        cv = KFold()
    elif isinstance(cv, int):
        cv = KFold(n_splits=cv)
    elif cv == 'leave-one-run-out':
        if run_start_indices is None:
            raise Exception('Missing run_start_indices: run_start_indices has'
                            + ' to be defined when setting cv to "leave-one-run-out"')
        else:
            from leave_one_run_out_splitter import LeaveOneRunOutSplitter
            cv = LeaveOneRunOutSplitter(run_start_indices)
    elif not hasattr(cv, 'split'):
        raise Exception(f'Invalid cv parameter: {cv}. Has to be one of int, None,'
                        + '"leave-one-run-out", or cv object that implements a split'
                        + ' method.')

    if estimator is None or estimator == 'RidgeCV':
        estimator = RidgeCV(**estimator_params)
    elif estimator == 'Ridge':
        estimator = Ridge(**estimator_params)
    elif estimator == 'LinearRegression':
        estimator = LinearRegression(**estimator_params)
    elif estimator == 'MultiTaskLassoCV':
        estimator = MultiTaskLassoCV(**estimator_params)
    elif estimator == 'SGDRegressor':
        estimator = SGDRegressor(**estimator_params)

    score_list = []
    pval_list = []
    bold_prediction = []
    train_indices = []
    test_indices = []
    cv_fold_idx = 0
    for train, test in cv.split(X, y):
        model = copy.deepcopy(estimator).fit(X[train], y[train])
        bold_prediction.append(model.predict(X[test]))
        train_indices.append(train)
        test_indices.append(test)
        scores, pvals = scorer(y[test], bold_prediction[-1])
        score_list.append(scores[:, None])
        pval_list.append(pvals[:, None])
        print('Saving '+model_dump_path.format(cv_fold_idx))
        joblib.dump(model, model_dump_path.format(cv_fold_idx))
        cv_fold_idx += 1
    score_list = np.concatenate(score_list, axis=-1)
    pval_list = np.concatenate(pval_list, axis=-1)

    if save_permutation_path is not None:
        n_permutations = permutation_params['n_permutations']
        permutation_start_seed = permutation_params['permutation_start_seed']
        print(f'Starting permutation test with {n_permutations} permutations and '
              + f'initial seed {permutation_start_seed}.')
        from joblib import Parallel, delayed
        from sklearn.base import clone
        Parallel(n_jobs=permutation_params['n_jobs'], verbose=10)(
            delayed(_permutation_test)(
                clone(estimator),
                X,
                y,
                cv,
                seed + permutation_start_seed,
                save_permutation_path
            )
            for seed in range(n_permutations)
        )

    return score_list, bold_prediction, train_indices, test_indices, pval_list


def _permutation_test(estimator, X, y, cv, seed, save_path):
    score_list = []
    for train, test in cv.split(X, y):
        y_train_shuffled = np.random.RandomState(seed=seed).permutation(y[train])
        estimator.fit(X[train], y_train_shuffled)
        bold_prediction = estimator.predict(X[test])
        scores, _ = _product_moment_corr(y[test], bold_prediction)
        score_list.append(scores)
    scores_avg = np.mean(np.stack(score_list), axis=0)
    joblib.dump(scores_avg, save_path.format(seed))
