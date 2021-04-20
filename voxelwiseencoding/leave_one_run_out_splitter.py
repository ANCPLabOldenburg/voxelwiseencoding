#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class LeaveOneRunOutSplitter():
    """ Leave-one-run-out cross-validation splitter.
    
    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds by runs. Each fold is then used once as a
    validation while the k - 1 remaining folds form the training set.
    
    Parameters
    ----------
    run_start_indices : list of int
        Start index of each run which is used to group data into 
        cross-validation folds.
    """
    def __init__(self, run_start_indices):
        self.run_start_indices = run_start_indices
    
    def split(self, X, y=None):
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
            
        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """
        all_indices = np.arange(X.shape[0])
        n_runs = len(self.run_start_indices)
        for i in n_runs:
            start_index = self.run_start_indices[i]
            # End index of each run is one before the start of the next run.
            # For the last run, the end index is the index of the last sample in X.
            end_index = self.run_start_indices[i+1]-1 if i < n_runs-1 else X.shape[0]-1
            test_indices = np.arange(start_index, end_index)
            train_indices = np.setdiff1d(all_indices, test_indices)
            yield train_indices, test_indices