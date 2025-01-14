#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:43:11 2021

@author: lmichalke
"""
import json

import numpy as np

CONFIG_FILENAME = 'runconfig.json'


def make_config():
    # Set values by of user variables standard to 'None'  
    dict_key_list = ['bids_dir', 'output_dir', 'sub', 'ses', 'task', 'run', 'scope',
                     'bold_suffix', 'bold_extension', 'stim_suffix', 'stim_extension',
                     'json_extension', 'mask', 'bold_prep_params', 'encoding_params',
                     'lagging_params', 'skip_bids_validator', 'encoding_score_suffix',
                     'rec', 'acq', 'derivatives']
    arg = dict.fromkeys(dict_key_list, None)

    # User defined variables. Variable names follow BIDS label/value keys if defined there
    arg['bids_dir'] = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump'
    #    arg['bids_dir'] = '/data2/fyilmaz/simulated_data/WITHOUT_NOISE/'
    arg['subfolder'] = '/derivatives/fmriprep'
    # arg['bids_dir'] = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep'
    arg['output_dir'] = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/' \
                        + 'derivatives/encoding_results/temporal_lobe_mask/lagging0to-15.3_test_alpha_1e6/'
    #    arg['output_dir'] = '/data2/fyilmaz/simulated_data/output_without_noise/'
    # The label(s) of the participant(s) that should be analyzed. The label corresponds to
    # sub-<participant_label> from the BIDS spec (so it does not include "sub-"). 
    # If this parameter is not provided all subjects should be analyzed. Multiple 
    # participants can be specified with a space separated list.
    arg['sub'] = ['10']  # BIDS
    #    arg['sub'] = ['03','09'] #BIDS
    #    arg['sub'] = ["01","02","03","04","05","06","07","08","09","10"] #BIDS
    # The label of the session to use. Corresponds to label in ses-<label> in the BIDS directory.
    #    arg['ses'] = ['S2'] #BIDS
    #    arg['ses'] = ["01","02","03"] #BIDS
    #    arg['ses'] = ['N4','S2'] #BIDS
    #     arg['ses'] = ['CS'] #BIDS
    arg['ses'] = ['CS', 'N4', 'S2']  # BIDS
    # The task-label to use for training the voxel-wise encoding model. Corresponds 
    # to label in task-<label> in BIDS naming.
    arg['task'] = ['aomovie']  # BIDS
    #    arg['rec'] = ['audio'] #BIDS
    #     arg['rec'] = ['audioenv4k'] #BIDS
    arg['rec'] = ['audioenvzband']  # BIDS
    #    arg['rec'] = ['noise'] #BIDS
    #     arg['acq'] = ['CS','S2'] #BIDS
    # arg['acq'] = ['N4'] #BIDS
    arg['acq'] = ['CS', 'N4', 'S2']  # BIDS
    # arg['acq'] = ['CS'] #BIDS
    arg['run'] = ["02", "03", "04", "05", "06", "07"]  # BIDS
    #    arg['space'] = ['MNI152NLin2009cAsym'] #BIDS
    arg['space'] = ['MNI152NLin6Asym']  # BIDS
    #    arg['space'] = ['native'] #BIDS
    #    arg['space'] = ['T1w'] #BIDS
    arg['desc'] = ['preproc']  # BIDS
    #    arg['desc'] = ['simuheschls'] #BIDS
    #    arg['desc'] = ['preprocsmoothed'] #BIDS
    arg['descboldjson'] = ['preproc']  # BIDS
    #    arg['descboldjson'] = ['simuheschls'] #BIDS
    # arg['scope'] = 'derivatives' #pyBIDS
    #    arg['derivatives'] = ['sorted','fmriprep']
    # arg['derivatives'] = True
    arg['bold_suffix'] = 'bold'  # BIDS
    arg['bold_extension'] = ['.nii', '.nii.gz']  # BIDS
    arg['stim_suffix'] = 'stim'  # BIDS
    arg['stim_extension'] = ['.tsv', '.tsv.gz']  # BIDS
    arg['json_extension'] = '.json'  # BIDS
    arg['encoding_score_suffix'] = 'score'  # BIDS
    # Whether or not to perform BIDS dataset validation
    arg['skip_bids_validator'] = True
    arg['stim_always_CS'] = False
    arg['remove_confounds'] = True
    arg['confounds_desc'] = 'confounds'
    arg['confounds_suffix'] = 'timeseries'
    arg['confounds_extension'] = '.tsv'
    arg['confounds_to_exclude'] = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y',
                                   'trans_z', 'csf', 'white_matter']
    # Generate and use artificial bold data
    arg['use_artificial_bold'] = True

    arg['save_lagged_stim'] = False
    arg['save_preprocessed_bold'] = True
    # When running permutation tests with many or all cores, always remember to lower your process priority by
    # either starting the script with "nice -n 19 python run_with_config.py" or when the process is already running
    # "renice 19 -p PID" where PID is the process ID that you can see in top/htop
    # (see https://wiki.ubuntuusers.de/nice/). Otherwise you will hog all the CPU resources and make working on the
    # server impossible for others.
    # If memory usage is an issue, lower n_jobs. Each job might still be able to use multithreading internally,
    # if the backend of the used solver supports it.
    arg['permutation_test'] = False
    # permutation_params is ignored when permutation_test is False
    arg['permutation_params'] = {'n_permutations': 2000, 'permutation_start_seed': 0, 'n_jobs': 16}

    #    arg['mask'] = 'epi'
    #    arg['mask'] = arg['bids_dir'] + '/derivatives/fmriprep/ROIs/HeschisGyrus/mni_Heschl_ROI.nii.gz'
    arg['mask'] = arg['bids_dir'] + '/derivatives/fmriprep/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz'
    #    arg['mask'] = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz'
    #    arg['mask'] = '/data2/lmichalke/github/ANCP/voxelwiseencoding/voxelwiseencoding/sub-03_mask.nii.gz'
    # Some parameters for BOLD preprocessing
    TR = 0.85  # the actual TR parameter is saved in bold.json, this is just to set TR in bold_prep_params and lagging_params

    arg['skip_bold_preprocessing'] = False
    bold_prep_params = {'standardize': 'zscore', 'detrend': True, 't_r': TR,
                        'low_pass': 0.1, 'high_pass': 0.01}
    # bold_prep_params = {'standardize': 'zscore', 'detrend': True, 't_r': TR,
    #                     'low_pass': None, 'high_pass': 0.01}
    arg['bold_prep_params'] = bold_prep_params

    # and for lagging the stimulus - we want to include 6 sec stimulus segments to predict fMRI
    #    lagging_params = {'lag_time': 6}
    lagging_params = {'lag_time': 18 * TR, 'offset_stim': 0}
    # lagging_params = {'lag_time': None,'offset_stim':0}
    #    lagging_params = {'lag_time': 7*TR,'offset_stim':4}
    arg['lagging_params'] = lagging_params

    # arg['estimator'] = 'RidgeCV'  # default with None is 'RidgeCV'
    arg['estimator'] = 'Ridge'  # default with None is 'RidgeCV'
    # arg['estimator'] = 'LinearRegression' # default with None is 'RidgeCV'
    # arg['estimator'] = 'MultiTaskLassoCV' # default with None is 'RidgeCV'

    # 'encoding_params' contains a dict with key-value pairs that are passed as
    # parameters to the encoding model (by default RidgeCV from sklearn). For more
    # information on the parameters of RidgeCV see the sklearn documentation.
    # If a different encoding model is passed in 'estimator', parameters in
    # 'encoding_params' need to match that estimator's parameters.
    # encoding_params = {'alphas': [1e-3, 1e-2, 2e-2, 1e-1], 'cv': 'leave-one-run-out', 'normalize': True}
    encoding_params = {'alpha': 1e6, 'cv': 'leave-one-run-out', 'normalize': True}
    # encoding_params = {'alphas': list(np.geomspace(20, 200, 20)), 'cv': 'leave-one-run-out', 'normalize': True}
    arg['encoding_params'] = encoding_params

    # write all params to config file
    with open(CONFIG_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(arg, file, indent=4, ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    make_config()
