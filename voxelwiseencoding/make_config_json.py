#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 17:43:11 2021

@author: lmichalke
"""
import json
from process_bids import get_bids_filenames_for_econding

CONFIG_FILENAME = 'runconfig.json'

def make_config():
    # Set values by of user variables standard to 'None'  
    dict_key_list = ['bids_dir','output_dir','sub','ses','task','run','scope',
                     'bold_suffix','bold_extension','stim_suffix','stim_extension',
                     'json_extension','mask','bold_prep_params','encoding_params',
                     'lagging_params','skip_bids_validator','encoding_score_suffix',
                     'bold_files','bold_jsons','stim_tsvs','stim_jsons','rec',
                     'acq']
    arg = dict.fromkeys(dict_key_list,None)
    
    # User defined variables. Variable names follow BIDS label/value keys if defined there
    arg['bids_dir'] = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'
    arg['output_dir'] = 'output/'
    # The label(s) of the participant(s) that should be analyzed. The label corresponds to
    # sub-<participant_label> from the BIDS spec (so it does not include "sub-"). 
    # If this parameter is not provided all subjects should be analyzed. Multiple 
    # participants can be specified with a space separated list.
    arg['sub'] = ['03'] #BIDS
    # The label of the session to use. Corresponds to label in ses-<label> in the BIDS directory.
    arg['ses'] = ['01'] #BIDS
    # The task-label to use for training the voxel-wise encoding model. Corresponds 
    # to label in task-<label> in BIDS naming.
    #arg['task'] = ['aomovieCS','aomovieN4','aomovieS2'] #BIDS
    arg['task'] = ['aomovie'] #BIDS
    #arg['rec'] = ['noise'] #BIDS
    arg['acq'] = ['CS'] #BIDS
    #arg['run'] = ['01','04a','04b','05'] #BIDS
    arg['run'] = ['01'] #BIDS
    #arg['run'] = ['01','05'] #BIDS
    arg['scope'] = 'raw' #pyBIDS
    arg['bold_suffix'] = 'bold' #BIDS
    arg['bold_extension'] = ['.nii', '.nii.gz'] #BIDS
    arg['stim_suffix'] = 'stim' #BIDS
    arg['stim_extension'] = ['.tsv', '.tsv.gz'] #BIDS
    arg['json_extension'] = '.json' #BIDS 
    arg['encoding_score_suffix']='score' #BIDS
    # Whether or not to perform BIDS dataset validation
    arg['skip_bids_validator'] = True
    arg['mask'] = 'epi'
    # Some parameters for BOLD preprocessing
    bold_prep_params = {'standardize': 'zscore', 'detrend': True}
    arg['bold_prep_params'] = bold_prep_params
    # 'encoding_params' contains a dict with key-value pairs that are passed as 
    # parameters to the encoding model (by default RidgeCV from sklearn). For more
    # information on the parameters of RidgeCV see the sklearn documentation.
    # If a different encoding model is passed in 'estimator', parameters in
    # 'encoding_params' need to match that estimator's parameters.
    encoding_params = {'alphas': [1e-1, 1, 100, 1000], 'cv': 5, 'normalize': True}
    arg['encoding_params'] = encoding_params
    # and for lagging the stimulus - we want to include 6 sec stimulus segments to predict fMRI
    lagging_params = {'lag_time': 6}
    arg['lagging_params'] = lagging_params
    arg['estimator'] = None # default with None is 'RidgeCV'
    
    # get the files from the bids directory
    bold_files, bold_jsons, stim_tsvs, stim_jsons =\
         get_bids_filenames_for_econding(**arg)
         
    arg['bold_files'] = bold_files
    arg['bold_jsons'] = bold_jsons
    arg['stim_tsvs'] = stim_tsvs
    arg['stim_jsons'] = stim_jsons

    # write all params to config file
    with open(CONFIG_FILENAME, 'w', encoding='utf-8') as file:
        json.dump(arg, file, indent=4, ensure_ascii=False, sort_keys=True)
        
if __name__ == "__main__":
    make_config()