#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
def start_here_gather_files()


Files we need in the end 
========================
#layout=BIDSLayout(bids_dir, derivatives=True)
#layout.get(suffix='bold',run='01', task='continuous', scope='raw',return_type='filename')

# bold_files is a list of bold files
# bold_folder ='/data2/tweis/StudySparseForrest/DATA/data_ForrestSparse/sub-OL2275/ses-01/func'
# in bold_folder do bold_glob='bold_glob='sub-OL2275_ses-01_task-continuous_desc-preproc_*bold*.nii.gz

# task-metadata from task-task*bold.json. Seems to be some metadata coming from the scanner
#   example file /data2/tweis/StudySparseForrest/DATA/data_ForrestSparse/task-continuous_bold.json
# where task_meta = json.load(fl) loads the metadata into task meta
# Seems like he assumes that this metada is the same for all runs (there are no runs in the original code)

# stim_tsv appears to be a list of stimulus files correponding to the bold files in order  
# stim_glob = 'sub-OL2275_ses-01_task-continuous_desc-preproc_*stim'
# stim_tsv = glob(os.path.join(bold_folder, '.'.join([stim_glob, 'tsv.gz'])))
# this should return a list of files ending with *stim.tsv.gz

# stim_json appears to hold a list of jsons corresponding to the stim_tsv-files
# stim_json = sorted(glob(os.path.join(bold_folder, '.'.join([stim_glob, 'json']))))





Created on Fri Mar 26 17:48:37 2021

@author: rieger
"""
import json
from voxelwiseencoding.process_bids import get_bids_filenames_for_econding, run_voxelwise_encoding

# Set values by of user variables standard to 'None'  
dict_key_list=['bids_dir','output_dir','sub','ses','task','run','scope','bold_suffix','bold_extension','stim_suffix','stim_extension','json_extension','mask','bold_prep_params_filename','encoding_params_filename','lagging_params_filename','skip_bids_validator','encoding_score_suffix']
arg=dict.fromkeys(dict_key_list,None)

# User defined variables. Variable names follow BIDS label/value keys if defined there
arg['bids_dir']='/home/rieger/DDATA/DATA/OL_MRT/data_ForrestSparse/'
arg['output_dir']='output'
arg['sub']=['OL2275'] #BIDS
arg['ses'] = ['01'] #BIDS
arg['task'] = ['continuous'] #BIDS
arg['run'] = ['01'] #BIDS
arg['scope'] = 'raw' #pyBIDS
arg['bold_suffix'] = 'bold' #BIDS
arg['bold_extension'] = ['.nii', '.nii.gz'] #BIDS
arg['stim_suffix'] = 'stim' #BIDS
arg['stim_extension'] = ['.tsv', '.tsv.gz'] #BIDS
arg['json_extension'] = 'json' #BIDS 
arg['encoding_score_suffix']='score' #BIDS
arg['skip_bids_validator'] = True
arg['mask']='epi'
arg['bold_prep_params_filename']='bold_prep_config.json'
arg['encoding_params_filename']='encoding_config.json'
arg['lagging_params_filename']='lagging_config.json'
arg['estimator']=None # default with None is 'RidgeCV'

# Some parameters for BOLD preprocessing
bold_prep_params = {'standardize': 'zscore', 'detrend': True}

# We also set some parameters for model fitting
ridge_params = {'alphas': [1e-1, 1, 100, 1000], 'cv': 5, 'normalize': True}

# and for lagging the stimulus - we want to include 6 sec stimulus segments to predict fMRI
lagging_params = {'lag_time': 6}

# get the files from the bids directory
bold_files, bold_jsons, stim_tsvs, stim_jsons =\
     get_bids_filenames_for_econding(**arg)

#write bold prep, ridge and lagging params to files 
with open(arg['bold_prep_params_filename'], 'w+') as fl:
    json.dump(bold_prep_params, fl)
    
with open(arg['encoding_params_filename'], 'w+') as fl:
    json.dump(ridge_params, fl)
    
with open(arg['lagging_params_filename'], 'w+') as fl:
    json.dump(lagging_params, fl)

#HERE we start model estimation. We may need to loop around subjects, sessions, tasks, and runs depending which data can be appended for model estimation.

run_voxelwise_encoding(bold_files, bold_jsons, stim_tsvs,stim_jsons,
                       **arg)
