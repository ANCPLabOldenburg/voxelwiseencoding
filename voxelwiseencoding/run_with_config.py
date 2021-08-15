#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:06:04 2021

@author: lmichalke
"""

import json
import joblib, os
from nilearn.masking import unmask
from nilearn.image import concat_imgs
from nibabel import save
from process_bids import run_model_for_subject, create_output_filename_from_args
from process_bids import get_bids_filenames_for_econding
from make_config_json import CONFIG_FILENAME

def run_analysis(subject,acq,kwargs):
    print(f'Running encoding model on subject {subject}, condition {acq}')
    args = kwargs.copy()
    args['sub'] = [subject]
    args['acq'] = [acq]
    
    # get the files from the bids directory
    if args.get('remove_confounds'):
        bold_files, bold_jsons, stim_tsvs, stim_jsons, confound_tsvs =\
             get_bids_filenames_for_econding(**args)
        args['confound_tsvs'] = confound_tsvs
    else:
        bold_files, bold_jsons, stim_tsvs, stim_jsons =\
             get_bids_filenames_for_econding(**args)
    
    # make output directory
    output_dir = os.path.join(args['output_dir'],f'sub-{subject}/acq-{acq}/')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename_output = create_output_filename_from_args(subject,None,args['task'][0],acq)
    args['model_dump_path'] = os.path.join(output_dir, filename_output+'_desc-ridgesfold{0}.pkl')
    if args.get('save_lagged_stim'):
        lagged_stim_dir = os.path.join(output_dir,'lagged_stim/')
        if not os.path.exists(lagged_stim_dir):
            os.makedirs(lagged_stim_dir)
        args['save_lagged_stim_path'] = os.path.join(lagged_stim_dir, filename_output+'_desc-laggedstim{0}.pkl')
    if args.get('save_preprocessed_bold'):
        args['save_preprocessed_bold_path'] = os.path.join(output_dir, filename_output+'_desc-boldpreprocessed.nii.gz')
#        args['save_preprocessed_bold_path'] = os.path.join(output_dir, filename_output+'_desc-boldpreprocessed.pkl')
    # run analysis
    scores, masks, bold_prediction, train_indices, test_indices = \
        run_model_for_subject(bold_files, bold_jsons, stim_tsvs, stim_jsons, **args)

    #save outputs
    #joblib.dump(ridges, os.path.join(output_dir, '{0}_desc-ridges.pkl'.format(filename_output)))
    joblib.dump(masks, os.path.join(output_dir, '{0}_desc-masks.pkl'.format(filename_output)))
    joblib.dump(train_indices, os.path.join(output_dir, '{0}_desc-trainindices.pkl'.format(filename_output)))
    joblib.dump(test_indices, os.path.join(output_dir, '{0}_desc-testindices.pkl'.format(filename_output)))
    
    scores_bold = concat_imgs([unmask(scores_fold, mask) for scores_fold, mask in zip(scores.T, masks)])
    save(scores_bold, os.path.join(output_dir, '{0}_desc-scores.nii.gz'.format(filename_output)))
    
    bold_prediction_nifti = unmask(bold_prediction, masks[0])
    save(bold_prediction_nifti, os.path.join(output_dir, '{0}_desc-boldprediction.nii.gz'.format(filename_output)))
    
    # also save config file that was used to perform analysis in the output folder
    config_filename_output = os.path.join(output_dir, '{0}_{1}'.format(filename_output,CONFIG_FILENAME))
    with open(config_filename_output, 'w', encoding='utf-8') as file:
        json.dump(args, file, indent=4, ensure_ascii=False, sort_keys=True)
    
if __name__ == "__main__":  
    import time
    tic = time.time()
    with open(CONFIG_FILENAME, 'r') as file:
        args = json.load(file)
    
    for subject in args['sub']:
        for acq in args['acq']:
            run_analysis(subject,acq,args)                  
    toc = time.time()
    print('\nElapsed time: {:.2f} s'.format(toc - tic))