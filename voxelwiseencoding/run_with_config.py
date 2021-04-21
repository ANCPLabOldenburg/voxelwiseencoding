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
from process_bids import run_voxelwise_encoding, create_output_filename_from_args
from make_config_json import CONFIG_FILENAME

with open(CONFIG_FILENAME, 'r') as file:
    args = json.load(file)

bold_files = args['bold_files']
bold_jsons = args['bold_jsons']
stim_tsvs = args['stim_tsvs']
stim_jsons = args['stim_jsons']
# run analysis
ridges, scores, mask, bold_prediction, train_indices, test_indices = \
    run_voxelwise_encoding(bold_files, bold_jsons, stim_tsvs,stim_jsons,**args)

#save outputs
filename_output = create_output_filename_from_args(args['sub'], **vars(args))
joblib.dump(ridges, os.path.join(args.output_dir, '{0}_ridges.pkl'.format(filename_output)))
#joblib.dump(bold_prediction, os.path.join(args.output_dir, '{0}_{1}bold_prediction.pkl'.format(filename_output, identifier)))
joblib.dump(train_indices, os.path.join(args.output_dir, '{0}_train_indices.pkl'.format(filename_output)))
joblib.dump(test_indices, os.path.join(args.output_dir, '{0}_test_indices.pkl'.format(filename_output,)))

if mask:
    scores_bold = concat_imgs([unmask(scores_fold, mask) for scores_fold in scores.T])
bold_prediction_nifti = concat_imgs([unmask(bold, mask) for bold in bold_prediction])

save(scores_bold, os.path.join(args.output_dir, '{0}_scores.nii.gz'.format(filename_output)))
save(bold_prediction_nifti, os.path.join(args.output_dir, '{0}_bold_prediction.nii.gz'.format(filename_output)))

# also save config file that was used to perform analysis in the output folder
config_filename_output = os.path.join(args.output_dir, '{0}_{1}'.format(filename_output,CONFIG_FILENAME))
with open(config_filename_output, 'w', encoding='utf-8') as file:
    json.dump(args, file, indent=4, ensure_ascii=False, sort_keys=True)