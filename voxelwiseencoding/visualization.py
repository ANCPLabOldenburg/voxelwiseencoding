#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:18:41 2021

@author: lmichalke
"""
import numpy as np
import os

def plot_scores(scores_path,save_path):
    print('Plotting ',save_path)
    from nilearn.image import mean_img
    from nilearn import plotting
    mean_scores = mean_img(scores_path)
    plotting.plot_stat_map(mean_scores, threshold=0.05, output_file=save_path)

def get_arg_highscore(scores_path):
    import nibabel as nib
    n1_img = nib.load(scores_path)
    data = n1_img.get_fdata()
    x,y,z,fold = data.shape
    highscore = np.argmax(data[...,0])
    #print(np.max(data[...,0]))
    # flattened array = i * (len(y)*len(z))+ j * (len(z)) + k
    arg_x, rem_x = highscore // (y * z), highscore % (y * z)
    arg_y, rem_y = rem_x // z, rem_x % z
    arg_z = rem_y
    arg_highscore = (arg_x, arg_y, arg_z)
    #print(data[arg_highscore+(0,)])
    return arg_highscore
    
def product_moment_corr(x,y):
    '''Product-moment correlation for two ndarrays x, y'''
    n = x.shape[0]
    r = (1/(n-1))*(x*y).sum(axis=0)
    return r

def plot_highest_score_bold_predicted_vs_actual(path_predicted,path_actual,arg_highscore,save_path):
    print('Plotting ',save_path)
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    bold_predicted = nib.load(path_predicted)
    bold = nib.load(path_actual)
    #print(bold_predicted.shape)
    #print(bold.shape)
    bold_predicted_high = bold_predicted.get_fdata()[arg_highscore][:1038]
    bold_high = bold.get_fdata()[arg_highscore]
    #print(bold_predicted_high.shape)
    bold_high = zscore(bold_high)
    bold_predicted_high = zscore(bold_predicted_high)
    plt.plot(bold_predicted_high,label='Predicted bold')
    plt.plot(bold_high,label='Actual bold')
    plt.legend()
    plt.xlabel('Volume')
    plt.ylabel('Zscore')
    plt.title('Predicted vs actual bold in highest correlation voxel r=%.4f' %\
              product_moment_corr(bold_predicted_high,bold_high))
    plt.savefig(save_path)
    plt.close()
    #print(product_moment_corr(bold_predicted_high,bold_high))

def get_max_coefs(ridges_path,mask_path,arg_highscore):
    import joblib
    from nilearn.masking import unmask
    ridges = joblib.load(ridges_path)
    mask = joblib.load(mask_path)
    coefs = unmask(ridges.coef_.T,mask)
    #print(ridges.coef_.shape)
    #print(coefs.shape)
    max_coefs = coefs._dataobj[arg_highscore]
    nmel = 48
    return max_coefs.reshape((-1,nmel))
    
def plot_max_coefs(max_coefs,save_path):
    print('Plotting ',save_path)
    import matplotlib.pyplot as plt
    nmel = 48
    nmel_stride = 4
    mel_frequencies = [100, 162, 224, 286, 348, 410, 472, 534, 596, 658, 721, 
                       783, 845, 907, 969, 1032, 1100, 1173, 1251, 1333, 1421,
                       1515, 1616, 1722, 1836, 1957, 2087, 2225, 2372, 2528, 
                       2696, 2874, 3064, 3266, 3482, 3712, 3957, 4219, 4497, 
                       4795, 5112, 5449, 5809, 6193, 6603, 7039, 7504, 8000]
    plt.imshow(max_coefs.T,aspect='auto',origin='lower')
    plt.gca().set_yticks(np.arange(0,nmel,nmel_stride))
    plt.gca().set_yticklabels(mel_frequencies[::nmel_stride])
    plt.colorbar()
    plt.xlabel('Time lag ')
    plt.ylabel('Mel frequency (Hz)')
    plt.savefig(save_path)
    plt.close()

def plot_max_coef_diff(max_coefs,max_coefs_CS,save_path):
    from scipy.stats import zscore
    diff = zscore(max_coefs,axis=1)-zscore(max_coefs_CS,axis=1)
    plot_max_coefs(diff,save_path)

def plot_mel_spectrogram():
    import matplotlib.pyplot as plt
    #tsv_fl = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep/sub-03/ses-CS/func/sub-03_ses-CS_task-aomovie_acq-CS_run-02_recording-audio_stim.tsv.gz'
    tsv_fl = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/sorted/sub-03/ses-CS/func/sub-03_ses-CS_task-aomovie_acq-CS_run-02_stim.tsv.gz'
    x = np.loadtxt(tsv_fl, delimiter='\t')
    #print(x.shape)
    plt.imshow(x.T,aspect='auto',origin='lower')
    plt.show()
    #plt.savefig(save_path)
    
if __name__=='__main__':
    cwd = os.getcwd()
#    output_dirs = [cwd+'/output_windowlen25_overlap0_masked_alwaysCS_removeconfounds/']
    output_dirs = [cwd+'/output_windowlen25_overlap0_masked_alwaysCS_removeconfounds/',
                   cwd+'/output_windowlen25_overlap0_masked/',
                   cwd+'/output_windowlen25_overlap0_masked_alwaysCS/',
                   cwd+'/output_windowlen25_overlap10_unmasked/',
                   cwd+'/output_windowlen850_masked/',
                   cwd+'/output_windowlen850_unmasked/']
    subjects = ['03','09']
    conditions = ['CS','N4','S2']
    do_scores = True
    do_bold_predicted_vs_actual = True
    do_max_coefs = True
    for output_dir in output_dirs:
        for sub in subjects:
            sub_dir = os.path.join(output_dir,f'sub-{sub}/')
            if not os.path.exists(sub_dir):
                continue
            max_coefs_CS = None
            for acq in conditions:
                acq_dir = os.path.join(sub_dir,f'acq-{acq}/')
                if not os.path.exists(acq_dir):
                    continue
                bids_str = acq_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                scores_path = bids_str + 'scores.nii.gz'
                scores_save = bids_str + 'scores.png'
                bold_predicted = bids_str + 'boldprediction.nii.gz'
                bold_save = bids_str + 'boldprediction.png'
                ridges_path = bids_str + 'ridgesfold0.pkl'
                mask_path = bids_str + 'mask.pkl'
                max_coefs_save = bids_str + 'maxcoefs.png'
                max_coefs_diff_save = bids_str + 'maxcoefsdiff.png'
                bold_actual = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                    +f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/sub-{sub}_ses-{acq}'\
                    +f'_task-aomovie_acq-{acq}_run-2_space-MNI152NLin2009cAsym_res-'\
                    +'2_desc-preproc_bold.nii.gz'
                if do_scores:
                    plot_scores(scores_path,scores_save)
                if do_bold_predicted_vs_actual or do_max_coefs:
                    arg_highscore = get_arg_highscore(scores_path)
                if do_bold_predicted_vs_actual:
                    plot_highest_score_bold_predicted_vs_actual(bold_predicted,bold_actual,
                                                                arg_highscore,bold_save)
                if do_max_coefs:
                    max_coefs = get_max_coefs(ridges_path,mask_path,arg_highscore)
                    plot_max_coefs(max_coefs,max_coefs_save)
                    if acq=='CS':
                        max_coefs_CS = max_coefs
                    elif max_coefs_CS is not None:
                        plot_max_coef_diff(max_coefs,max_coefs_CS,max_coefs_diff_save)