#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:18:41 2021

@author: lmichalke
"""
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from nilearn.image import load_img
from nilearn.masking import apply_mask

OUTPUT_BASE = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
              +'derivatives/encoding_results/'
    
def plot_scores(scores_path,save_path,glassbrain_save):
    print('Plotting',save_path)
    from nilearn.image import mean_img, load_img
    from nilearn import plotting
    temporal_lobe_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/"\
        + "derivatives/fmriprep/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz"
    heschl_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/"\
        + "derivatives/fmriprep/ROIs/HeschisGyrus/mni_Heschl_ROI.nii.gz"
    mean_scores = mean_img(scores_path)
    img = load_img(scores_path)
    data = img.dataobj
    avg_max = np.max(mean_scores.dataobj)
    avg_argmax = np.unravel_index(np.argmax(mean_scores.dataobj), mean_scores.shape)
    total_max = np.max(data)
    total_argmax = np.unravel_index(np.argmax(data), data.shape)
    fold0_max = np.max(data[...,0])
    fold0_argmax = np.unravel_index(np.argmax(data[...,0]), data.shape[:-1])
    title = 'Avg%.3f%sfold0 %.3f%stotal%.3f%s' % \
        (avg_max, str(avg_argmax), fold0_max, str(fold0_argmax), total_max, str(total_argmax))
    thresh = 0.0 #0.05
    display = plotting.plot_stat_map(mean_scores, threshold=thresh, title=title)
    display.add_markers([avg_argmax],marker_color='c')
    display.add_markers([fold0_argmax],marker_color='y')
    #display.add_markers([total_argmax],marker_color='m')
#    display.add_contours(temporal_lobe_mask,filled=False,colors='m')
#    display.add_contours(heschl_mask,filled=False,colors='g')
    plt.gcf().savefig(save_path)
    plt.close()
    display = plotting.plot_glass_brain(mean_scores, threshold=thresh, colorbar=True,
                                        display_mode='lzry', plot_abs=False)
    display.add_contours(temporal_lobe_mask,filled=False,colors='m')
    display.add_contours(heschl_mask,filled=False,colors='g')
    display.add_markers([avg_argmax],marker_color='c')
    display.add_markers([fold0_argmax],marker_color='y')
    #display.add_markers([total_argmax],marker_color='m')
    # proxy artist trick to make legend
    from matplotlib.patches import Rectangle
    cont1 = Rectangle((0,0),1,1,fc="magenta")
    cont2 = Rectangle((0,0),1,1,fc="green")
    plt.legend([cont1,cont2],['Temporal lobe','Heschl gyrus'])
    plt.gcf().savefig(glassbrain_save)
    plt.close()
    
def plot_avg_score(scores_path,mask_path,cond=None,offset=0):
    scores = load_img(scores_path)
    mask = joblib.load(mask_path)
    scores = apply_mask(scores,mask[0])
#    scores = np.square(scores) # R to R^2
    #print(scores.shape)
#    avgs = np.mean(scores,axis=1) # mean across all voxels for each fold
#    stds = np.std(scores,axis=1) # std across all voxels for each fold
    thresh = 0.05
    # mean and std across all voxels where R > threshold for each fold
    avgs = [np.mean(scores_fold_i[scores_fold_i>thresh]) for scores_fold_i in scores]
    stds = [np.std(scores_fold_i[scores_fold_i>thresh]) for scores_fold_i in scores]
    plt.errorbar(np.arange(len(avgs))+offset,avgs,yerr=stds,linestyle='None',
                 marker='o',label=cond)
    
def plot_avg_score_per_fold(scores_path,mask_path,save_path,hist_save=None):
    print('Plotting avg score per fold',save_path)
    plot_avg_score(scores_path,mask_path)
    plt.xlabel('Cross-validation fold')
    plt.ylabel('R averaged over voxels where R>0.05')
    plt.gcf().tight_layout()
#    plt.show()
    plt.savefig(save_path)
    plt.close()
    
#    if hist_save is not None:
#        fig, axes = plt.subplots(2,3)
#        for i, ax in enumerate(axes.reshape(-1)):
#            plt.sca(ax)
#            ax.set_xlim(0,0.2)
#            ax.set_ylim(0,1000)
#            plt.hist(r2[i])
#            plt.title('Fold %d' % i)
#            if i>2:
#                plt.xlabel('R2')
#            if i%3==0:
#                plt.ylabel('Count')
#        plt.gcf().tight_layout()
#    #    plt.show()
#        plt.savefig(hist_save)
#        plt.close()
        
def plot_avg_score_all_conditions(scores_paths,mask_paths,save_path):
    print('Plotting avg score all conditions',save_path)
    conds = ['CS','N4','S2']
    offsets = [-0.1,0.0,0.1]
    for scores_path,mask_path,cond,offset in zip(scores_paths,mask_paths,conds,offsets):
        plot_avg_score(scores_path,mask_path,cond,offset)
    plt.xlabel('Cross-validation fold')
    plt.ylabel('R averaged over voxels where R>0.05')
    plt.legend(loc='upper left')
    plt.gcf().tight_layout()
#    plt.show()
    plt.savefig(save_path)
    plt.close()
    
def plot_avg_score_all_conditions_plus_diffs(scores_paths,mask_paths,save_path):
    print('Plotting avg score all conditions plus diffs',save_path)
    conds = ['CS-CS','N4-N4','S2-S2','CS-N4','CS-S2']
    offsets = [-0.2,-0.1,0.0,0.1,0.2]
    for scores_path,mask_path,cond,offset in zip(scores_paths,mask_paths,conds,offsets):
        plot_avg_score(scores_path,mask_path,cond,offset)
    plt.xlabel('Cross-validation fold')
    plt.ylabel('R averaged over voxels where R>0.05')
    plt.legend(loc='upper left')
    plt.gcf().tight_layout()
#    plt.show()
    plt.savefig(save_path)
    plt.close()
    
def gather_files_and_plot_avg_scores():
    output_dirs = [
                   OUTPUT_BASE+'heschl/offset0/allstim/',
                   OUTPUT_BASE+'heschl/offset0/alwaysCS/',
                   ]
    conditions = ['CS','N4','S2']
    subjects = ['03','09']
    for sub in subjects:
        scores_paths = []
        mask_paths = []
        save_path = None
        first_CS = True
        for output_dir in output_dirs:
            sub_dir = os.path.join(output_dir,f'sub-{sub}/')
            for acq in conditions:
                if acq=='CS':
                    if first_CS:
                        first_CS = False
                    else:
                        continue
                acq_dir = os.path.join(sub_dir,f'acq-{acq}/')
                bids_str = acq_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                scores_path = bids_str + 'scores.nii.gz'
                mask_path = bids_str + 'masks.pkl'
                scores_paths.append(scores_path)
                mask_paths.append(mask_path)
                if not save_path:
                    save_path = bids_str + 'scoresall.png'
        plot_avg_score_all_conditions_plus_diffs(scores_paths,mask_paths,save_path)
    
def get_arg_highscore(scores_path):
    import nibabel as nib
    n1_img = nib.load(scores_path)
    data = n1_img.dataobj[...,0] # Only use first cv fold
    arg_highscore = np.argmax(data)
    return np.unravel_index(arg_highscore,data.shape)
    
def product_moment_corr(x,y):
    '''Product-moment correlation for two ndarrays x, y'''
    n = x.shape[0]
    r = (1/(n-1))*(x*y).sum(axis=0)
    return r

def plot_highest_score_bold_predicted_vs_actual(path_predicted,path_actual,arg_highscore,save_path):
    print('Plotting',save_path)
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.stats import zscore
    
    bold_predicted = nib.load(path_predicted)
    bold = nib.load(path_actual)
    #print(bold_predicted.shape)
    #print(bold.shape)
    # bold.shape[3] == 1038
    # for some reason loading a 1.4GB file with get_fdata() takes forever and
    # uses up hundreds of GB ?!?
#    start = 50
#    shortest = 900 #min(bold.shape[3],bold_predicted.shape[3])
#    offset = -11 # 11 * 0.85s = 9.35s = 5.1s lag + 4.25s offset
#    start = 0
    shortest = 1000
#    offset = 0
    x,y,z = arg_highscore
    bold_predicted_high = bold_predicted.dataobj[x,y,z,:shortest]
#    bold_predicted_high = bold_predicted.dataobj[x,y,z,start+offset:start+offset+shortest]
    bold_predicted_high = zscore(bold_predicted_high)
    bold_high = bold.dataobj[x,y,z,:shortest]
#    bold_high = bold.dataobj[x,y,z,start:start+shortest]
    bold_high = zscore(bold_high)
    plt.plot(bold_predicted_high,label='Predicted bold')
    plt.plot(bold_high,label='Actual bold')
    plt.legend()
    plt.xlabel('Volume')
    plt.ylabel('Zscore')
    plt.title('Predicted vs actual bold, voxel %s, r=%.4f' % (str(arg_highscore),\
              product_moment_corr(bold_predicted_high,bold_high)))
    plt.savefig(save_path)
#    plt.show()
    plt.close()
    #print(product_moment_corr(bold_predicted_high,bold_high))

def plot_original_bold_spectrograms():
    import nibabel as nib
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram, periodogram, welch, hilbert
    from scipy.stats import zscore
#    from nilearn.signal import clean
#    subjects = ['03']
#    conditions = ['CS']
    subjects = ['03','09']
    conditions = ['CS','N4','S2']
    for sub in subjects:
        for acq in conditions:
            bold_base = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                +f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/sub-{sub}_ses-{acq}'\
                +f'_task-aomovie_acq-{acq}_run-2_space-MNI152NLin2009cAsym_res-'\
                +'2_desc-preproc_bold'
            bold_path = bold_base + '.nii.gz'
            bold_save = bold_base + '/'
            print('Plotting',bold_path)
            if not os.path.exists(bold_save):
                os.mkdir(bold_save)
                
            bold = nib.load(bold_path)
            print(bold.shape)
            voxel = (70,57,40)
            TR = 0.85
            bold = bold.dataobj[70,57,40]
#            bold = clean(bold.reshape((-1,1)),low_pass=0.1,high_pass=0.01,t_r=TR).squeeze()
            bold = zscore(bold)
            # Envelope is the magnitude of the analytic signal computed by the 
            # hilbert transform
            envelope = np.abs(hilbert(bold))
#            n_volumes = 100
#            bold = bold[:n_volumes]
#            envelope = envelope[:n_volumes]
            plt.plot(bold)
            plt.xlabel('Volume')
            plt.ylabel('Bold (z-scored)')
            plt.title('Timeseries, voxel %s' % str(voxel))
            plt.savefig(bold_save+'bold_timeseries.png')
#            plt.show()
            plt.close()
            
            f, t, Sxx = spectrogram(bold, fs=1/TR, nperseg=64)
            plt.pcolormesh(t, f, Sxx, shading='gourad')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram, voxel %s' % str(voxel))
            plt.savefig(bold_save+'bold_spectrogram_FFT.png')
#            plt.show()
            plt.close()
            
            f, Pxx = periodogram(bold, fs=1/TR)
            plt.semilogy(f[1:],Pxx[1:])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Periodogram, voxel %s' % str(voxel))
            plt.savefig(bold_save+'bold_periodogram.png')
#            plt.show()
            plt.close()
            
            f, Pxx = welch(bold, fs=1/TR)
            plt.semilogy(f,Pxx)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Welch PSD, voxel %s' % str(voxel))
            plt.savefig(bold_save+'bold_periodogram_welch.png')
#            plt.show()
            plt.close()
            
            plt.plot(bold,label='Bold')
            plt.plot(envelope,label='Envelope')
#            plt.plot(-envelope)
#            plt.ylim(8600,8700)
            plt.xlabel('Volume')
            plt.ylabel('Bold (z-scored)')
            plt.legend()
            plt.title('Envelope, voxel %s' % str(voxel))
            plt.savefig(bold_save+'envelope_timeseries.png')
#            plt.show()
            plt.close()
            
            f, t, Sxx = spectrogram(envelope, fs=1/TR, nperseg=64)
            plt.pcolormesh(t, f, Sxx, shading='gourad')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Envelope spectrogram, voxel %s' % str(voxel))
            plt.savefig(bold_save+'envelope_spectrogram_FFT.png')
#            plt.show()
            plt.close()
            
            f, Pxx = periodogram(envelope, fs=1/TR)
            plt.semilogy(f[1:],Pxx[1:])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Envelope periodogram, voxel %s' % str(voxel))
            plt.savefig(bold_save+'envelope_periodogram.png')
#            plt.show()
            plt.close()
            
            f, Pxx = welch(envelope, fs=1/TR)
            plt.semilogy(f,Pxx)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Envelope Welch PSD, voxel %s' % str(voxel))
            plt.savefig(bold_save+'envelope_periodogram_welch.png')
#            plt.show()
            plt.close()
            
    
#nmel = 1
nmel = 48
def get_max_coefs(ridges_path,mask_path,arg_highscore):
    import joblib
    from nilearn.masking import unmask
    ridges = joblib.load(ridges_path)
    mask = joblib.load(mask_path)
    coefs = unmask(ridges.coef_.T,mask[0])
    #print(ridges.coef_.shape)
    #print(coefs.shape)
    max_coefs = coefs._dataobj[arg_highscore]
#    nmel = 48
    return max_coefs.reshape((-1,nmel))
    
def plot_max_coefs(max_coefs,save_path):
    print('Plotting',save_path)
    import matplotlib.pyplot as plt
#    nmel = 48
    nmel_stride = 4
    mel_frequencies = [100, 162, 224, 286, 348, 410, 472, 534, 596, 658, 721, 
                       783, 845, 907, 969, 1032, 1100, 1173, 1251, 1333, 1421,
                       1515, 1616, 1722, 1836, 1957, 2087, 2225, 2372, 2528, 
                       2696, 2874, 3064, 3266, 3482, 3712, 3957, 4219, 4497, 
                       4795, 5112, 5449, 5809, 6193, 6603, 7039, 7504, 8000]
    n_lag_bins = max_coefs.shape[0]
#    lagging_offset = 4.25
#    lagging_offset = 0.0
    lagging_offset = 4.0
    x_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
    x_stride = 20
#    x_stride = 80
#    plt.plot(max_coefs)
    plt.imshow(max_coefs.T,aspect='auto',origin='lower')
    plt.gca().set_yticks(np.arange(0,nmel,nmel_stride))
    plt.gca().set_yticklabels(mel_frequencies[::nmel_stride])
    plt.gca().set_xticks(np.arange(3,n_lag_bins,x_stride))
    plt.gca().set_xticklabels(["%.1f"% tick for tick in x_ticks[3::x_stride]])
    cbar = plt.colorbar()
    cbar.set_label('Ridge coefficient', rotation=270)
    plt.xlabel('Time lag (s)')
    plt.ylabel('Mel frequency (Hz)')
#    plt.ylabel('Ridge coefficient')
    plt.savefig(save_path)
    plt.close()

def plot_max_coef_diff(max_coefs,max_coefs_CS,save_path):
#    from scipy.stats import zscore
#    diff = zscore(max_coefs,axis=1)-zscore(max_coefs_CS,axis=1)
    diff = max_coefs - max_coefs_CS
    plot_max_coefs(diff,save_path)

def plot_mel_spectrogram(spec_path,save_path):
    print('Plotting',save_path)
    import matplotlib.pyplot as plt
#    nmel = 48
    nmel_stride = 4
    mel_frequencies = [100, 162, 224, 286, 348, 410, 472, 534, 596, 658, 721, 
                       783, 845, 907, 969, 1032, 1100, 1173, 1251, 1333, 1421,
                       1515, 1616, 1722, 1836, 1957, 2087, 2225, 2372, 2528, 
                       2696, 2874, 3064, 3266, 3482, 3712, 3957, 4219, 4497, 
                       4795, 5112, 5449, 5809, 6193, 6603, 7039, 7504, 8000]
    x = np.loadtxt(spec_path, delimiter='\t')
    #print(x.shape)
    x_len = x.shape[0]
    x = x[:int(x_len/20.78)]
    #print(x.shape)
    window_length = 0.025
    x_ticks = (np.arange(0,x.shape[0])*window_length).astype(int)
    x_stride = round(10/window_length)
    plt.imshow(x.T,aspect='auto',origin='lower')
    plt.gca().set_xticks(np.arange(0,x.shape[0],x_stride))
    plt.gca().set_xticklabels(x_ticks[::x_stride])
    plt.gca().set_yticks(np.arange(0,nmel,nmel_stride))
    plt.gca().set_yticklabels(mel_frequencies[::nmel_stride])
#    cbar = plt.colorbar()
#    cbar.set_label('Ridge coefficient', rotation=270)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel frequency (Hz)')
    plt.savefig(save_path)
#    plt.show()
    plt.close()

def plot_lagged_stimulus_spectrograms(spec_path,save_path):
    import matplotlib.pyplot as plt
    import joblib
    print('Plotting',save_path)
    spectrogram = joblib.load(spec_path)
#    x = joblib.load(spec_path)
    #print(spectrogram.shape)
    # x: ~30000-40000 time bins from original mel spectrogram with windowlength 0.025s
    # and overlap 0 are resampled to ~1000 fMRI volumes
    # y: 48 mel frequencies are repeated 204 times, 48 * 204 = 9792
    # stimulus lag is set to 6s * 0.85tr = 5.1s
    # lagging stimulus by 5.1s with 0.025s bins produces 204 lag bins, 5.1/0.025 = 204
    # Shift in the 204 lag bins is tiny (25ms) in comparison to stimulus tr on x-axis (850ms).
    # The coarse sampling rate on x axis prevents the tiny lag shift from being visible.
    # Have to shift 34 times (850ms/25ms) to move one bin to the right on x axis.
    # The original stimulus has 25ms windowlength so the lagging would be visible in
    # that space, but the resampling to ~1000 fMRI volumes (850ms windowlength)
    # makes it impossible to see the lagging when comparing neighboring lag bins directly.
    
    # Select every lag bin that moves the signal one bin on the x-axis, so we 
    # can actually see the lagged stimulus move:
    n_lags = 7
    offset = int(spectrogram.shape[1]/n_lags)
#    nmel = 48
    fMRI_volumes_to_plot = 50
    x = np.hstack([spectrogram[:fMRI_volumes_to_plot,(i*offset):((i*offset)+nmel)] for i in range(n_lags)])
    #print(x.shape)
#    x = x[:50]
    TR = 0.85
    x_ticks = (np.arange(0,x.shape[0])*TR).astype(int)
    x_stride = round(10/TR)
#    lagging_offset = 4.25
#    lagging_offset = 0.0
    lagging_offset = 4.0
    n_lag_bins = x.shape[1]
#    y_ticks = (np.arange(-n_lag_bins, 0) * 0.025) - lagging_offset
    y_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
    y_stride = 20
#    y_stride = 80
    plt.imshow(x.T,aspect='auto',origin='lower')
    plt.gca().set_xticks(np.arange(0,x.shape[0],x_stride))
    plt.gca().set_xticklabels(x_ticks[::x_stride])
    plt.gca().set_yticks(np.arange(0,n_lag_bins,y_stride))
    plt.gca().set_yticklabels(["%.1f"% tick for tick in y_ticks[0::y_stride]])
    #cbar = plt.colorbar()
    #cbar.set_label('Ridge coefficient', rotation=270)
    plt.xlabel('Time (s)')
#    plt.ylabel('Mel frequencies')
    plt.ylabel('Lagged features (mel frequencies * every 34th lag bin)')
#    plt.ylabel('Lag (s)')
    plt.savefig(save_path)
#    plt.show()
    plt.close()
    
def plot_lagged_stimulus_one_column(spec_path,save_path):
    import matplotlib.pyplot as plt
    import joblib
    from scipy.signal import welch
    print('Plotting',save_path)
    X = joblib.load(spec_path)
#    print(X.shape)
#    plt.plot(X[50])
    f, Pxx = welch(X[20], fs=40)
    plt.semilogy(f,Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V²/Hz)')
    max_freq = f[np.argmax(Pxx)]
    plt.axvline(max_freq,color='r')
    plt.title(f'Volume 20, max freq: {max_freq}')
    plt.savefig(save_path)
#    plt.show()
    plt.close()
    
if __name__=='__main__':
    import time
    tic = time.time()
#    gather_files_and_plot_avg_scores()
#    plot_original_bold_spectrograms()
    output_dirs = [
#                   OUTPUT_BASE+'temporal_lobe_mask/masked/',
#                   OUTPUT_BASE+'temporal_lobe_mask/masked_alwaysCS/',
#                   OUTPUT_BASE+'temporal_lobe_mask/masked_alwaysCS_removeconfounds/',
#                   OUTPUT_BASE+'temporal_lobe_mask/masked_noise/',
#                   OUTPUT_BASE+'heschl/offset0/allstim/',
#                   OUTPUT_BASE+'heschl/offset0/alwaysCS/',
#                   OUTPUT_BASE+'heschl/offset0/removeconfounds/',
#                   OUTPUT_BASE+'heschl/offset4.25_bandpass0.01-0.1_removeconfounds/nosmoothing/',
#                   OUTPUT_BASE+'heschl/offset4.25_bandpass0.01-0.1_removeconfounds/smoothing/'
#                   OUTPUT_BASE+'temporal_lobe_mask/offset4.25_bandpass0.01-0.1_removeconfounds/nosmoothing/',
#                   OUTPUT_BASE+'temporal_lobe_mask/offset4.25_bandpass0.01-0.1_removeconfounds/smoothing/'
#                   OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_envelope_MNI/',
#                   OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_envelope_T1w/',
                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_MNI/',
#                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_T1w/',
#                   OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_melspec_MNI/',
#                   OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_melspec_T1w/'
                   ]
    subjects = ['10']
    conditions = ['CS','N4','S2']
    runs = ["02","03","04","05","06","07"]
#    subjects = ['10']
#    conditions = ['CS']
#    runs = ["02"]
    do_scores = False
    do_bold_predicted_vs_actual = False
    do_max_coefs = False
    do_avg_scores_per_fold = False
    do_lagged_stim = False
    do_orig_stim = False
    do_lagged_stim_one_column = True
    for output_dir in output_dirs:
        for sub in subjects:
            sub_dir = os.path.join(output_dir,f'sub-{sub}/')
            if not os.path.exists(sub_dir):
                continue
            max_coefs_CS = None
            arg_highscore_CS = None
            for acq in conditions:
                acq_dir = os.path.join(sub_dir,f'acq-{acq}/')
                if not os.path.exists(acq_dir):
                    continue
                bids_str = acq_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                scores_path = bids_str + 'scores.nii.gz'
                if do_scores:
                    scores_save = bids_str + 'scores.png'
                    scores_glassbrain_save = bids_str + 'scoresglassbrain.png'
                    plot_scores(scores_path,scores_save,scores_glassbrain_save)
                if do_bold_predicted_vs_actual or do_max_coefs:
                    arg_highscore = get_arg_highscore(scores_path)
                if do_bold_predicted_vs_actual:
                    bold_predicted = bids_str + 'boldprediction.nii.gz'
                    bold_save = bids_str + 'boldprediction.png'
#                    bold_actual = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
#                        +f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/sub-{sub}_ses-{acq}'\
#                        +f'_task-aomovie_acq-{acq}_run-2_space-MNI152NLin2009cAsym_res-'\
#                        +'2_desc-preproc_bold.nii.gz'
                    bold_actual = bids_str + 'boldpreprocessed.nii.gz'
                    plot_highest_score_bold_predicted_vs_actual(bold_predicted,bold_actual,
                                                                arg_highscore,bold_save)
                if do_max_coefs:
                    ridges_path = bids_str + 'ridgesfold0.pkl'
                    mask_path = bids_str + 'masks.pkl'
                    max_coefs_save = bids_str + 'maxcoefs.png'
                    max_coefs_diff_save = bids_str + 'maxcoefsdiff.png'
                    max_coefs = get_max_coefs(ridges_path,mask_path,arg_highscore)
                    plot_max_coefs(max_coefs,max_coefs_save)
                    if acq=='CS':
                        max_coefs_CS = max_coefs
                        arg_highscore_CS = arg_highscore
                    elif max_coefs_CS is not None:
                        max_coefs = get_max_coefs(ridges_path,mask_path,arg_highscore_CS)
                        plot_max_coef_diff(max_coefs,max_coefs_CS,max_coefs_diff_save)
                if do_avg_scores_per_fold:
                    mask_path = bids_str + 'masks.pkl'
                    r2_save = bids_str + 'scoresavg.png'
#                    hist_save = bids_str + 'r2hist.png'
                    plot_avg_score_per_fold(scores_path,mask_path,r2_save)
                    if acq=='CS': 
                        allcond_save = bids_str + 'scoresallcond.png'
                        N4_str = os.path.join(sub_dir,f'acq-N4/sub-{sub}_task-aomovie_acq-N4_desc-')
                        N4scores = N4_str + 'scores.nii.gz'
                        N4mask = N4_str + 'masks.pkl'
                        S2_str = os.path.join(sub_dir,f'acq-S2/sub-{sub}_task-aomovie_acq-S2_desc-')
                        S2scores = N4_str + 'scores.nii.gz'
                        S2mask = N4_str + 'masks.pkl'
                        if os.path.exists(N4scores) and os.path.exists(S2scores):
                            plot_avg_score_all_conditions([scores_path,N4scores,S2scores],
                                                          [mask_path,N4mask,S2mask],
                                                          allcond_save)
                lagged_stim_dir = os.path.join(acq_dir,'lagged_stim/')
                if not os.path.exists(lagged_stim_dir):
                    os.mkdirs(lagged_stim_dir)
                if do_orig_stim:
                    stim_path = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                        +f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/'
                    for run in runs:
                        stim_fn = f'sub-{sub}_ses-{acq}_task-aomovie_acq-{acq}_run-{run}_recording-audio_stim.tsv.gz'
                        save_path = lagged_stim_dir + stim_fn.split('.')[0] + '.png'
                        plot_mel_spectrogram(stim_path+stim_fn,save_path)
                if do_lagged_stim:
                    bids_str_lagged = lagged_stim_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                    lagged_stim_path = bids_str_lagged + 'laggedstim{0}.pkl'
                    lagged_stim_save = bids_str_lagged + 'laggedstim{0}.png'
                    lagged_stim_all_path = bids_str_lagged + 'laggedstimall.pkl'
                    lagged_stim_all_save = bids_str_lagged + 'laggedstimall.png'
                    for i in range(len(runs)):
                        if not os.path.exists(lagged_stim_path.format(i)):
                            continue
                        plot_lagged_stimulus_spectrograms(lagged_stim_path.format(i),
                                                          lagged_stim_save.format(i))
#                    if not os.path.exists(lagged_stim_all_path):
#                        continue
#                    plot_lagged_stimulus_spectrograms(lagged_stim_all_path,
#                                                      lagged_stim_all_save)       
                if do_lagged_stim_one_column:
                    bids_str_lagged = lagged_stim_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                    lagged_stim_path = bids_str_lagged + 'laggedstim{0}.pkl'
                    lagged_stim_save = bids_str_lagged + 'laggedstimonecol{0}.png'
                    for i in range(len(runs)):
                        if not os.path.exists(lagged_stim_path.format(i)):
                            continue
                        plot_lagged_stimulus_one_column(lagged_stim_path.format(i),
                                                          lagged_stim_save.format(i))              
    toc = time.time()
    print('\nElapsed time: {:.2f} s'.format(toc - tic))