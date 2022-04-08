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
import nibabel as nib
from sklearn.decomposition import PCA
from scipy.stats import zscore
from nilearn.image import load_img, mean_img, coord_transform, resample_img
from nilearn.masking import apply_mask, unmask, intersect_masks
from encoding import product_moment_corr

OUTPUT_BASE = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/' \
              + 'derivatives/encoding_results/'
temporal_lobe_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/" \
                     + "derivatives/fmriprep/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz"
heschl_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/" \
              + "derivatives/fmriprep/ROIs/HeschisGyrus/mni_Heschl_ROI.nii.gz"


def plot_scores(scores_path, save_path, glassbrain_save, mask_path, select_voxels_by_pval=False, perm_path=None,
                roi_mask=None):
    print('Plotting', save_path)
    from nilearn import plotting
    mask = joblib.load(mask_path)[0]
    mean_scores = mean_img(scores_path)
    img = load_img(scores_path)
    if roi_mask:
        roi_mask = resample_img(roi_mask, mask._affine, mask.shape, interpolation='nearest')
        roi_mask = intersect_masks([mask, roi_mask])
    if select_voxels_by_pval:
        perm_all = joblib.load(perm_path.format('s'))
        pval = joblib.load(perm_path.format('pval'))
        selection_mask = mask
        if roi_mask:
            pval = unmask(pval, mask)
            pval = apply_mask(pval, roi_mask)
            selection_mask = roi_mask
        n_permutations = perm_all.shape[1]
        thresh = 1.0 / n_permutations
        pval_mask = pval > thresh

        mean_scores = apply_mask(mean_scores, selection_mask)
        mean_scores[pval_mask] = 0
        mean_scores = unmask(mean_scores, selection_mask)
        img = apply_mask(img, selection_mask)
        img[np.broadcast_to(pval_mask, (img.shape[0], pval_mask.shape[0]))] = 0
        img = unmask(img, selection_mask)
    elif roi_mask:
        mean_scores = apply_mask(mean_scores, roi_mask)
        mean_scores = unmask(mean_scores, roi_mask)
        img = apply_mask(img, roi_mask)
        img = unmask(img, roi_mask)

    data = img.dataobj
    avg_max = np.max(mean_scores.dataobj)
    avg_argmax = np.unravel_index(np.argmax(mean_scores.dataobj), mean_scores.shape)
    total_max = np.max(data)
    total_argmax = np.unravel_index(np.argmax(data), data.shape)
    fold0_max = np.max(data[..., 0])
    fold0_argmax = np.unravel_index(np.argmax(data[..., 0]), data.shape[:-1])
    avg_argmax_mni = coord_transform(*avg_argmax, mask.affine)
    fold0_argmax_mni = coord_transform(*fold0_argmax, mask.affine)
    total_argmax_mni = coord_transform(*total_argmax[:3], mask.affine)

    def _plot_helper(scores, markers, save_suffix, title):
        thresh = 0.0  # 0.05
        display = plotting.plot_stat_map(scores, threshold=thresh, title=title)
        display.add_markers([markers], marker_color='w', marker_size=50)
        #    display.add_contours(temporal_lobe_mask,filled=False,colors='m')
        #    display.add_contours(heschl_mask,filled=False,colors='g')
        plt.gcf().savefig(save_path + save_suffix)
        plt.close()
        display = plotting.plot_glass_brain(scores, threshold=thresh, colorbar=True,
                                            display_mode='lzry', plot_abs=False)
        display.add_contours(temporal_lobe_mask, filled=False, colors='m')
        display.add_contours(heschl_mask, filled=False, colors='g')
        display.add_markers([markers], marker_color='w', marker_size=50)
        # proxy artist trick to make legend
        from matplotlib.patches import Rectangle
        cont1 = Rectangle((0, 0), 1, 1, fc="magenta")
        cont2 = Rectangle((0, 0), 1, 1, fc="green")
        plt.legend([cont1, cont2], ['Temporal lobe', 'Heschl gyrus'])
        plt.gcf().savefig(glassbrain_save + save_suffix)
        plt.close()

    title = 'Avg %.3f %s' % (avg_max, str(avg_argmax_mni))
    _plot_helper(mean_scores, avg_argmax_mni, '_avg.png', title)
    title = 'Fold 0 %.3f %s' % (fold0_max, str(fold0_argmax_mni))
    _plot_helper(img.slicer[..., 0], fold0_argmax_mni, '_fold0.png', title)
    title = 'Total %.3f %s fold %d' % (total_max, str(total_argmax_mni), total_argmax[3])
    _plot_helper(img.slicer[..., total_argmax[3]], total_argmax_mni, '_total.png', title)


def plot_avg_score(scores_path, mask_path, cond=None, offset=0):
    scores = load_img(scores_path)
    mask = joblib.load(mask_path)
    scores = apply_mask(scores, mask[0])
    #    scores = np.square(scores) # R to R^2
    # print(scores.shape)
    #    avgs = np.mean(scores,axis=1) # mean across all voxels for each fold
    #    stds = np.std(scores,axis=1) # std across all voxels for each fold
    thresh = 0.05
    # mean and std across all voxels where R > threshold for each fold
    avgs = [np.mean(scores_fold_i[scores_fold_i > thresh]) for scores_fold_i in scores]
    stds = [np.std(scores_fold_i[scores_fold_i > thresh]) for scores_fold_i in scores]
    plt.errorbar(np.arange(len(avgs)) + offset, avgs, yerr=stds, linestyle='None',
                 marker='o', label=cond)


def plot_avg_score_per_fold(scores_path, mask_path, save_path, hist_save=None):
    print('Plotting avg score per fold', save_path)
    plot_avg_score(scores_path, mask_path)
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


def plot_avg_score_all_conditions(scores_paths, mask_paths, save_path):
    print('Plotting avg score all conditions', save_path)
    conds = ['CS', 'N4', 'S2']
    offsets = [-0.1, 0.0, 0.1]
    for scores_path, mask_path, cond, offset in zip(scores_paths, mask_paths, conds, offsets):
        plot_avg_score(scores_path, mask_path, cond, offset)
    plt.xlabel('Cross-validation fold')
    plt.ylabel('R averaged over voxels where R>0.05')
    plt.legend(loc='upper left')
    plt.gcf().tight_layout()
    #    plt.show()
    plt.savefig(save_path)
    plt.close()


def plot_avg_score_all_conditions_plus_diffs(scores_paths, mask_paths, save_path):
    print('Plotting avg score all conditions plus diffs', save_path)
    conds = ['CS-CS', 'N4-N4', 'S2-S2', 'CS-N4', 'CS-S2']
    offsets = [-0.2, -0.1, 0.0, 0.1, 0.2]
    for scores_path, mask_path, cond, offset in zip(scores_paths, mask_paths, conds, offsets):
        plot_avg_score(scores_path, mask_path, cond, offset)
    plt.xlabel('Cross-validation fold')
    plt.ylabel('R averaged over voxels where R>0.05')
    plt.legend(loc='upper left')
    plt.gcf().tight_layout()
    #    plt.show()
    plt.savefig(save_path)
    plt.close()


def gather_files_and_plot_avg_scores():
    output_dirs = [
        OUTPUT_BASE + 'heschl/offset0/allstim/',
        OUTPUT_BASE + 'heschl/offset0/alwaysCS/',
    ]
    conditions = ['CS', 'N4', 'S2']
    subjects = ['03', '09']
    for sub in subjects:
        scores_paths = []
        mask_paths = []
        save_path = None
        first_CS = True
        for output_dir in output_dirs:
            sub_dir = os.path.join(output_dir, f'sub-{sub}/')
            for acq in conditions:
                if acq == 'CS':
                    if first_CS:
                        first_CS = False
                    else:
                        continue
                acq_dir = os.path.join(sub_dir, f'acq-{acq}/')
                bids_str = acq_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                scores_path = bids_str + 'scores.nii.gz'
                mask_path = bids_str + 'masks.pkl'
                scores_paths.append(scores_path)
                mask_paths.append(mask_path)
                if not save_path:
                    save_path = bids_str + 'scoresall.png'
        plot_avg_score_all_conditions_plus_diffs(scores_paths, mask_paths, save_path)


def get_arg_highscore(scores_path, fold=0):
    n1_img = nib.load(scores_path)
    data = n1_img.dataobj[..., fold]
    arg_highscore = np.argmax(data)
    return np.unravel_index(arg_highscore, data.shape)


def get_total_arg_highscore(scores_path):
    n1_img = nib.load(scores_path)
    data = n1_img.dataobj
    arg_highscore = np.argmax(data)
    return np.unravel_index(arg_highscore, data.shape)


def plot_highest_score_bold_predicted_vs_actual(path_predicted, path_actual, arg_highscore, save_path, mask_path):
    print('Plotting', save_path)
    from scipy.stats import zscore

    x, y, z, fold = arg_highscore
    # bold = []
    # for i in range(6):
    #     # bold.append(joblib.load(path_actual.format(i))[:-1])
    #     bold.append(joblib.load(path_actual.format(i)))
    # bold = np.concatenate(bold) # (time, voxel)
    bold = joblib.load(path_actual.format(fold))
    masks = joblib.load(mask_path)
    bold = unmask(bold, masks[fold])

    # bold_predicted = nib.load(path_predicted)
    # bold_predicted = []
    # for i in range(6):
    #     # bold_actual.append(joblib.load(path_predicted.format(i))[:-1])
    #     bold_predicted.append(joblib.load(path_predicted.format(i)))
    # bold_predicted = np.concatenate(bold_predicted)  # (time, voxel)
    bold_predicted = joblib.load(path_predicted.format(fold))
    bold_predicted = unmask(bold_predicted, masks[fold])
    # print(bold_predicted.shape)
    # print(bold.shape)
    bold_predicted_high = bold_predicted.dataobj[x, y, z]
    bold_predicted_high = zscore(bold_predicted_high)
    bold_high = bold.dataobj[x, y, z]
    bold_high = zscore(bold_high)
    plt.plot(bold_predicted_high, label='Predicted bold')
    plt.plot(bold_high, label='Actual bold')
    plt.legend()
    plt.xlabel('Volume')
    plt.ylabel('Zscore')
    r, prob = product_moment_corr(bold_predicted_high.reshape(-1, 1), bold_high.reshape(-1, 1))
    plt.title('Predicted vs actual bold, voxel %s, r=%.4f, p=%.4f' % (str(arg_highscore), r, prob))
    plt.savefig(save_path)
    #    plt.show()
    plt.close()
    # print(product_moment_corr(bold_predicted_high,bold_high))


def plot_first_pc_bold_predicted_vs_actual(path_predicted, path_actual, save_path, mask_path):
    print('Plotting', save_path)

    bold_actual = []
    for fold in range(6):
        filename = path_actual.format(fold)
        if not os.path.exists(filename):
            continue
        # bold_actual.append(joblib.load(path_actual.format(i))[:-1])
        bold_actual.append(joblib.load(filename))
    bold_actual = np.concatenate(bold_actual)  # (time, voxel)
    # print(bold.shape)
    bold_predicted = []
    for fold in range(6):
        filename = path_predicted.format(fold)
        if not os.path.exists(filename):
            continue
        # bold_actual.append(joblib.load(path_predicted.format(i))[:-1])
        bold_predicted.append(joblib.load(filename))
    bold_predicted = np.concatenate(bold_predicted)  # (time, voxel)
    # print(bold_predicted.shape)

    pca = PCA(n_components=1)
    bold_predicted_first_pc = zscore(pca.fit_transform(bold_predicted))
    # bold_actual_first_pc = zscore(pca.fit_transform(bold_actual))
    # bold_actual_first_pc = zscore(np.dot(bold_actual - bold_actual.mean(axis=0), pca.components_.T))
    bold_actual_first_pc = zscore(np.dot(bold_actual - pca.mean_, pca.components_.T))

    plt.plot(bold_predicted_first_pc, label='Predicted bold 1st PC')
    plt.plot(bold_actual_first_pc, label='Actual bold 1st PC')
    plt.legend()
    plt.xlabel('Volume')
    plt.ylabel('Zscore')
    r, prob = product_moment_corr(bold_predicted_first_pc, bold_actual_first_pc)
    plt.title('r=%.4f, p=%.4f, explained variance ratio=%.4f' % (r, prob, pca.explained_variance_ratio_))
    plt.savefig(save_path + '.svg')
    #    plt.show()
    plt.close()

    from nilearn import plotting
    mask = joblib.load(mask_path)
    first_pc = unmask(np.squeeze(pca.components_), mask[0])

    thresh = 0.0  # 0.05
    display = plotting.plot_glass_brain(first_pc, threshold=thresh, colorbar=True,
                                        display_mode='lzry', plot_abs=False)
    display.add_contours(temporal_lobe_mask, filled=False, colors='m')
    display.add_contours(heschl_mask, filled=False, colors='g')
    # display.add_markers([markers], marker_color='w', marker_size=50)
    # proxy artist trick to make legend
    from matplotlib.patches import Rectangle
    cont1 = Rectangle((0, 0), 1, 1, fc="magenta")
    cont2 = Rectangle((0, 0), 1, 1, fc="green")
    plt.legend([cont1, cont2], ['Temporal lobe', 'Heschl gyrus'])
    plt.gcf().savefig(save_path + 'glassbrain.svg')
    plt.close()

    n_components = 20
    pca = PCA(n_components=n_components)
    pca.fit(bold_predicted)

    plt.bar(np.arange(n_components), pca.explained_variance_ratio_)
    # plt.plot(pca.singular_values_, label='Singular values')
    # plt.legend()
    plt.xlabel('Principal component')
    plt.ylabel('Explained variance ratio')
    plt.savefig(save_path + 'explainedvariance.svg')
    #    plt.show()
    plt.close()


def plot_original_bold_spectrograms():
    from scipy.signal import spectrogram, periodogram, welch, hilbert
    from scipy.stats import zscore
    #    from nilearn.signal import clean
    #    subjects = ['03']
    #    conditions = ['CS']
    subjects = ['03', '09']
    conditions = ['CS', 'N4', 'S2']
    for sub in subjects:
        for acq in conditions:
            bold_base = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/' \
                        + f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/sub-{sub}_ses-{acq}' \
                        + f'_task-aomovie_acq-{acq}_run-2_space-MNI152NLin2009cAsym_res-' \
                        + '2_desc-preproc_bold'
            bold_path = bold_base + '.nii.gz'
            bold_save = bold_base + '/'
            print('Plotting', bold_path)
            if not os.path.exists(bold_save):
                os.mkdir(bold_save)

            bold = nib.load(bold_path)
            print(bold.shape)
            voxel = (70, 57, 40)
            TR = 0.85
            bold = bold.dataobj[70, 57, 40]
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
            plt.savefig(bold_save + 'bold_timeseries.png')
            #            plt.show()
            plt.close()

            f, t, Sxx = spectrogram(bold, fs=1 / TR, nperseg=64)
            plt.pcolormesh(t, f, Sxx, shading='gourad')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'bold_spectrogram_FFT.png')
            #            plt.show()
            plt.close()

            f, Pxx = periodogram(bold, fs=1 / TR)
            plt.semilogy(f[1:], Pxx[1:])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Periodogram, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'bold_periodogram.png')
            #            plt.show()
            plt.close()

            f, Pxx = welch(bold, fs=1 / TR)
            plt.semilogy(f, Pxx)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Welch PSD, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'bold_periodogram_welch.png')
            #            plt.show()
            plt.close()

            plt.plot(bold, label='Bold')
            plt.plot(envelope, label='Envelope')
            #            plt.plot(-envelope)
            #            plt.ylim(8600,8700)
            plt.xlabel('Volume')
            plt.ylabel('Bold (z-scored)')
            plt.legend()
            plt.title('Envelope, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'envelope_timeseries.png')
            #            plt.show()
            plt.close()

            f, t, Sxx = spectrogram(envelope, fs=1 / TR, nperseg=64)
            plt.pcolormesh(t, f, Sxx, shading='gourad')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Envelope spectrogram, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'envelope_spectrogram_FFT.png')
            #            plt.show()
            plt.close()

            f, Pxx = periodogram(envelope, fs=1 / TR)
            plt.semilogy(f[1:], Pxx[1:])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Envelope periodogram, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'envelope_periodogram.png')
            #            plt.show()
            plt.close()

            f, Pxx = welch(envelope, fs=1 / TR)
            plt.semilogy(f, Pxx)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (V²/Hz)')
            plt.title('Envelope Welch PSD, voxel %s' % str(voxel))
            plt.savefig(bold_save + 'envelope_periodogram_welch.png')
            #            plt.show()
            plt.close()


def plot_max_coefs(ridges_path, mask_path, arg_highscore, save_path):
    mask = joblib.load(mask_path)
    for fold in range(len(mask)):
        filename = ridges_path.format(fold)
        if not os.path.exists(filename):
            continue

        print('Plotting', save_path.format(fold))
        ridges = joblib.load(filename)
        # print(ridges.alpha_)
        coefs = unmask(ridges.coef_.T, mask[fold])
        max_coefs = coefs.dataobj[arg_highscore[:3]]
        if use_mel_spec_features:
            max_coefs = max_coefs.reshape((-1, n_mel))

        n_lag_bins = max_coefs.shape[0]
        #    lagging_offset = 4.25
        lagging_offset = 0.0
        #    lagging_offset = 4.0
        x_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
        # x_stride = 20
        x_stride = 80

        if use_mel_spec_features:
            n_mel_stride = 4
            plt.imshow(max_coefs.T, aspect='auto', origin='lower')
            plt.ylabel('Mel frequency (Hz)')
            plt.gca().set_yticks(np.arange(0, n_mel, n_mel_stride))
            plt.gca().set_yticklabels(mel_frequencies[:n_mel:n_mel_stride])
            cbar = plt.colorbar()
            cbar.set_label('Ridge coefficient', rotation=270)
        else:
            plt.plot(max_coefs)
            plt.ylabel('Ridge coefficient')

        plt.xlabel('Time lag (s)')
        plt.gca().set_xticks(np.arange(3, n_lag_bins, x_stride))
        plt.gca().set_xticklabels(["%.1f" % tick for tick in x_ticks[3::x_stride]])
        plt.tight_layout()
        plt.savefig(save_path.format(fold) + '.svg')
        plt.close()


def plot_coef_first_pc(save_path, ridges_path):
    print('Plotting', save_path)
    from sklearn.decomposition import PCA
    n_folds = 0
    coefficients = []
    for fold in range(6):
        filename = ridges_path.format(fold)
        if not os.path.exists(filename):
            continue
        ridges = joblib.load(filename)
        coefficients.append(ridges.coef_.T)
        n_folds += 1
    coefficients = np.stack(coefficients, axis=0)
    coefficients = np.mean(coefficients, axis=0)

    pca = PCA(n_components=1)
    coefficients_first_pc = pca.fit_transform(coefficients)
    # coefficients_first_pc = coefficients_first_pc.reshape(n_folds, -1).mean(axis=0)

    if use_mel_spec_features:
        coefficients_first_pc = coefficients_first_pc.reshape((-1, n_mel))
        n_mel_stride = 4
        n_lag_bins = coefficients_first_pc.shape[0]
        #    lagging_offset = 4.25
        lagging_offset = 0.0
        #    lagging_offset = 4.0
        x_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
        # x_stride = 20
        x_stride = 80
        plt.imshow(coefficients_first_pc.T, aspect='auto', origin='lower')
        plt.gca().set_yticks(np.arange(0, n_mel, n_mel_stride))
        plt.gca().set_yticklabels(mel_frequencies[:n_mel:n_mel_stride])
        plt.gca().set_xticks(np.arange(3, n_lag_bins, x_stride))
        plt.gca().set_xticklabels(["%.1f" % tick for tick in x_ticks[3::x_stride]])
        cbar = plt.colorbar()
        cbar.set_label('First PC of ridge coefficient', rotation=270)
        plt.xlabel('Time lag (s)')
        plt.ylabel('Mel frequency (Hz)')
        plt.tight_layout()
        plt.savefig(save_path + '.svg')
        plt.close()
    else:
        n_lag_bins = coefficients_first_pc.shape[0]
        #    lagging_offset = 4.25
        lagging_offset = 0.0
        #    lagging_offset = 4.0
        x_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
        #    x_stride = 20
        x_stride = 80
        plt.plot(coefficients_first_pc)
        plt.gca().set_xticks(np.arange(3, n_lag_bins, x_stride))
        plt.gca().set_xticklabels(["%.1f" % tick for tick in x_ticks[3::x_stride]])
        plt.xlabel('Time lag (s)')
        plt.ylabel('First PC of ridge coefficient')
        plt.savefig(save_path + '.svg')
        plt.close()


def plot_mel_spectrogram(spec_path, save_path):
    print('Plotting', save_path)
    nmel_stride = 4
    x = np.loadtxt(spec_path, delimiter='\t')
    # print(x.shape)
    x_len = x.shape[0]
    x = x[:int(x_len / 20.78)]
    # print(x.shape)
    window_length = 0.025
    x_ticks = (np.arange(0, x.shape[0]) * window_length).astype(int)
    x_stride = round(10 / window_length)
    plt.imshow(x.T, aspect='auto', origin='lower')
    plt.gca().set_xticks(np.arange(0, x.shape[0], x_stride))
    plt.gca().set_xticklabels(x_ticks[::x_stride])
    plt.gca().set_yticks(np.arange(0, n_mel, nmel_stride))
    plt.gca().set_yticklabels(mel_frequencies[::nmel_stride])
    #    cbar = plt.colorbar()
    #    cbar.set_label('Ridge coefficient', rotation=270)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel frequency (Hz)')
    plt.savefig(save_path)
    #    plt.show()
    plt.close()


def plot_lagged_stimulus_spectrograms(spec_path, save_path):
    print('Plotting', save_path)
    x = joblib.load(spec_path)

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
    # n_lags = 7
    # offset = int(x.shape[1]/n_lags)
    # fMRI_volumes_to_plot = 50
    # x = np.hstack([x[:fMRI_volumes_to_plot, (i*offset):((i*offset)+n_mel)] for i in range(n_lags)])
    # # print(x.shape)
    x = x[:50]
    TR = 0.85
    x_ticks = (np.arange(0, x.shape[0]) * TR).astype(int)
    x_stride = round(10 / TR)
    #    lagging_offset = 4.25
    lagging_offset = 0.0
    #    lagging_offset = 4.0
    n_lag_bins = x.shape[1]
    #    y_ticks = (np.arange(-n_lag_bins, 0) * 0.025) - lagging_offset
    y_ticks = (np.arange(n_lag_bins) * 0.025) + lagging_offset
    #    y_stride = 20
    y_stride = 800
    plt.figure(figsize=[32, 24])
    plt.imshow(x.T, aspect='auto', origin='lower')
    plt.gca().set_xticks(np.arange(0, x.shape[0], x_stride))
    plt.gca().set_xticklabels(x_ticks[::x_stride])
    plt.gca().set_yticks(np.arange(0, n_lag_bins, y_stride))
    plt.gca().set_yticklabels(["%.1f" % tick for tick in y_ticks[0::y_stride]])
    # cbar = plt.colorbar()
    # cbar.set_label('Ridge coefficient', rotation=270)
    plt.xlabel('Time (s)')
    #    plt.ylabel('Mel frequencies')
    #    plt.ylabel('Lagged features (mel frequencies * every 34th lag bin)')
    plt.ylabel('Lag (s)')
    plt.savefig(save_path + '.svg')
    #    plt.show()
    plt.close()


#    for lag in range(34):
#        points_to_plot = []
#        for volume in range(15):
#            points_to_plot.append(x[15+volume,lag+volume*34])
#        plt.plot(points_to_plot)
#    plt.xlabel('Lag (s)')
#    plt.savefig(save_path+'_slope.svg')
##    plt.show()
#    plt.close()


def plot_lagged_stimulus_one_column(spec_path, save_path):
    from scipy.signal import welch
    print('Plotting', save_path)
    X = joblib.load(spec_path)
    #    print(X.shape)
    #    plt.plot(X[50])
    f, Pxx = welch(X[20], fs=40)
    plt.semilogy(f, Pxx)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V²/Hz)')
    max_freq = f[np.argmax(Pxx)]
    plt.axvline(max_freq, color='r')
    plt.title(f'Volume 20, max freq: {max_freq}')
    plt.savefig(save_path)
    #    plt.show()
    plt.close()


def permutation_test_prep(perm_dir, save_permutation_path, scores_path, mask_path):
    # if os.path.exists(save_permutation_path.format('s')):
    #     return
    import fnmatch
    perm_fns = fnmatch.filter(os.listdir(perm_dir), '*permutation*.pkl')
    permutation_scores = []
    for fn in perm_fns:
        permutation_scores.append(joblib.load(perm_dir + fn))
    permutation_scores = np.stack(permutation_scores, axis=1)
    n_permutations = permutation_scores.shape[1]
    joblib.dump(permutation_scores, save_permutation_path.format('s'))

    scores = mean_img(scores_path)
    mask = joblib.load(mask_path)
    scores = apply_mask(scores, mask[0])
    # score = np.mean(score_list, axis=1).reshape(-1, 1)
    joblib.dump(scores, save_permutation_path.format('truescores'))
    scores = scores.reshape(-1, 1)
    pvalue = (np.sum(permutation_scores >= scores, axis=1) + 1.0) / (n_permutations + 1)
    joblib.dump(pvalue, save_permutation_path.format('pval'))


def permutation_test(perm_path, mask_path, save_path):
    print('Plotting', save_path)
    mask = joblib.load(mask_path)[0]
    scores = joblib.load(perm_path.format('truescores'))
    perm_all = joblib.load(perm_path.format('s'))
    arg_highscore = np.argmax(scores)
    # arg_highscore = 0
    pval_orig = joblib.load(perm_path.format('pval'))

    from nilearn import plotting
    from matplotlib.patches import Rectangle

    def _plot_pvals(pvals, save_suffix):
        display = plotting.plot_glass_brain(pvals, threshold=0.00, colorbar=True,
                                            display_mode='lzry', plot_abs=False)
        display.add_contours(temporal_lobe_mask, filled=False, colors='m')
        display.add_contours(heschl_mask, filled=False, colors='g')
        # proxy artist trick to make legend
        cont1 = Rectangle((0, 0), 1, 1, fc="magenta")
        cont2 = Rectangle((0, 0), 1, 1, fc="green")
        plt.legend([cont1, cont2], ['Temporal lobe', 'Heschl gyrus'])
        plt.savefig(save_path + save_suffix)
        plt.close()

    n_permutations = perm_all.shape[1]
    thresh = 1.0 / n_permutations
    pval = pval_orig.copy()
    pval[pval > thresh] = 0
    pvals = unmask(pval, mask)
    _plot_pvals(pvals, '.png')

    pval = pval_orig.copy()
    pval[pval < 1] = 0
    pvals = unmask(pval, mask)
    _plot_pvals(pvals, 'negativecorr.png')

    fig, ax = plt.subplots(1)
    ax.hist(perm_all[arg_highscore], bins=16)
    ax.axvline(scores[arg_highscore], color='r', linestyle='-', label='Observed mean')
    ax.set_xlabel('R score')
    ax.set_ylabel('Count (permutations)')
    plt.title(f'Highest correlation voxel, p={pval_orig[arg_highscore]:.3}')
    plt.legend()
    fig.tight_layout()
    fig.savefig(save_path + 'hist.svg', format='svg')
    plt.close()

    fig, ax = plt.subplots(1)
    ax.hist(pval_orig, bins=100)
    ax.set_xlabel('p-value')
    ax.set_ylabel('Count (Voxels)')
    # plt.title(f'Highest correlation voxel, p={pval_orig[arg_highscore]:.3}')
    # plt.legend()
    fig.tight_layout()
    fig.savefig(save_path + 'histpvals.svg', format='svg')
    plt.close()


if __name__ == '__main__':
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
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_MNI/',
        #                  OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_MNI_nopreprocessing/',
        # OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_MNI_noconfounds/',
        # OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_scannernoise/',
        # OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_T1w/',
        # OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_melspec_MNI/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging-4to-9.95_melspec_T1w/'
        #                  '/data2/fyilmaz/simulated_data/output_without_noise/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_MNI_nolowpass/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope4k/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope2k/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope1k/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_linearregression/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_lassocv/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope_SAGsolver/'
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_envelope80/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15envelopezband/',
        #                   OUTPUT_BASE+'temporal_lobe_mask/lagging0to-15.3_permutation_train_only/',
        # OUTPUT_BASE + 'temporal_lobe_mask/lagging0to-15melspec/',
        # OUTPUT_BASE + 'temporal_lobe_mask/lagging0to-15mps/',
        # OUTPUT_BASE + 'temporal_lobe_mask/lagging0to-15.3_test_alphas2/',
        # OUTPUT_BASE + 'temporal_lobe_mask/lagging0to-15.3_test_alpha_0.02/',
        OUTPUT_BASE + 'temporal_lobe_mask/lagging0to-15.3_test_alpha_1e6/',
    ]
    subjects = ['10']
    # subjects = ["04","05","06","07",'10']
    # conditions = ['CS']
    conditions = ['CS', 'N4', 'S2']
    runs = ["02", "03", "04", "05", "06", "07"]
    #     runs = ["02"]

    use_mel_spec_features = False
    n_mel = 32 if use_mel_spec_features else 1
     mel_frequencies = [100, 194, 288, 382, 476, 570, 664, 759, 853, 947, 1043,
                       1149, 1266, 1395, 1537, 1694, 1867, 2057, 2266, 2497, 2752,
                       3032, 3341, 3681, 4056, 4470, 4925, 5427, 5980, 6589, 7260, 8000]
      
    #mel_frequencies = [100, 162, 224, 286, 348, 410, 472, 534, 596, 658, 721,
                       #783, 845, 907, 969, 1032, 1100, 1173, 1251, 1333, 1421,
                       #1515, 1616, 1722, 1836, 1957, 2087, 2225, 2372, 2528,
                       #2696, 2874, 3064, 3266, 3482, 3712, 3957, 4219, 4497,
                      # 4795, 5112, 5449, 5809, 6193, 6603, 7039, 7504, 8000]
            
   

    select_voxels_by_pval = False  # Only works when folder has permutation test data.
    roi_mask = None
    # roi_mask = heschl_mask
    # roi_mask = temporal_lobe_mask

    do_scores = True
    do_bold_predicted_vs_actual = True
    do_max_coefs = True
    do_timeseries_first_pc = True
    do_coef_first_pc = True
    do_avg_scores_per_fold = False
    do_lagged_stim = False
    do_orig_stim = False
    do_lagged_stim_one_column = False
    do_perm_test_prep = False
    do_perm_test = False

    for output_dir in output_dirs:
        for sub in subjects:
            sub_dir = os.path.join(output_dir, f'sub-{sub}/')
            if not os.path.exists(sub_dir):
                continue
            max_coefs_CS = None
            arg_highscore_CS = None
            for acq in conditions:
                acq_dir = os.path.join(sub_dir, f'acq-{acq}/')
                if not os.path.exists(acq_dir):
                    continue
                bids_str = acq_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                scores_path = bids_str + 'scores.nii.gz'
                mask_path = bids_str + 'masks.pkl'
                save_permutation_path = None

                if do_perm_test_prep or select_voxels_by_pval:
                    perm_dir = acq_dir + 'permutation_test/'
                    save_permutation_path = bids_str + 'permutation{0}.pkl'
                    permutation_test_prep(perm_dir, save_permutation_path, scores_path, mask_path)
                if do_scores:
                    scores_save = bids_str + 'scores'
                    scores_glassbrain_save = bids_str + 'scoresglassbrain'
                    plot_scores(scores_path, scores_save, scores_glassbrain_save, mask_path, select_voxels_by_pval,
                                save_permutation_path, roi_mask)
                if do_bold_predicted_vs_actual or do_max_coefs or do_coef_first_pc:
                    # arg_highscore = get_arg_highscore(scores_path)
                    arg_highscore = get_total_arg_highscore(scores_path)
                if do_bold_predicted_vs_actual:
                    bold_save = bids_str + 'boldprediction.svg'
                    # bold_predicted = bids_str + 'boldprediction.nii.gz'
                    #                     bold_actual = bids_str + 'boldpreprocessed.nii.gz'
                    bold_predicted = acq_dir + f'predicted_bold/sub-{sub}_task-aomovie_acq-{acq}_desc-boldpredicted' + '{0}.pkl'
                    bold_actual = acq_dir + f'preprocessed_bold/sub-{sub}_task-aomovie_acq-{acq}_desc-boldpreprocessed' + '{0}.pkl'
                    plot_highest_score_bold_predicted_vs_actual(bold_predicted, bold_actual,
                                                                arg_highscore, bold_save, mask_path)
                if do_timeseries_first_pc:
                    bold_save = bids_str + '1stPC'
                    # bold_predicted = bids_str + 'boldprediction.nii.gz'
                    #                     bold_actual = bids_str + 'boldpreprocessed.nii.gz'
                    bold_predicted = acq_dir + f'predicted_bold/sub-{sub}_task-aomovie_acq-{acq}_desc-boldpredicted' + '{0}.pkl'
                    bold_actual = acq_dir + f'preprocessed_bold/sub-{sub}_task-aomovie_acq-{acq}_desc-boldpreprocessed' + '{0}.pkl'
                    plot_first_pc_bold_predicted_vs_actual(bold_predicted, bold_actual, bold_save, mask_path)
                if do_max_coefs:
                    ridges_path = bids_str + 'ridgesfold{0}.pkl'
                    max_coefs_save = bids_str + 'maxcoefsfold{0}'
                    plot_max_coefs(ridges_path, mask_path, arg_highscore, max_coefs_save)
                if do_coef_first_pc:
                    ridges_path = bids_str + 'ridgesfold{0}.pkl'
                    first_pc_save = bids_str + 'coeffirstpc'
                    plot_coef_first_pc(first_pc_save, ridges_path)
                if do_avg_scores_per_fold:
                    r2_save = bids_str + 'scoresavg.svg'
                    #                    hist_save = bids_str + 'r2hist.png'
                    plot_avg_score_per_fold(scores_path, mask_path, r2_save)
                    if acq == 'CS':
                        allcond_save = bids_str + 'scoresallcond.png'
                        N4_str = os.path.join(sub_dir, f'acq-N4/sub-{sub}_task-aomovie_acq-N4_desc-')
                        N4scores = N4_str + 'scores.nii.gz'
                        N4mask = N4_str + 'masks.pkl'
                        S2_str = os.path.join(sub_dir, f'acq-S2/sub-{sub}_task-aomovie_acq-S2_desc-')
                        S2scores = S2_str + 'scores.nii.gz'
                        S2mask = S2_str + 'masks.pkl'
                        if os.path.exists(N4scores) and os.path.exists(S2scores):
                            plot_avg_score_all_conditions([scores_path, N4scores, S2scores],
                                                          [mask_path, N4mask, S2mask],
                                                          allcond_save)
                lagged_stim_dir = os.path.join(acq_dir, 'lagged_stim/')
                if not os.path.exists(lagged_stim_dir):
                    os.makedirs(lagged_stim_dir)
                if do_orig_stim:
                    stim_path = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/' \
                                + f'derivatives/fmriprep/sub-{sub}/ses-{acq}/func/'
                    for run in runs:
                        stim_fn = f'sub-{sub}_ses-{acq}_task-aomovie_acq-{acq}_run-{run}_recording-audio_stim.tsv.gz'
                        save_path = lagged_stim_dir + stim_fn.split('.')[0] + '.svg'
                        plot_mel_spectrogram(stim_path + stim_fn, save_path)
                if do_lagged_stim:
                    bids_str_lagged = lagged_stim_dir + f'sub-{sub}_task-aomovie_acq-{acq}_desc-'
                    lagged_stim_path = bids_str_lagged + 'laggedstim{0}.pkl'
                    lagged_stim_save = bids_str_lagged + 'laggedstim{0}'
                    #                    lagged_stim_all_path = bids_str_lagged + 'laggedstimall.pkl'
                    #                    lagged_stim_all_save = bids_str_lagged + 'laggedstimall.svg'
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
                if do_perm_test:
                    perm_path = bids_str + 'permutation{0}.pkl'
                    save_path = bids_str + 'permutation'
                    permutation_test(perm_path, mask_path, save_path)
    toc = time.time()
    print('\nElapsed time: {:.2f} s'.format(toc - tic))
