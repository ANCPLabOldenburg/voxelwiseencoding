import numpy as np
import joblib
from nilearn.masking import unmask, apply_mask, intersect_masks
from nilearn.image import mean_img, resample_img
from nilearn import plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


temporal_lobe_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/" \
                     + "derivatives/fmriprep/ROIs/TemporalLobeMasks/mni_Temporal_mask_ero5_bin.nii.gz"
heschl_mask = "/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/" \
              + "derivatives/fmriprep/ROIs/HeschisGyrus/mni_Heschl_ROI.nii.gz"

base_path = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/encoding_results/temporal_lobe_mask' \
            '/lagging0to-15.3_permutation_train_only/sub-10/acq-N4/'
scores_path = base_path + 'sub-10_task-aomovie_acq-N4_desc-scores.nii.gz'
pval_path = base_path + 'sub-10_task-aomovie_acq-N4_desc-permutationpval.pkl'
masks = base_path + 'sub-10_task-aomovie_acq-N4_desc-masks.pkl'
perm_all_path = base_path + 'sub-10_task-aomovie_acq-N4_desc-permutations.pkl'

mask = joblib.load(masks)[0]
roi_mask = resample_img(heschl_mask, mask._affine, mask.shape, interpolation='nearest')
roi_mask = intersect_masks([mask, roi_mask])
mean_scores = mean_img(scores_path)
perm_all = joblib.load(perm_all_path)
pval = joblib.load(pval_path)
pval = unmask(pval, mask)
pval = apply_mask(pval, roi_mask)
n_permutations = perm_all.shape[1]
thresh = 1.0 / n_permutations
pval_mask = pval > thresh
pval[pval_mask] = 0
pvals = unmask(pval, roi_mask)
selected_r_scores = apply_mask(mean_scores, roi_mask)
selected_r_scores[pval_mask] = 0
selected_r_scores = unmask(selected_r_scores, roi_mask)

display = plotting.plot_glass_brain(pvals, threshold=0.00, colorbar=True,
                                            display_mode='lzry', plot_abs=False)
display.add_contours(temporal_lobe_mask,filled=False,colors='m')
display.add_contours(heschl_mask,filled=False,colors='g')
# proxy artist trick to make legend
cont1 = Rectangle((0,0),1,1,fc="magenta")
cont2 = Rectangle((0,0),1,1,fc="green")
plt.legend([cont1,cont2],['Temporal lobe','Heschl gyrus'])
plt.savefig(base_path + 'test_pvals.png')
plt.close()

display = plotting.plot_glass_brain(selected_r_scores, threshold=0.00, colorbar=True,
                                            display_mode='lzry', plot_abs=False)
display.add_contours(temporal_lobe_mask,filled=False,colors='m')
display.add_contours(heschl_mask,filled=False,colors='g')
# proxy artist trick to make legend
cont1 = Rectangle((0,0),1,1,fc="magenta")
cont2 = Rectangle((0,0),1,1,fc="green")
plt.legend([cont1,cont2],['Temporal lobe','Heschl gyrus'])
plt.savefig(base_path + 'test_select.png')
plt.close()
