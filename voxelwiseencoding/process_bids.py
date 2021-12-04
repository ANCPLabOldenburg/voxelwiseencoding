from preprocessing import preprocess_bold_fmri, make_X_Y
from encoding import get_model_plus_scores
import json
import numpy as np
from nilearn.image import load_img
from nilearn.masking import compute_epi_mask


def create_output_filename_from_args(sub=None, ses=None, task=None, acq=None,
                                     run=None, desc=None, recording=None):
    """ Creates filename for the model output """
    output_expr = ['sub-{}'.format(sub) if sub else None,
                   'ses-{}'.format(ses) if ses else None,
                   'task-{}'.format(task) if task else None,
                   'acq-{}'.format(acq) if acq else None,
                   'run-{}'.format(run) if run else None,
                   'desc-{}'.format(desc) if desc else None,
                   'recording-{}'.format(recording) if recording else None]
    output_expr = '_'.join([term for term in output_expr if term])
    return output_expr


def get_bids_filenames_for_encoding(**kwargs):
    """ Returns file names for encoding modelling in BIDS directory structure.

    The method takes as input BIDS compliant label value/pairs to select the
    filenames of the bold and stimulus files, including the corresponding json
    files with metadata. BOLD files have nifti format and hold the time series
    of the functional scans. They are recognized by their label "_bold" and
    extension "nii". Stimulus files are recognized by the label "_stim" and
    extension ".tsv". The names of the BOLD and stimulus files have the same
    label/value pairs. Thus they live in same directory.

    Args:
        **kwargs : A dictionary holding BIDS the entries listed below
                 arg['bids_dir']='/home/myhome/DATA/NameOfStudy/' The path to the
                 BIDS directory. The BIDS directory is where the file
                 dataset_description.json, the (raw) subject and the derivatives
                 folders are. (string)
                 arg['sub']=['OL3485'] The BIDS subject label (string)
                 arg['ses'] = ['01'] The BIDS session label (string)
                 arg['task'] = ['continuous'] The BIDS task label (string)
                 arg['run'] = ['01'] The BIDS label for individual runs (string or
                 list of strings for multiple runs)
                 arg['scope'] = 'raw' pyBIDS defines a scope from where the
                 filenames are extracted. Examples are 'raw' for the unprocessed
                 data, 'derivatives' for processed data in the derivatives folder,
                 the name of a pipleine that produced the derivative like
                 'fmriprep'. (string)
                 arg['bold_suffix'] = 'bold' The BIDS suffix for files containing
                 BOLD-data. (string)
                 arg['bold_extension'] = ['.nii', '.nii.gz'] BIDS extension for
                 BOLD data. (string or list of strings)
                 arg['stim_suffix'] = 'stim' The BIDS suffix for stimulus files.
                 (string)
                 arg['stim_extension'] = ['.tsv', '.tsv.gz'] The BIDS extension for
                 stimulus files. (string or list of strings)
                 arg['json_extension'] = 'json' The BIDS extension for json sidecar
                 files to BOLD and stimulus files. They hold the metadata (string)

    Returns:
        bold_files : A list of filenames holding the BOLD files matching the
                     search pattern. Includes path. (list of strings)
        bold_jsons : A list of filenames holding the _bold json sidecar files
                     matchings the search pattern. Includes path. (list of strings)
        stim_tsv :   A list of filenames holding the stimulus files matching the
                     search pattern. Includes path. (list of strings)
        stim_json :  A list of filenames holding the stimulus json sidecar files
                     matchings the search pattern. Includes path. (list of strings)

    TODO (ancpJR):
        So far we assume BOLD and corresponding stimulus files live in the same
        directory and have the same descriptive names. This is only a subset of
        BIDS definitions
    """
    # from bids import BIDSLayout
#    import bids # the pybids functions
    import ancpbids as bids
    # Avoid FutureWarning about dots in file extensions
#    bids.config.set_option('extension_initial_dot', True)

    # get all files in bids_dir
#    layout = bids.BIDSLayout(kwargs['bids_dir'], derivatives=kwargs['derivatives'],
#                             database_path='./temp/')
    subfolder = kwargs.get('subfolder', '')
    layout = bids.BIDSLayout(kwargs['bids_dir'] + subfolder)

    # select the nifti bold file names in the scope 
    bold_files = layout.get(subject=kwargs['sub'],
                            session=kwargs['ses'],
                            task=kwargs['task'], 
                            run=kwargs['run'],
                            #scope=kwargs['scope'], 
                            suffix=kwargs['bold_suffix'],
                            extension=kwargs['bold_extension'],
                            acquisition=kwargs['acq'],
                            space=kwargs['space'],
                            desc=kwargs['desc'],
                            return_type='filename')
    print('Found', len(bold_files), 'bold files.')
    for f in bold_files: print(f)

    # get the _bold.json names matching the same search pattern
    bold_jsons = layout.get(subject=kwargs['sub'],
                            session=kwargs['ses'],
                            task=kwargs['task'], 
                            run=kwargs['run'],
                            #scope=kwargs['scope'], 
                            suffix=kwargs['bold_suffix'],
                            extension=kwargs['json_extension'],
                            acquisition=kwargs['acq'],
                            space=kwargs['space'],
                            desc=kwargs.get('descboldjson'),
                            return_type='filename')
    print('Found', len(bold_jsons), 'corresponding json files.')
    for f in bold_jsons: print(f)

    if kwargs.get('remove_confounds'):
        # select confound tsv file names in the scope 
        confound_tsvs = layout.get(subject=kwargs['sub'],
                                   session=kwargs['ses'],
                                   task=kwargs['task'], 
                                   run=kwargs['run'],
                                   #scope=kwargs['scope'], 
                                   suffix=kwargs['confounds_suffix'],
                                   extension=kwargs['confounds_extension'],
                                   acquisition=kwargs['acq'],
                                   desc=kwargs['confounds_desc'],
                                   return_type='filename')
        print('Found', len(confound_tsvs), 'confounds tsv files.')
        for f in confound_tsvs: print(f)

    if kwargs.get('stim_always_CS'):
        acq = 'CS'
        session = 'CS'
    else:
        acq = kwargs['acq']
        session = kwargs['ses']
    # select the stimulus tsv file names in the scope. 
    # TODO so far we assume they live in the same  directory a the nifties 
    # and have the same descriptive names
    stim_tsv = layout.get(subject=kwargs['sub'], 
                          session=session,
                          task=kwargs['task'], 
                          run=kwargs['run'],
                          #scope=kwargs['scope'],
                          suffix=kwargs['stim_suffix'],
                          extension=kwargs['stim_extension'],
                          recording=kwargs['rec'],
                          acquisition=acq,
                          return_type='filename')
    print('Found', len(stim_tsv), 'stim files')
    for f in stim_tsv: print(f)

    # get the json filenames corresponding to the stim tsv filenames
    stim_json = layout.get(subject=kwargs['sub'],
                           session=session,
                           task=kwargs['task'], 
                           run=kwargs['run'],
                           #scope=kwargs['scope'],
                           suffix=kwargs['stim_suffix'],
                           extension=kwargs['json_extension'],
                           recording=kwargs['rec'],
                           acquisition=acq,
                           return_type='filename')
    print('Found', len(stim_json), 'stim.json files')
    for f in stim_json: print(f)

    # return the lists of filenames
    if kwargs.get('remove_confounds'):
        return bold_files, bold_jsons, stim_tsv, stim_json, confound_tsvs
    else:
        return bold_files, bold_jsons, stim_tsv, stim_json


def run_model_for_subject(bold_files, bold_json, stim_tsv, stim_json, **kwargs):
    """ Runs voxel-wise encoding model for a single subject and returns Ridges and scores

    Parameters
        kwargs :
            bold_files :    A list of the BOLD nifties to process
            bold_json :     A list of corresponding json files containing the
                            relevant metadata.
            stim_tsv :      A list of stimulus files (containing e.g the
                            spectrogram
                            or modulogram of the sound played)
            stim_json :     A list of the corresponding json holding the stimulus
                            metadata (e.g. descriptors of the columns in the
                            tsv(.gz))
            subject_label : the BIDS subject label
            bids_dir :      the path to the BIDS directory
            mask :          path to mask file or 'epi' if an epi mask should be
                            computed from the first BOLD run

            bold_prep_kwargs : None or dict containing the parameters for
                            preprocessing the BOLD files everything that is
                            accepted by nilearn's clean function is an acceptable
                            parameter
            preprocess_kwargs : None or dict containing the parameters for lagging
                            and aligning fMRI and stimulus
                            acceptable parameters are ones used by
                            preprocessing.make_X_Y
            estimator : None or sklearn-like estimator to use as an encoding model
                        default uses RidgeCV with individual alpha per target when
                        possible
            encoding_kwargs : None or dict containing the parameters for
                        evaluating the encoding model. Valid parameters are the
                        ones accepted by encoding.get_model_plus_scores

            additional BIDS specific arguments such as task, ses, desc,
            and recording

    Returns
        list of Ridge regressions, scores per voxel per fold

    """
    # Copy kwargs just for safety
    args = kwargs.copy()

    bold_prep_params = args['bold_prep_params']
    lagging_params = args['lagging_params']
    encoding_params = args['encoding_params']

    # metadata from the first BOLD file only. May not problematic as long as
    # only one RT is used in all nifties
    bold_meta = {}
    with open(bold_json[0], 'r') as fp:
        bold_meta = json.load(fp)

    if args['mask'] == 'epi':
        # get an epi mask
        mask = compute_epi_mask(bold_files[0])
    else:
        mask = load_img(args['mask'])

    # do BOLD preprocessing      
    preprocessed_bold = []
    masks = []
    if args.get('remove_confounds'):
        import pandas as pd
        for bold_file, confound_tsv in zip(bold_files, args['confound_tsvs']):
            confounds = pd.read_csv(confound_tsv, sep='\t')
            confounds = confounds[args['confounds_to_exclude']]
            prep_bold, resampled_mask = preprocess_bold_fmri(bold_file, mask=mask,
                                                             confounds=confounds,
                                                             **bold_prep_params)
            preprocessed_bold.append(prep_bold)
            masks.append(resampled_mask)
    else:
        for bold_file in bold_files:
            prep_bold, resampled_mask = preprocess_bold_fmri(bold_file, mask=mask,
                                                             **bold_prep_params)
            preprocessed_bold.append(prep_bold)
            masks.append(resampled_mask)

    # load stimuli and append their time series 
    stim_meta = []
    stim_data = []
    for tsv_fl, json_fl in zip(stim_tsv, stim_json):
        with open(json_fl, 'r') as fl:
            stim_meta.append(json.load(fl))
        stim = np.loadtxt(tsv_fl, delimiter='\t')
        if stim.ndim == 1:
            stim = stim[:, np.newaxis]
        stim_data.append(stim)

    stim_start_times = [st_meta['StartTime'] for st_meta in stim_meta]
    stim_TR = 1. / stim_meta[0]['SamplingFrequency']

    # temporally align stimulus and fmri data
    # ancpJR: The image o stim_data lagged does look a bit odd
    stim_data_lagged, preprocessed_bold, run_start_indices = make_X_Y(
        stim_data, preprocessed_bold, bold_meta['RepetitionTime'],
        stim_TR, stim_start_times=stim_start_times,
        save_lagged_stim_path=args.get('save_lagged_stim_path'),
        save_bold_path=args.get('save_preprocessed_bold_path'), **lagging_params)

    # compute ridge and scores for folds
    scores, bold_prediction, train_indices, test_indices, pval_list = \
        get_model_plus_scores(stim_data_lagged, preprocessed_bold,
                              estimator=args['estimator'],
                              run_start_indices=run_start_indices,
                              model_dump_path=args.get('model_dump_path'),
                              save_permutation_path=args.get('save_permutation_test_path'),
                              permutation_params=args.get('permutation_params'),
                              **encoding_params)

    return scores, masks, bold_prediction, train_indices, test_indices, run_start_indices, pval_list
