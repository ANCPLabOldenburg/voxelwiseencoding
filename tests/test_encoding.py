import os
import json

import numpy as np
from nibabel import load 
from scipy.stats import gamma, zscore

from voxelwiseencoding import encoding as enc
from voxelwiseencoding import preprocessing as preproc
from voxelwiseencoding.process_bids import run_model_for_subject

def make_random_test_data():
    '''Random predictor and target data to test voxelwise encoding models.
    
    Calculates the target from the predictor using y=X*beta+epsilon. Adds
    different noise levels epsilon to the y's so that the model fitting
    process finds different optimal regularization parameters
    
    Returns:
        X: Predictor time series
        y: Target time series
    '''
    
    X = np.reshape(np.random.randn(1000, 5), (100, -1))
    betas = np.random.randn(X.shape[-1], 27) * 2
    y = X.dot(betas) + np.random.rand(100, 27).dot(np.diag([0.1]*26 + [10]))
    return X, y


def test_encoding():
    ''' Only tests if the correct number of dimensions are returned
    
    Throws an error if not
    '''
    
    X, y  = make_random_test_data()
    #ridges, scores = enc.get_model_plus_scores(X, y, n_splits=2)
    #assert len(ridges) == 2
    #assert scores.shape == (27, 2)
    models, score_list, bold_prediction, train_indices, test_indices =\
        enc.get_model_plus_scores(X, y, cv=2)    
    assert len(models) == 2 
    assert score_list.shape == (27, 2)
    
def make_hrf(time_points,overshoot_delay=6, undershoot_delay=12, amplitiude_factor_undershoot=0.35):
    ''' Make a simple hrf for the time points passed.
    '''
    hrf=gamma.pdf(time_points, overshoot_delay)-gamma.pdf(time_points, undershoot_delay)*amplitiude_factor_undershoot
    
    #Normalize to integral 1  
    return hrf/sum(hrf)

def test_encoding_synthethic_data(predictor_filename, lags=None,
                                  zscore_predictor=False):
    ''' Simulate an encoding model with synthetic data.
    
    Uses a stimulus time series as predictors and generates targets for a
    multivariate regression by convolving the stimulus times series with a
    haemodynamic response function (hrf). An example is a spectrogram of 
    a sound pressure time series with 800 time points (samples) and 31
    frequencies in clomuns. If the predictor matrix has
    multiple columns they are interpreted as features and each feature is
    convolved with a hrf to yield a target. In the example the function would  
    yield 31 target time series (conceive as voxel time series in MRI).
    Each is predicted by the full set of stimulus features. This should lead
    to maximum correlation with the stmulus feature the target was created
    from.
    
    Encoding models often include derived features which are appended to the
    as additional columns to the predictor matrix. The augmentaion
    implemented here are lagged predictor time series. They can compensate
    for an unknown lag of the response to the stimulus (e.g. hrf delay). Note
    that adding many lags increases encoding model complexity

    Args:
        predictor_filename: BIDS compatible stimulus file with format tsv or
          tsv.gz. Stimulus features are in columns and time along rows. Has a
          sidecar *.json defining the sampling rate in a field named
          'SamplingFrequency. (fileobtain   type tsv or tsv.gz)
        
        lags: The length of the maximum stimulus lag in seconds. The number
          of lags depends on the sampling frequency of the stimulus. for
          example a lag of 4s at a sampling frequency of 0.5 Hz (2s sample
          duration) The lags adds two additional sets of predictor time 
          series with lags 2s and 4s, respectively. Lags do not add targets.
          
        zscore_predictor: Whether to zscore the predictor time series.
          Default is False.
        
    Returns:
        models: Scikit-learn regression (encoding) models
        
        cv: Number of cross validation splits.
        
        score_list: The quality of the fit. scikit-learn standard scorer
          returns coefficient of determination r**2=sum((y-y_hat)**2)
        
        target_prediction: Responses predicted by encoding model (y_hat)
        
        y: Target time series 
        
        X: Predictor time series augment by lags. Columns with lowest indices
          hold original predictors 
        
        train_indices: Indices to model training samples for each fold 
        
        test_indices: Indices to test samples for each fold
    '''

    DEBUG =False     
    
    if DEBUG:
        predictor_filename=("/home/rieger/DDATA/DATA/OL_MRT/"
                            "data_ForrestSparse/sub-OL2275/ses-01/func/"
                            "sub-OL2275_ses-01_task-"
                            "continuous_run-01_stim.tsv.gz")
    
    #read the predictor and zscore if demanded
    stim_data=np.loadtxt(predictor_filename, delimiter='\t')
    if zscore_predictor:
        stim_data=zscore(stim_data)
    
    #read corresponding json (may not work on windows systems) 
    if predictor_filename[-3:] == ".gz": # may first have to remove .gz
        predictor_no_extension, extension = os.path.splitext(
            predictor_filename)
        predictor_no_extension, extension = os.path.splitext(
            predictor_no_extension)
    elif predictor_filename[-4:] == ".tsv":
        predictor_no_extension, extension = os.path.splitext(
            predictor_filename)
    else :
        raise NameError("Unknown extension {}".predictor_filename)
    
    #Read the corresponding json
    predictor_json_filename=predictor_no_extension+'.json'
    with open(predictor_json_filename,"r") as fp:
        predictor_params=json.load(fp)
        
    #convert sampling frequency to duration
    sample_duration=1/predictor_params['SamplingFrequency'] 
    
    #The total duration of the stimulus
    #predictor_duration=stim_data.shape[0]*sample_duration
    
    #Convolve the stimulus with a hrf
    time_points = np.arange(0, 25,sample_duration ) #25 seconds long
    hrf=make_hrf(time_points) #All prameters as defined in make_hrf
    
    if DEBUG:
        stim_data[:,10]=0
        stim_data[::25,10]=1
    
    stim_data_convolved=np.zeros(stim_data.shape)
    
    # hrf.shape-1 corrects for the relative shift of convolution 
    for k in np.arange(stim_data.shape[1]):
        #stim_data_convolved[hrf.shape[0]-1:,k]=conv_dot(stim_data[:,k],hrf)
        stim_data_convolved[hrf.shape[0]-1:,k] = np.convolve(
            stim_data[:,k],hrf,mode='valid')
    
    # Use convolved stim data to run the voxelwise encoding   
    # Functions below want ndarrays in a list of ndarrys
    stim_data_as_list=[]
    stim_data_as_list.append(stim_data)
    stim_data_convolved_as_list=[]
    stim_data_convolved_as_list.append(stim_data_convolved)
    
    # This function seems to loose samples. Seem to be missing at the start
    X,y,_ = preproc.make_X_Y(stim_data_as_list, stim_data_convolved_as_list,
                           sample_duration, sample_duration)
    models, score_list, target_prediction, train_indices, test_indices =\
        enc.get_model_plus_scores(X, y, cv=2)
    
    return  models, score_list, target_prediction, y, X, train_indices, test_indices