# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import glob
import os.path
import json
import re
import scipy.io.wavfile as wav
import librosa as lbr 
#import librosa.display

def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate
    of that spectrogram and names of the frequencies in the Mel spectrogram

    Parameters
    ----------
    filename : str, path to wav file to be converted
    sr : int, sampling rate for wav file
         if this differs from actual sampling rate in wav it will be resampled
    log : bool, indicates if log mel spectrogram will be returned
    kwargs : additional keyword arguments that will be
             transferred to librosa's melspectrogram function

    Returns
    -------
    a tuple consisting of the Melspectrogram of shape (time, mels), the 
    repetition time in seconds, and the frequencies of the Mel filters in Hertz 
    '''
    wav, _ = lbr.load(filename, sr=sr)
    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
                                              **kwargs)
    if log:
        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
        melspecgrams = np.log(melspecgrams)
    log_dict = {True: 'Log ', False: ''}
    freqs = lbr.core.mel_frequencies(
            **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
                if param in kwargs})
    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
    return melspecgrams.T, sr / hop_length, freqs

if __name__ == "__main__":
    output_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep/'
    output_sorted = True
#    use_noise = False
    
    stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                  +'sourcedata/stimuli/PresentedStimuli/'
#    if use_noise:
#        stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
#                      +'sourcedata/stimuli/RecordedScannerNoise/'
#    else:
#        stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
#                      +'sourcedata/stimuli/RecordedStimuli/'
#    stim_extension_old = 'recstimuli'
#    stim_extension = 'stim'
#    recording_extension = 'recording-rec_'
#    recording_extension = ''
#    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    subjects = ["10"]
#    subjects = ["03","09"]
    sessions = ["01","02","03"]
    # Name of the folders where unincluded runs (01 and 08) go for each session
    unincluded_runs = {"01":"1st","02":"2nd","03":"3rd"}
#    sessions = ["01"]
    for subject in subjects:
        for session in sessions:
#            subses_stim_folder = os.path.join(stim_folder, f"sub-{subject}/ses-{session}/")
            subses_stim_folder = os.path.join(stim_folder, f"ses-{session}/")
            wav_files = glob.glob(subses_stim_folder + "*.wav")
            if output_sorted:
                subses_output_folder_sorted = os.path.join(output_folder, f"sub-{subject}/ses-%s/func/")
            else:
                subses_output_folder = os.path.join(output_folder, f"sub-{subject}/ses-{session}/func/")
            
            for wav_file in wav_files:
#                if 'run-01' in wav_file or 'run-08' in wav_file:
##                    print("Skipping ",wav_file)
#                    continue
                print("Converting ",wav_file)
                rate, sig = wav.read(wav_file)
                if len(sig.shape) > 1:
                       sig = np.mean(sig,axis=1) # convert a WAV from stereo to mono
                
                ## set parameters ##
                #rate        = 44100                     # sampling rate
                winlen      = int(np.rint(rate*0.025))  # 1102 Window length std 0.025s
#                overlap     = int(np.round(rate*0.010)) # 441 Over_laplength std 0.01s
#                winlen      = int(np.rint(rate*0.850))
                overlap     = 0
                hoplen      = winlen-overlap            # 661 hop_length
                nfft        = winlen    # standard is = winlen = 1102 ... winlen*2 = 2204 ... nfft = the FFT size. Default for speech processing is 512.
                nmel        = 48        # n = the number of cepstrum to return = the number of filters in the filterbank
                lowfreq     = 100       # lowest band edge of mel filters. In Hz
                highfreq    = 8000      # highest band edge of mel filters. In Hz
                noay        = 3         # subplot: y-dimension
                noax        = 1         # subplot: x-dimension
                foursec     = int(np.rint(rate*4.0)) # 4 seconds
                start_time  = 0
            
                config = {"n_fft":nfft, "sr":rate, "win_length":winlen, 
                          "hop_length":hoplen, "n_mels":nmel, "fmax":highfreq, 
                          "fmin":lowfreq}
                
                melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)
                outfile_base = os.path.basename(wav_file).split('.')[0]
#                if use_noise:
#                    outfile_base += '_stim'
                outfile_base = f"sub-{subject}_" + outfile_base.replace("presstimuli","recording-audio_stim")
                if output_sorted:
                    if 'run-01' in wav_file or 'run-08' in wav_file:
                        condition = unincluded_runs[session]
                    elif 'acq-CS' in wav_file:
                        condition = 'CS'
                    elif 'acq-N4' in wav_file:
                        condition = 'N4'
                    elif 'acq-S2' in wav_file:
                        condition = 'S2'
                    else:
                        raise Exception('No valid condition tag found in wav ' 
                                        +'file name. Valid tags: CS, N4, S2')
                    subses_output_folder = subses_output_folder_sorted % condition
                    outfile_base = re.sub(r'ses-[0-9]*','ses-'+condition,outfile_base)
                tsv_file = os.path.join(subses_output_folder, outfile_base+'.tsv.gz')
                json_file = os.path.join(subses_output_folder, outfile_base+'.json')
#                print('Saving ', json_file)
                np.savetxt(tsv_file, melspec, delimiter='\t')
                metadata = {'SamplingFrequency': sr_spec, 'StartTime': start_time,
                            'Columns': freqs}
                with open(json_file, 'w+') as fp:
                    json.dump(metadata, fp)