# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
import glob
import os.path
import json
import scipy.io.wavfile as wav
import librosa as lbr 
#import librosa.display

def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate of that spectrogram and names of the frequencies in the Mel spectrogram

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
    a tuple consisting of the Melspectrogram of shape (time, mels), the repetition time in seconds, and the frequencies of the Mel filters in Hertz 
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
    stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                  +'sourcedata/stimuli/RecordedStimuli/'
    output_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'
    
    stim_extension_old = 'recstimuli'
    stim_extension = 'stim'
    recording_extension = 'recording-rec_'
    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    sessions = ["01","02","03"]
    for subject in subjects:
        for session in sessions:
            subses_stim_folder = os.path.join(stim_folder, f"sub-{subject}/ses-{session}/")
            subses_output_folder = os.path.join(output_folder, f"sub-{subject}/ses-{session}/func/")
            wav_files = glob.glob(subses_stim_folder + "*.wav")
            
            for wav_file in wav_files:
                print("Converting ",wav_file)
                rate, sig = wav.read(wav_file)
                if len(sig.shape) > 1:
                       sig = np.mean(sig,axis=1) # convert a WAV from stereo to mono
                
                ## set parameters ##
                #rate        = 44100                     # sampling rate
                winlen      = int(np.rint(rate*0.025))  # 1102 Window length std 0.025s
                overlap     = int(np.round(rate*0.010)) # 441 Over_laplength std 0.01s
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
                outfile_base = outfile_base.replace(stim_extension_old,
                                                    recording_extension+stim_extension)
                tsv_file = os.path.join(subses_output_folder, outfile_base+'.tsv.gz')
                json_file = os.path.join(subses_output_folder, outfile_base+'.json')
                np.savetxt(tsv_file, melspec, delimiter='\t')
                metadata = {'SamplingFrequency': sr_spec, 'StartTime': start_time,
                            'Columns': freqs}
                with open(json_file, 'w+') as fp:
                    json.dump(metadata, fp)
            
#                #create plots for smaller time intervalls
#                short_sig = sig[0:foursec]
#               
#                #Mel
#                Smel = lbr.feature.melspectrogram(y=sig, n_fft=nfft, sr=rate, win_length=winlen, hop_length=hoplen, n_mels=nmel, fmax=highfreq, fmin=lowfreq) 
#                S_dB = lbr.power_to_db(Smel, ref=np.max)
#                #RMS
#                rms_np = np.zeros([1,len(Smel.T)])
#                for i in np.arange(len(Smel.T)):
#                    rms_np[0,i] = np.sqrt(np.mean(Smel[:,i]**2))
#            
#                #plot
#                head, tail      = os.path.split(files[k-1]) # extracts the filename
#                plt.figure(figsize=(5,8))
#                
#                ax1 = plt.subplot(noay,noax,1) # Plot the amplitude envelope of a waveform.
#                lbr.display.waveplot(y=sig, sr=rate, x_axis='time', offset=0.0, max_sr=1000) 
#                plt.title('Stereo waveform for: %s' %tail)
#                plt.ylim([-25000, 25000])
#                ax1.set_xlabel('Time [sec]')
#                
#                ax2 = plt.subplot(noay,noax,2)
#                lbr.display.specshow(S_dB, y_axis='mel', sr=rate, fmax=highfreq)              
#                plt.title('Mel-frequency spectrogram for: %s' %tail) 
#                
#                ax3 = plt.subplot(noay,noax,3)
#                plt.semilogy(rms_np.T, label='RMS Energy')
#                plt.xticks([])
#                plt.xlim([0, rms_np.shape[-1]])
#                plt.ylim([1, 10**11])
#                plt.title('RMS Energy for: %s' %tail)
#                plt.legend()
#                
#                plt.tight_layout()
#                plt.savefig(stim_folder+'4 seconds librosa Mel and RMS analysis for %s.png' %tail) # save plot to file 
#                plt.show()               
                
        
