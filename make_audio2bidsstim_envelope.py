# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import glob
import os.path
import json
import re
#import scipy.io.wavfile as wav
import librosa as lbr 
from scipy.signal import hilbert, decimate
import librosa.display

#def get_mel_spectrogram(filename, log=True, sr=44100, hop_length=512, **kwargs):
#    '''Returns the (log) Mel spectrogram of a given wav file, the sampling rate
#    of that spectrogram and names of the frequencies in the Mel spectrogram
#
#    Parameters
#    ----------
#    filename : str, path to wav file to be converted
#    sr : int, sampling rate for wav file
#         if this differs from actual sampling rate in wav it will be resampled
#    log : bool, indicates if log mel spectrogram will be returned
#    kwargs : additional keyword arguments that will be
#             transferred to librosa's melspectrogram function
#
#    Returns
#    -------
#    a tuple consisting of the Melspectrogram of shape (time, mels), the 
#    repetition time in seconds, and the frequencies of the Mel filters in Hertz 
#    '''
#    wav, _ = lbr.load(filename, sr=sr)
#    melspecgrams = lbr.feature.melspectrogram(y=wav, sr=sr, hop_length=hop_length,
#                                              **kwargs)
#    if log:
#        melspecgrams[np.isclose(melspecgrams, 0)] = np.finfo(melspecgrams.dtype).eps
#        melspecgrams = np.log(melspecgrams)
#    log_dict = {True: 'Log ', False: ''}
#    freqs = lbr.core.mel_frequencies(
#            **{param: kwargs[param] for param in ['n_mels', 'fmin', 'fmax', 'htk']
#                if param in kwargs})
#    freqs = ['{0:.0f} Hz ({1}Mel)'.format(freq, log_dict[log]) for freq in freqs]
#    return melspecgrams.T, sr / hop_length, freqs

if __name__ == "__main__":
    output_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep/'
    output_sorted = True
    use_noise = False
    
    if use_noise:
        stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                      +'sourcedata/stimuli/RecordedScannerNoise/'
    else:
        stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
                      +'sourcedata/stimuli/PresentedStimuli/'
                      
#    subjects = ["01","02","03","04","05","06","07","08","09","10"]
#    subjects = ["03","09"]
    sessions = ["01"]
    subjects = ["10"]
#    sessions = ["01","02","03"]
    # Name of the folders where unincluded runs (01 and 08) go for each session
    unincluded_runs = {"01":"1st","02":"2nd","03":"3rd"}
    for subject in subjects:
        for session in sessions:
            if use_noise:
                subses_stim_folder = os.path.join(stim_folder, f"sub-{subject}/ses-{session}/")
            else:
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
#                rate, sig = wav.read(wav_file)
                sig, Fs = lbr.load(wav_file, sr=None, mono=True)
                
                from scipy.signal import welch
                #print('Plotting',save_path)
                #X = joblib.load(spec_path)
            #    print(X.shape)
            #    plt.plot(X[50])
                f, Pxx = welch(sig, fs=Fs)
                plt.semilogy(f,Pxx)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD (V²/Hz)')
                max_freq = f[np.argmax(Pxx)]
                plt.axvline(max_freq,color='r')
                plt.title(f'Max freq: {max_freq}')
#                plt.savefig(save_path)
                plt.show()
                plt.close()
                
                sig, Fs = lbr.load(wav_file, sr=200, mono=True)
                
                #print('Plotting',save_path)
                #X = joblib.load(spec_path)
            #    print(X.shape)
            #    plt.plot(X[50])
                f, Pxx = welch(sig, fs=Fs)
                plt.semilogy(f,Pxx)
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('PSD (V²/Hz)')
                max_freq = f[np.argmax(Pxx)]
                plt.axvline(max_freq,color='r')
                plt.title(f'Max freq: {max_freq}')
#                plt.savefig(save_path)
                plt.show()
                plt.close()
    
#                if len(sig.shape) > 1:
#                       sig = np.mean(sig,axis=1) # convert a WAV from stereo to mono
#                
                ## set parameters ##
                #rate        = 44100                     # sampling rate
                winlen      = int(np.rint(Fs*0.025))  # 1102 Window length std 0.025s
                overlap     = 0
                hoplen      = winlen-overlap            # 661 hop_length
                nfft        = winlen    # standard is = winlen = 1102 ... winlen*2 = 2204 ... nfft = the FFT size. Default for speech processing is 512.
                nmel        = 48        # n = the number of cepstrum to return = the number of filters in the filterbank
                lowfreq     = 100       # lowest band edge of mel filters. In Hz
                highfreq    = 8000      # highest band edge of mel filters. In Hz
                noay        = 3         # subplot: y-dimension
                noax        = 1         # subplot: x-dimension
#                foursec     = int(np.rint(rate*4.0)) # 4 seconds
#                start_time  = 0
            
#                config = {"n_fft":nfft, "sr":rate, "win_length":winlen, 
#                          "hop_length":hoplen, "n_mels":nmel, "fmax":highfreq, 
#                          "fmin":lowfreq}
                envelope = np.abs(hilbert(sig))
                envelope = decimate(envelope,5)
                Fs = 40
#                melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)
                outfile_base = os.path.basename(wav_file).split('.')[0]
                if use_noise:
                    outfile_base += '_stim'
                else:
                    outfile_base = f"sub-{subject}_" + outfile_base.replace("presstimuli","recording-audioenv_stim")
					
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
                png_file = os.path.join(subses_output_folder, outfile_base+'.png')
                
#                print('Saving ', json_file)
#                np.savetxt(tsv_file, envelope, delimiter='\t')
#                metadata = {'SamplingFrequency': 40, 'StartTime': 0,
#                            'Columns': [0]}
#                with open(json_file, 'w+') as fp:
#                    json.dump(metadata, fp)
                    
#                Smel = lbr.feature.melspectrogram(y=envelope, n_fft=nfft, sr=Fs, 
#                                                  win_length=winlen, hop_length=hoplen,
#                                                  n_mels=nmel, fmax=highfreq, fmin=lowfreq) 
                
                
                
#                Smel = lbr.feature.melspectrogram(y=envelope, sr=Fs) 
#                S_dB = lbr.power_to_db(Smel, ref=np.max)
#                #RMS
#                rms_np = np.zeros([1,len(Smel.T)])
#                for i in np.arange(len(Smel.T)):
#                    rms_np[0,i] = np.sqrt(np.mean(Smel[:,i]**2))
#            
#                #plot
##                head, tail      = os.path.split(wav_file) # extracts the filename
#                plt.figure(figsize=(5,8))
#                
#                ax1 = plt.subplot(noay,noax,1) # Plot the amplitude envelope of a waveform.
##                lbr.display.waveplot(y=envelope, sr=Fs, x_axis='time', offset=0.0, max_sr=1000)
#                t = np.arange(len(envelope))/Fs
#                plt.plot(t,envelope)
#                plt.title('Envelope')
#    #                    plt.ylim([-25000, 25000])
#                ax1.set_xlabel('Time [sec]')
#                
#                ax2 = plt.subplot(noay,noax,2)
#                lbr.display.specshow(S_dB,y_axis='mel', sr=Fs,fmax=20)  
##                lbr.display.specshow(S_dB, y_axis='mel', sr=Fs, fmax=highfreq)              
#                plt.title('Mel-frequency spectrogram') 
#                
#                ax3 = plt.subplot(noay,noax,3)
#                plt.semilogy(rms_np.T, label='RMS Energy')
#                plt.xticks([])
##                plt.xlim([0, rms_np.shape[-1]])
##                plt.ylim([1, 10**11])
#                plt.title('RMS Energy')
##                plt.legend()
#                
#                plt.tight_layout()
#                plt.savefig(png_file) # save plot to file 
#    #                    plt.show()  
#                plt.close()
