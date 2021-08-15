# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
import glob
import os.path
#import json
import scipy.io.wavfile as wav
import librosa as lbr
import librosa.display
from scipy.signal import spectrogram, welch
import matplotlib.pyplot as plt
from scipy.signal import hilbert
#import argparse
import gc

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

def plot_timeseries(t, x, title, save):
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.gcf().tight_layout()
#    plt.show()
    plt.savefig(save)
    plt.close()

if __name__ == "__main__":
#    output_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/derivatives/fmriprep/'
#    output_sorted = True
#    use_noise = False
#    parser = argparse.ArgumentParser()
#    parser.add_argument("subject",type=str)
#    parser.add_argument("noise",type=str)
#    args = parser.parse_args()
#    subject = args.subject
#    use_noise = args.noise
    use_noise = False
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
#    subject = "03"
    subjects = ["03"]
    sessions = ["01","02","03"]
#    sessions = ["01"]
    # Name of the folders where unincluded runs (01 and 08) go for each session
#    unincluded_runs = {"01":"1st","02":"2nd","03":"3rd"}
#    for use_noise in [False,True]:
#        if use_noise:
#            stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
#                          +'sourcedata/stimuli/RecordedScannerNoise/'
#        else:
#            stim_folder = '/data2/azubaidi/ForrestGumpHearingLoss/BIDS_ForrGump/'\
#                          +'sourcedata/stimuli/RecordedStimuli/'
#    for subject in subjects:
    for session in sessions:
#            subses_stim_folder = os.path.join(stim_folder, f"sub-{subject}/ses-{session}/")
        subses_stim_folder = os.path.join(stim_folder, f"ses-{session}/")
        wav_files = glob.glob(subses_stim_folder + "*.wav")
        for wav_file in wav_files:
            outfile_base = os.path.basename(wav_file).split('.')[0]
            output_dir = os.path.join(subses_stim_folder, outfile_base)
            print("Converting", wav_file)
#                sig2, Fs = load(wav_file, sr=None, mono=False, dtype=np.float64)
            Fs, sig = wav.read(wav_file)
            if sig.ndim > 1:
                sig = np.mean(sig,axis=1,dtype=np.float32) # convert a WAV from stereo to mono
            else:
                sig = sig.astype(np.float32)
#            t = np.arange(0, len(sig)) / Fs
#            
#            plot_timeseries(t, sig, "Wavfile timeseries", output_dir+"_timeseries.png")
                
            ## set parameters ##
            #rate        = 44100                     # sampling rate
            winlen      = int(np.rint(Fs*0.025))  # 1102 Window length std 0.025s
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
            foursec     = int(np.rint(Fs*4.0)) # 4 seconds
            start_time  = 0
        
            config = {"n_fft":nfft, "sr":Fs, "win_length":winlen, 
                      "hop_length":hoplen, "n_mels":nmel, "fmax":highfreq, 
                      "fmin":lowfreq}
            
#            melspec, sr_spec, freqs = get_mel_spectrogram(wav_file, **config)
                
#            plt.imshow(melspec,aspect='auto',origin='lower')
##            plt.gca().set_yticks(np.arange(0,nmel,nmel_stride))
##            plt.gca().set_yticklabels(mel_frequencies[::nmel_stride])
##            plt.gca().set_xticks(np.arange(3,n_lag_bins,x_stride))
##            plt.gca().set_xticklabels(["%.1f"% tick for tick in x_ticks[3::x_stride]])
##            cbar = plt.colorbar()
##            cbar.set_label('Ridge coefficient', rotation=270)
#            plt.xlabel('Time lag (s)')
#            plt.ylabel('Mel frequency (Hz)')
#            plt.title('Spectrogram')
#            plt.savefig(output_dir+'_spectrogram.png')
#            plt.close()
            #create plots for smaller time intervalls
#                    short_sig = sig[0:foursec]
           
            envelope = np.abs(hilbert(sig))
#                plot_timeseries(t, envelope, "Envelope", output_dir+"_envelope.png")
            
            TR = 0.85
        
            f, Pxx = welch(envelope, fs=Fs, nfft=22050)
            plt.semilogy(f[:1000], Pxx[:1000])
            plt.xlabel('Frequency (Hz)')
#                plt.ylabel('dB')
            plt.title('Envelope Welch PSD')
            plt.savefig(output_dir+'_envelope_periodogram_welch.png')
#                plt.show()
            plt.close()
            #Mel
#            Smel = lbr.feature.melspectrogram(y=sig, n_fft=nfft, sr=Fs, 
#                                              win_length=winlen, hop_length=hoplen,
#                                              n_mels=nmel, fmax=highfreq, fmin=lowfreq) 
#            S_dB = lbr.power_to_db(Smel, ref=np.max)
#            #RMS
#            rms_np = np.zeros([1,len(Smel.T)])
#            for i in np.arange(len(Smel.T)):
#                rms_np[0,i] = np.sqrt(np.mean(Smel[:,i]**2))
#        
#            #plot
#            head, tail      = os.path.split(wav_file) # extracts the filename
#            plt.figure(figsize=(5,8))
#            
#            ax1 = plt.subplot(noay,noax,1) # Plot the amplitude envelope of a waveform.
#            lbr.display.waveplot(y=sig, sr=Fs, x_axis='time', offset=0.0, max_sr=1000) 
#            plt.title('Stereo waveform')
##                    plt.ylim([-25000, 25000])
#            ax1.set_xlabel('Time [sec]')
#            
#            ax2 = plt.subplot(noay,noax,2)
#            lbr.display.specshow(S_dB, y_axis='mel', sr=Fs, fmax=highfreq)              
#            plt.title('Mel-frequency spectrogram') 
#            
#            ax3 = plt.subplot(noay,noax,3)
#            plt.semilogy(rms_np.T, label='RMS Energy')
#            plt.xticks([])
#            plt.xlim([0, rms_np.shape[-1]])
#            plt.ylim([1, 10**11])
#            plt.title('RMS Energy')
#            plt.legend()
#            
#            plt.tight_layout()
#            plt.savefig(subses_stim_folder+'%s.png' %tail) # save plot to file 
##                    plt.show()  
#            plt.close()
#            
#            del sig
#            print(gc.collect())