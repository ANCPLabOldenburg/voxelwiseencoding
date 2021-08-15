# -*- coding: utf-8 -*-

#!/usr/bin/env python
import numpy as np
import glob
import os.path
#import json
import re
#import scipy.io.wavfile as wav
from librosa import load
from scipy.stats import gamma
from scipy.signal import spectrogram, welch, hilbert, fftconvolve, decimate
import matplotlib.pyplot as plt

def make_hrf(time_points, overshoot_delay=6, undershoot_delay=12, 
             amplitiude_factor_undershoot=0.35):
    """ Make a simple hrf for the time points passed. """
    hrf = gamma.pdf(time_points, overshoot_delay) - \
        gamma.pdf(time_points, undershoot_delay) * amplitiude_factor_undershoot
    # Normalize to integral 1  
    return hrf / sum(hrf)

def plot_timeseries(t, x, title, save):
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.gcf().tight_layout()
#    plt.show()
    plt.savefig(save)
    plt.close()

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
#    stim_extension_old = 'recstimuli'
#    stim_extension = 'stim'
#    recording_extension = 'recording-rec_'
#    recording_extension = ''
#    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    subjects = ["03"]
#    subjects = ["09"]
#    sessions = ["01","02","03"]
    sessions = ["01"]
    # Name of the folders where unincluded runs (01 and 08) go for each session
    unincluded_runs = {"01":"1st","02":"2nd","03":"3rd"}
    for subject in subjects:
        for session in sessions:
            subses_stim_folder = os.path.join(stim_folder, f"/ses-{session}/")
#            subses_stim_folder = os.path.join(stim_folder, f"sub-{subject}/ses-{session}/")
            wav_files = glob.glob(subses_stim_folder + "*.wav")
            if output_sorted:
                subses_output_folder_sorted = os.path.join(output_folder, f"sub-{subject}/ses-%s/func/")
            else:
                subses_output_folder = os.path.join(output_folder, f"sub-{subject}/ses-{session}/func/")
            
            for wav_file in wav_files:
#                if 'run-01' in wav_file or 'run-08' in wav_file:
##                    print("Skipping ",wav_file)
#                    continue
                
                outfile_base = os.path.basename(wav_file).split('.')[0]
                if use_noise:
                    outfile_base += '_stim'
#                outfile_base = outfile_base.replace(stim_extension_old,
#                                                    recording_extension+stim_extension)
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
                output_dir = os.path.join(subses_output_folder, "stim/")
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                output_dir += outfile_base
                # Found a memory leak in scipy.io.wavfile when reading a 16bit stereo
                # wav file: sys.getsizeof and numpys .__sizeof__ return 112
                # when the actual size is 159293432
                # garbage collection seems to fail and parts of the array are
                # overwritten while in use
                print("Converting", wav_file)
                sig2, Fs = load(wav_file, sr=None, mono=False, dtype=np.float64)
#                Fs, sig = wav.read(wav_file)
                # For some reason sig only has size 96 when it should have 
                # size 155771172. Recasting the array as a different type
                # or performing an operation like mean seems to fix it.
                if sig2.ndim > 1:
                    sig = np.mean(sig2,axis=0,dtype=np.float32) # convert a WAV from stereo to mono
                else:
                    sig = sig2.copy()
                t = np.arange(0, len(sig)) / Fs
                
#                plot_timeseries(t, sig, "Wavfile timeseries", output_dir+"_timeseries.png")
                envelope = np.abs(hilbert(sig))
#                plot_timeseries(t, envelope, "Envelope", output_dir+"_envelope.png")
                
                TR = 0.85
                f, t, Sxx = spectrogram(envelope, fs=1/TR, nperseg=64)
                plt.pcolormesh(t, f, Sxx, shading='gourad')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Envelope spectrogram')
                plt.savefig(output_dir+'_envelope_spectrogram_nohrf.png')
#                plt.show()
                plt.close()
                
                # Make hemodynamic response function
#                sample_duration = 1 / Fs
#                time_points = np.arange(0, 25, sample_duration) # 25 seconds long
#                hrf = make_hrf(time_points)
#                plot_timeseries(time_points, hrf, "HRF", output_dir+"_hrf.png")
                # Convolve envelope with HRF
#                envelope_convolved = fftconvolve(envelope, hrf, mode='valid')
#                t = np.arange(0, len(envelope_convolved)) / Fs
#                plot_timeseries(t, envelope_convolved, "Envelope convolved with HRF",
#                                output_dir+"_envelope_convolved.png")
#                # Downsampling from wav file sampling rate 44100 Hz to 
#                # fMRI sampling rate 1/TR (~1.176 Hz).
#                # Decimation factor: 44100 * TR = 37485 = 3^2 * 5 * 7^2 * 17
#                envelope_convolved = decimate(envelope_convolved,17)
#                envelope_convolved = decimate(envelope_convolved,9)
#                envelope_convolved = decimate(envelope_convolved,7)
#                envelope_convolved = decimate(envelope_convolved,7)
#                envelope_convolved = decimate(envelope_convolved,5)
#                # For some reason envelope_convolved only has size 96 when it 
#                # should have size 8176. 
#                envelope_convolved = np.asarray(envelope_convolved,dtype=np.float32)
#                t = np.arange(0, len(envelope_convolved)) * TR
#                plot_timeseries(t, envelope_convolved,
#                                "Envelope convolved with HRF and downsampled to fMRI Fs",
#                                output_dir+"_envelope_convolved_decimated.png")
#                
#                f, t, Sxx = spectrogram(envelope_convolved, fs=1/TR, nperseg=64)
#                plt.pcolormesh(t, f, Sxx, shading='gourad')
#                plt.xlabel('Time (s)')
#                plt.ylabel('Frequency (Hz)')
#                plt.title('Envelope spectrogram')
#                plt.savefig(output_dir+'_envelope_spectrogram.png')
##                plt.show()
#                plt.close()
#                
#                f, Pxx = welch(envelope_convolved, fs=1/TR)
#                plt.semilogy(f, Pxx)
#                plt.xlabel('Frequency (Hz)')
##                plt.ylabel('dB')
#                plt.title('Envelope Welch PSD')
#                plt.savefig(output_dir+'_envelope_periodogram_welch.png')
##                plt.show()
#                plt.close()
##                break
#                del sig, envelope, envelope_convolved
#                import gc
#                print(gc.collect())