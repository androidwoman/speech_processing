import os.path as path
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import ex3_windowing as win
import matplotlib.pyplot as plt

# Read the audio file SX83.WAV and sampling rate
file_path = path.join('.', 'Sounds')
sound_file = path.join(file_path, 'SX83.wav')
Fs, in_sig = wav.read(sound_file)

# Make sure the sampling rate is 16kHz, resample if necessary
Fs_target = 16000
if not (Fs == Fs_target):
    in_sig = sig.resample_poly(in_sig, Fs_target, Fs)
    Fs = Fs_target

# Parameters for windowing
frame_duration_ms = 25  # milliseconds
overlap_percent = 50  # 50% overlap
window_functions = ['hamming','rect', 'hann', 'cosine' ]
# 
# Calculate frame length and hop size
frame_length = int((frame_duration_ms / 1000) * Fs)  # Convert milliseconds to samples
overlap_size = int(frame_length * overlap_percent / 100)
hop_size = frame_length - overlap_size

for windowing_function in window_functions:
    # Obtain windowed frames using the windowing_3ex function
    windowed_data = win.ex3_windowing(in_sig, frame_length, hop_size, windowing_function)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # Original audio signal
    # Define time axis for the original audio signal
    time_orig = np.arange(len(in_sig)) / Fs
    axs[0].plot(time_orig, in_sig, color='b')
    axs[0].set_title('Original Audio Signal')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Amplitude')

    # Calculate the energy of each frame
    frame_energies = np.sum(windowed_data ** 2, axis=0)

    # Find the frame with the maximum energy (most voiced frame)
    voiced_frame_idx = np.argmax(frame_energies)
    voiced_frame = windowed_data[:, voiced_frame_idx]

    # Subplot 2: Plot the voiced frame
    start_time = voiced_frame_idx * hop_size / Fs_target
    time_axis_frame = np.arange(len(voiced_frame)) / Fs_target + start_time
    axs[1].plot(time_axis_frame * 1000, voiced_frame)
    axs[1].set_title(f'Most Voiced Frame ({windowing_function.capitalize()} Window)')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Amplitude')

    # Plot frequency domain of the voiced frame
    freq_axis = np.fft.rfftfreq(len(voiced_frame), 1 / Fs_target)
    voiced_frame_fft = np.abs(np.fft.rfft(voiced_frame))
    axs[2].plot(freq_axis, voiced_frame_fft)
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_title('Frequency Domain of Voiced Frame')
    fig.tight_layout()

    # # Plot the first half of the magnitude spectrum separately
   
    # Calculate the magnitude spectrum for each frame
    magnitude_spectrums = np.abs(np.fft.fft(windowed_data[:windowed_data.shape[0]//2,:], axis=1))
    
    # Get the number of frames and number of frequency bins
    number_of_frames, number_of_freq_bins = magnitude_spectrums.shape

    frequencies =np.fft.rfftfreq(len(windowed_data[:windowed_data.shape[0]//2,:]), 1 / Fs_target)
  
    plt.subplots(1, 1, figsize=(10, 8))
    # Plot the magnitude spectrum
    plt.imshow(magnitude_spectrums, aspect='auto', origin='lower', extent=[0, number_of_frames, 0, max(frequencies)])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Frame Number')
    plt.ylabel('Frequency (Hz)')
    plt.title(f'Magnitude Spectrum ({windowing_function.capitalize()} Window)')
    plt.show()

    plt.show()