"""
EX3_WINDOWING Based on the input parameters, generate a n x m matrix of windowed
frames, with n corresponding to frame_length and m corresponding to number
of frames. The first frame starts at the beginning of the data.
"""

import os
import numpy as np


    




def ex3_windowing(data, frame_length, hop_size, windowing_function):
    data_length = len(data)
    number_of_frames = 1 + ((data_length - frame_length) // hop_size)
    windowed_frames = np.zeros((frame_length, number_of_frames))

    for i in range(number_of_frames):
        start_index = i * hop_size
        end_index = start_index + frame_length

        frame = data[start_index:end_index]
        if windowing_function == 'rect':
            window= np.ones(frame_length)
        elif windowing_function == 'hann':
            window= np.hanning(frame_length)
        elif windowing_function == 'cosine':
            window= np.cos(np.linspace(0, np.pi, frame_length))
        elif windowing_function == 'hamming':
            window= np.hamming(frame_length)
        else:
            raise ValueError("Invalid windowing function specified")
        windowed_frame = frame * window
        windowed_frames[:, i] = windowed_frame

    return windowed_frames
