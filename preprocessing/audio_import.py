import glob
import os

import numpy as np
from pydub import AudioSegment


def mp3_to_array(file):
    audio = AudioSegment.from_file(file, format='mp3').set_channels(1)
    return np.array(audio.get_array_of_samples()), audio.frame_rate


# # files
# src = '../example.mp3'
#
# array, frame_rate = mp3_to_array(src)
# print(array)


def get_files_from_library(path, ext='wav'):
    files = glob.glob(os.path.join(path, '*' + ext))
    files.sort()
    return files


def audio_to_array(file, ext='wav'):
    audio = AudioSegment.from_file(file, format=ext).set_channels(1)
    return np.array(audio.get_array_of_samples()), audio.frame_rate, audio.duration_seconds
