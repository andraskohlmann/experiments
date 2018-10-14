import glob
import os

import numpy as np
from pydub import AudioSegment


def get_files_from_library(path, ext='wav'):
    files = glob.glob(os.path.join(path, '*' + ext))
    files.sort()
    return files


def audio_to_array(file, ext='wav'):
    audio = AudioSegment.from_file(file, format=ext).set_channels(1)
    return np.array(audio.get_array_of_samples()), audio.frame_rate, audio.duration_seconds


def array_to_audio(array, path, ext='mp3'):
    audio = AudioSegment(array, sample_width=2, frame_rate=44100, channels=1)
    audio.export(path + '.' + ext, format(ext))
