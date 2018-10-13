import numpy as np
from pydub import AudioSegment


def mp3_to_array(file):
    audio = AudioSegment.from_file(file, format='mp3').set_channels(1)
    return np.array(audio.get_array_of_samples()), audio.frame_rate


# files
src = '../example.mp3'

array, frame_rate = mp3_to_array(src)
print(array)
