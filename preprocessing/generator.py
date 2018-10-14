import numpy as np

from preprocessing.audio_import import get_files_from_library, audio_to_array


def audio_segment_generator(sample_size, batch_size, path, ext):
    files = get_files_from_library(path, ext)
    audio_bytes = [audio_to_array(file, ext)[0] for file in files]

    while True:
        indices = np.random.random_integers(0, len(files) - 1, batch_size)
        samples = np.zeros((batch_size, sample_size))
        for i, idx in enumerate(indices):
            frame = np.random.randint(0, len(audio_bytes[idx]) - sample_size)
            samples[i] = audio_bytes[idx][frame:frame + sample_size]
        yield samples


def audio_segments_from_single_file(sample_size, path_to_file):
    audio_bytes, ff, sec = audio_to_array(path_to_file, path_to_file.split('.')[-1])
    sample_num = int(ff * sec)
    return [audio_bytes[i:i + sample_size] for i in range(0, sample_num - sample_size, sample_size)]
