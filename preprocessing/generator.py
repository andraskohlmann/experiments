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
        yield np.expand_dims(samples, -1), np.expand_dims(samples, -1)
