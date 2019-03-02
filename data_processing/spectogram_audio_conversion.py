import glob
import os
from collections import namedtuple

import cv2
import numpy as np
from scipy.io import wavfile

from data_processing.spectogram_utils import pretty_spectrogram, create_mel_filter, make_mel, mel_to_spectrogram, \
    invert_pretty_spectrogram

spec_config = namedtuple(
    'spec_config',
    [
        'fft_size',  # window size for the FFT
        'step_size',  # distance to slide along the window (data time)
        # threshold for spectrograms (lower filters out more noise)
        'spec_thresh',
        'n_mel_freq_components',  # number of mel frequency channels
        'shorten_factor',  # how much should we compress the x-axis (time)
        'start_freq',  # Hz # What frequency to start sampling our melS from
        'end_freq'
    ]
)

config = spec_config(
    fft_size=4096,
    step_size=128,
    spec_thresh=5,
    n_mel_freq_components=128,
    shorten_factor=24,
    start_freq=28,
    end_freq=4000
)


def mel_spec_images_from_segment(segment):
    # Hz # What frequency to stop sampling our melS from
    wav_spectrogram = pretty_spectrogram(segment.astype('float64'), fft_size=config.fft_size,
                                         step_size=config.step_size, log=True, thresh=config.spec_thresh)
    # Generate the mel filters
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size=config.fft_size,
                                                         n_freq_components=config.n_mel_freq_components,
                                                         start_freq=config.start_freq,
                                                         end_freq=config.end_freq)
    # generate mel spectogram
    mel_spec = make_mel(wav_spectrogram, mel_filter,
                        shorten_factor=config.shorten_factor)
    # converting to the [0..1] domain
    return mel_spec.astype(float) / config.spec_thresh + 1


def audio_from_mel_spec(spectogram):
    # converting back to the original domain
    combined = (spectogram - 1) * config.spec_thresh
    # Generate the mel filters
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size=config.fft_size,
                                                         n_freq_components=config.n_mel_freq_components,
                                                         start_freq=config.start_freq,
                                                         end_freq=config.end_freq)
    # generating spectogram from the mel spectogram
    mel_inverted_spectrogram = mel_to_spectrogram(combined, mel_inversion_filter,
                                                  spec_thresh=config.spec_thresh,
                                                  shorten_factor=config.shorten_factor)
    # inverting the spectogram to audio
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size=config.fft_size,
                                                   step_size=config.step_size, log=True, n_iter=10)
    return inverted_mel_audio


def generate_images_from_audio_files(audio_folder, spec_folder, ext='wav'):
    for input_wav_file in glob.glob(os.path.join(audio_folder, '*.' + ext)):
        audio_name = os.path.split(input_wav_file)[-1].split('.')[0]
        print('Generating {}'.format(input_wav_file))
        rate, data = wavfile.read(input_wav_file)
        mel_spec = mel_spec_images_from_segment(data)
        cv2.imwrite(os.path.join(spec_folder, audio_name + '.png'), mel_spec * 255)


def restore_audio_from_images(spec_folder, audio_folder, ext='wav'):
    for input_spec_file in glob.glob(os.path.join(spec_folder, '*.png')):
        image_name = os.path.split(input_spec_file)[-1].split('.')[0]
        print('Restoring {}'.format(input_spec_file))
        spectogram = cv2.imread(input_spec_file, cv2.IMREAD_GRAYSCALE)[:, :512] / 255.
        audio = audio_from_mel_spec(spectogram)
        # multiplying with a magic number and exporting as wav
        wavfile.write(os.path.join(audio_folder, image_name + '_restored.' + ext), 44100, audio * 255)


if __name__ == '__main__':
    generate_images_from_audio_files('../data/audio_test', '../data/spectograms')
    restore_audio_from_images('../data/spectograms', '../data/recreated_audio')
