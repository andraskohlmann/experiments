import glob
import os

import cv2
import numpy as np
from scipy.io import wavfile
from spectogram import pretty_spectrogram, create_mel_filter, make_mel, butter_bandpass_filter, mel_to_spectrogram, \
    invert_pretty_spectrogram

from preprocessing.audio_import import (audio_to_array, get_files_from_library,
                                        normalize_audio)


def audio_segment_generator(sample_size, batch_size, path, ext):
    files = get_files_from_library(path, ext)
    audio_bytes = [audio_to_array(file, ext)[0] for file in files]

    while True:
        indices = np.random.random_integers(0, len(files) - 1, batch_size)
        samples = np.zeros((batch_size, sample_size))
        for i, idx in enumerate(indices):
            frame = np.random.randint(0, len(audio_bytes[idx]) - sample_size)
            samples[i] = normalize_audio(
                audio_bytes[idx][frame:frame + sample_size])
        yield np.expand_dims(samples, -1), np.expand_dims(samples, -1)


def audio_segments_from_single_file(sample_size, path_to_file):
    audio_bytes, ff, sec = audio_to_array(
        path_to_file, path_to_file.split('.')[-1])
    audio_bytes = normalize_audio(audio_bytes)
    sample_num = int(ff * sec)
    return [audio_bytes[i:i + sample_size] for i in range(0, sample_num - sample_size, sample_size)]


def mel_spec_images_from_file(filename, output_folder):
    ### Parameters ###
    fft_size = 2048  # window size for the FFT
    step_size = 128  # distance to slide along the window (in time)
    # threshold for spectrograms (lower filters out more noise)
    spec_thresh = 4
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 10000  # Hz # High cut for our butter bandpass filter
    # For mels
    n_mel_freq_components = 64  # number of mel frequency channels
    shorten_factor = 4  # how much should we compress the x-axis (time)
    start_freq = 100  # Hz # What frequency to start sampling our melS from
    end_freq = 3000  # Hz # What frequency to stop sampling our melS from
    # Grab your wav and filter it
    # mywav = 'nocturne.wav'
    rate, data = wavfile.read(filename)
    data = butter_bandpass_filter(data, lowcut, highcut, rate, order=1)

    # iterating thru one second segments
    for i in range(0, len(data), rate):
        segment = data[i:i + rate]
        # create spec
        wav_spectrogram = pretty_spectrogram(segment.astype('float64'), fft_size=fft_size,
                                             step_size=step_size, log=True, thresh=spec_thresh)
        # Generate the mel filters
        mel_filter, mel_inversion_filter = create_mel_filter(fft_size=fft_size,
                                                             n_freq_components=n_mel_freq_components,
                                                             start_freq=start_freq,
                                                             end_freq=end_freq)
        # generate mel spectogram
        mel_spec = make_mel(wav_spectrogram, mel_filter,
                            shorten_factor=shorten_factor)
        # converting to the [0..1] domain
        mel_spec = mel_spec.astype(float) / spec_thresh + 1
        # saving as image
        cv2.imwrite(os.path.join(output_folder, 'segment_{0:04d}.png'.format(i // rate)), mel_spec * 255)


def audio_from_mel_spec(input_folder, filename):
    ### Parameters ###
    fft_size = 2048  # window size for the FFT
    step_size = 128  # distance to slide along the window (in time)
    # threshold for spectrograms (lower filters out more noise)
    spec_thresh = 4
    lowcut = 500  # Hz # Low cut for our butter bandpass filter
    highcut = 10000  # Hz # High cut for our butter bandpass filter
    # For mels
    n_mel_freq_components = 64  # number of mel frequency channels
    shorten_factor = 4  # how much should we compress the x-axis (time)
    start_freq = 100  # Hz # What frequency to start sampling our melS from
    end_freq = 3000  # Hz # What frequency to stop sampling our melS from

    files = glob.glob(os.path.join(input_folder, '*' + 'png'))
    combined = np.ndarray((n_mel_freq_components, 0), dtype=np.float16)

    # select only a couple segments
    for segment in files[:20]:
        segment_data = cv2.imread(segment, cv2.IMREAD_GRAYSCALE)
        # appending them together
        combined = np.append(combined, segment_data, axis=1)
    # visualize appended spectogram 
    cv2.imwrite('combined.png', combined.astype(float))
    # converting back to the original domain
    combined = (combined / 255. - 1) * spec_thresh
    # Generate the mel filters
    mel_filter, mel_inversion_filter = create_mel_filter(fft_size=fft_size,
                                                         n_freq_components=n_mel_freq_components,
                                                         start_freq=start_freq,
                                                         end_freq=end_freq)
    # generating spectogram from the mel spectogram
    mel_inverted_spectrogram = mel_to_spectrogram(combined, mel_inversion_filter,
                                                  spec_thresh=spec_thresh,
                                                  shorten_factor=shorten_factor)
    # inverting the spectogram to audio
    inverted_mel_audio = invert_pretty_spectrogram(np.transpose(mel_inverted_spectrogram), fft_size=fft_size,
                                                   step_size=step_size, log=True, n_iter=10)
    # multiplying with a magic number and exporting as wav
    wavfile.write(filename, 44100, inverted_mel_audio * 255)


for i, input_wav_file in enumerate(glob.glob(os.path.join('../in', '*' + 'wav'))):
    mel_spec_images_from_file(input_wav_file, output_folder='../out')

audio_from_mel_spec(input_folder='../out', filename='restored.wav')
