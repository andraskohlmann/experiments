import glob
import os
import shutil

import cv2
import numpy as np
from keras.utils import Sequence
from scipy.io import wavfile

from preprocessing.audio_import import (audio_to_array, get_files_from_library,
                                        normalize_audio)
from preprocessing.spectogram import pretty_spectrogram, create_mel_filter, make_mel, mel_to_spectrogram, \
    invert_pretty_spectrogram


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


def mel_spec_images_from_file(filename, output_folder, segment_seconds=None):
    rate, data = wavfile.read(filename)
    if segment_seconds:
        # iterating thru sec second segments
        for i in range(0, len(data), rate * segment_seconds):
            segment = data[i:i + rate * segment_seconds]
            mel_spec = mel_spec_images_from_segment(segment)
            cv2.imwrite(os.path.join(output_folder, 'segment_{0:04d}.png'.format(i // rate)), mel_spec * 255)
    else:
        # whole data
        mel_spec = mel_spec_images_from_segment(data)
        cv2.imwrite(os.path.join(output_folder, 'segment_ALL.png'), mel_spec * 255)


def mel_spec_images_from_segment(segment):
    fft_size = 4096  # window size for the FFT
    step_size = 164  # distance to slide along the window (in time)
    # threshold for spectrograms (lower filters out more noise)
    spec_thresh = 4
    n_mel_freq_components = 128  # number of mel frequency channels
    shorten_factor = 16  # how much should we compress the x-axis (time)
    start_freq = 28  # Hz # What frequency to start sampling our melS from
    end_freq = 4000  # Hz # What frequency to stop sampling our melS from
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
    return mel_spec.astype(float) / spec_thresh + 1


def audio_from_mel_spec(input_folder, filename):
    fft_size = 4096  # window size for the FFT
    step_size = 164  # distance to slide along the window (in time)
    # threshold for spectrograms (lower filters out more noise)
    spec_thresh = 4
    n_mel_freq_components = 128  # number of mel frequency channels
    shorten_factor = 16  # how much should we compress the x-axis (time)
    start_freq = 28  # Hz # What frequency to start sampling our melS from
    end_freq = 4000  # Hz # What frequency to stop sampling our melS from

    files = glob.glob(os.path.join(input_folder, '*' + 'png'))
    combined = np.ndarray((n_mel_freq_components, 0), dtype=np.float16)

    # select only a couple segments
    for segment in files:
        segment_data = cv2.imread(segment, cv2.IMREAD_GRAYSCALE)
        segment_data = segment_data[:, :512]
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


def generate_images_from_audio_files(folder_name):
    if os.path.exists(os.path.join('out', folder_name)):
        shutil.rmtree(os.path.join('out', folder_name))
        os.makedirs(os.path.join('out', folder_name))
    for input_wav_file in glob.glob(os.path.join('in', '*' + 'wav')):
        audio_name = os.path.split(input_wav_file)[-1].split('.')[0]
        os.makedirs(os.path.join('out', folder_name, audio_name))
        mel_spec_images_from_file(input_wav_file, output_folder=os.path.join('out', folder_name, audio_name))


def restore_audio_from_images(input_folder_name, output_folder_name):
    if not os.path.exists(os.path.join('out', output_folder_name)):
        os.makedirs(os.path.join('out', output_folder_name))
    for in_dir in glob.glob(os.path.join('out', input_folder_name, '*', '')):
        audio_name = os.path.split(os.path.split(in_dir)[0])[-1]
        audio_from_mel_spec(input_folder=in_dir,
                            filename=os.path.join('out', output_folder_name, audio_name + '_restored.wav'))


class ImageSequence(Sequence):
    def __init__(self, image_folders, batch_size):
        self.files = sorted(glob.glob(os.path.join(image_folders, '*.png')))
        self.batch_size = batch_size
        self.images = [cv2.imread(self.files[i], cv2.IMREAD_GRAYSCALE) / 255. for i in range(len(self.files))]

    def __len__(self):
        return len(self.files) * 100 // self.batch_size

    def random_crop(self, image):
        h, w = image.shape[0:2]
        start = np.random.randint(0, w - h)
        return image[:, start:start + h]

    def __getitem__(self, index):
        images = np.expand_dims(np.array([self.random_crop(self.images[i % len(self.files)]) for i in
                                          range(index * self.batch_size, (index + 1) * self.batch_size)]), -1)
        return images, images
