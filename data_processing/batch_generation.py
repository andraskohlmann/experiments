import cv2
import glob
import os
import numpy as np
import tensorflow as tf


# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _sampling_function(resolution):
    def crop_and_resize_function(spec):
        crop_size = [spec.shape[0], spec.shape[0]]
        sampled = tf.image.random_crop(spec, crop_size)
        resized = tf.image.resize(sampled, resolution)
        return resized
    return crop_and_resize_function


# Parsing thru a folder, reading in all spectograms, then generating random samples from them
def random_images_from_spectograms(spec_folder, resolution, batch_size):
    all_combined_spectograms = np.concatenate([
        cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        for filename
        in glob.glob(os.path.join(spec_folder, '*.png'))
    ], axis=1)
    dataset = tf.data.Dataset.from_tensor_slices(all_combined_spectograms)
    dataset = dataset.map(_sampling_function(resolution)).batch(batch_size).repeat()
    return dataset
