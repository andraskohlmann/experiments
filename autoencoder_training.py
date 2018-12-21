import glob
import os.path
import shutil

import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint

from model.autoencoder import conv_ae_2d
from preprocessing.generator import ImageSequence, \
    restore_audio_from_images, generate_images_from_audio_files

encoder_setup = [
    (128, 3, 2),
    (256, 3, 1),
    (256, 3, 2),
    (128, 3, 2)
]

decoder_setup = [
    (128, 3, 2),
    (256, 3, 2),
    (256, 3, 1),
    (128, 3, 2)
]

# generate_images_from_audio_files('segments')
# restore_audio_from_images('segments', 'restored')

model = conv_ae_2d(input_shape=(128, 128, 1), encoder_setup=encoder_setup, decoder_setup=decoder_setup)
print(model.summary())

batch_size = 16
image_generator = ImageSequence(os.path.join('out', 'segments', '*'), batch_size=batch_size)
# d = image_generator[0]
# print(d[0].shape)

model.compile(optimizer='adam', loss='mse')

checkpoint_path = os.path.join('checkpoints', 'epoch-{epoch:02d}-{loss:.4f}.hdf5')
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min')

# model.fit_generator(image_generator, epochs=20, callbacks=[])
# model.save_weights(os.path.join('weights', 'model.h5'))

model.load_weights(os.path.join('weights', 'model.h5'))

validation_samples = glob.glob(os.path.join('out', 'segments', '*', '*.png'))

if os.path.exists(os.path.join('out', 'predictions')):
    shutil.rmtree(os.path.join('out', 'predictions'))

for i, s in enumerate(validation_samples):
    spec = cv2.imread(s, cv2.IMREAD_GRAYSCALE) / 255.
    spec_shape = spec.shape
    print(spec_shape)
    preds = np.zeros((*spec_shape, 1))
    for window in range(0, spec_shape[1] - spec_shape[0], spec_shape[0]):
        preds[:, window:window + spec_shape[0]] = model.predict(np.expand_dims(
            np.expand_dims(spec[:, window:window + spec_shape[0]], -1), 0))

    os.makedirs(os.path.join('out', 'predictions', '{:02d}'.format(i)))
    cv2.imwrite(os.path.join('out', 'predictions', '{:02d}'.format(i), '{0:04d}.png'.format(i)), np.squeeze(preds * 255))

restore_audio_from_images('predictions', 'predictions_restored')
