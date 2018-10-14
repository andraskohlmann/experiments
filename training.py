import numpy as np

from model.autoencoder import conv_ae_1d
from preprocessing.generator import audio_segment_generator, audio_segments_from_single_file

encoder_setup = [
    (64, (5), 4),
    (128, (5), 4),
    (256, (5), 4),
    (256, (5), 2),
    (512, (5), 2)
]

decoder_setup = [
    (512, (5), 2),
    (256, (5), 2),
    (256, (5), 4),
    (128, (5), 4),
    (64, (5), 4)
]

model = conv_ae_1d(input_shape=(16384, 1), encoder_setup=encoder_setup, decoder_setup=decoder_setup)
print(model.summary())

b = audio_segment_generator(1000, 2, '.', 'mp3')
d = next(b)
print(d.shape)

validation_samples = audio_segments_from_single_file(16384, 'example.mp3')

for s in validation_samples:
    pred = model.predict(np.expand_dims(np.expand_dims(s, -1), 0))
