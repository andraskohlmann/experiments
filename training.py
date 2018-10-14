from preprocessing.generator import audio_segment_generator

encoder_setup = [
    (64, (5), 4),
    (128, (5), 4),
    (256, (5), 4),
    (256, (5), 4),
    (512, (5), 4)
]

decoder_setup = [
    (512, (5), 4),
    (256, (5), 4),
    (256, (5), 4),
    (128, (5), 4),
    (64, (5), 4)
]

# model = conv_ae_1d(input_shape=(16384, 1), encoder_setup=encoder_setup, decoder_setup=decoder_setup)
# print(model.summary())

b = audio_segment_generator(1000, 2, '.', 'mp3')
d = next(b)
print(d.shape)
