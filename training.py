from model.autoencoder import conv_ae_1d

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

model = conv_ae_1d(input_shape=(16384, 1), encoder_setup=encoder_setup, decoder_setup=decoder_setup)
print(model.summary())
