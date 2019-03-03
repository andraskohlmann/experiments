from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import LSTM, Reshape, TimeDistributed, Conv2D, Conv2DTranspose


def encoder(input_layer, encoder_setup, trainable=True):
    x = input_layer
    for filters, kernel_size, strides in encoder_setup:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', padding='same',
                   trainable=trainable)(x)
    return x


def decoder(input_layer, decoder_setup, trainable=True):
    x = input_layer
    for filters, kernel_size, upsampling in decoder_setup:
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=upsampling, activation='relu',
                            padding='same', trainable=trainable)(x)
    return x


def conv_ae_2d(input_tensor, encoder_setup, decoder_setup):
    encoded = encoder(input_tensor, encoder_setup)
    encoded = Conv2D(4, 3, padding='same')(encoded)
    decoded = decoder(encoded, decoder_setup)
    decoded = Conv2D(1, 1, activation='sigmoid', padding='same')(decoded)
    return decoded


def encoder_td(input_layer, encoder_setup, trainable=True):
    x = input_layer
    for filters, kernel_size, strides in encoder_setup:
        x = TimeDistributed(
            Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation='relu', padding='same', ),
            trainable=trainable)(x)
    return x


def conv_lstm_ae_1d(timesteps, input_shape, encoder_setup, lstm_num, decoder_setup):
    input_layer = Input(shape=(timesteps, *input_shape))
    encoded = encoder_td(input_layer, encoder_setup, trainable=False)
    encoded = TimeDistributed(Conv2D(4, 3, padding='same'), trainable=False)(encoded)
    times = encoded._keras_shape[1]
    encoded_shape = encoded._keras_shape[2:]
    units = encoded_shape[0] * encoded_shape[1] * encoded_shape[2]
    lstm = Reshape((times, units))(encoded)
    for i in range(lstm_num):
        lstm = LSTM(units=units, return_sequences=True if i < lstm_num - 1 else False)(lstm)
    lstm = Reshape(encoded_shape)(lstm)
    decoded = decoder(lstm, decoder_setup, trainable=False)
    decoded = Conv2D(1, 1, activation='sigmoid', padding='same', trainable=False)(decoded)
    return decoded
