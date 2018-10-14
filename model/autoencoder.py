from keras import Input, Model
from keras.layers import Conv1D, UpSampling1D, Flatten, LSTM, Reshape


def encoder(input_layer, encoder_setup):
    x = input_layer
    for filters, kernel_size, strides in encoder_setup:
        x = Conv1D(filters=filters, kernel_size=(kernel_size), strides=strides, activation='relu', padding='same')(x)
    return x


def decoder(input_layer, decoder_setup):
    x = input_layer
    for filters, kernel_size, upsampling in decoder_setup:
        x = Conv1D(filters=filters, kernel_size=(kernel_size), activation='relu', padding='same')(x)
        x = UpSampling1D(size=upsampling)(x)
    return x


def conv_ae_1d(input_shape, encoder_setup, decoder_setup):
    input_layer = Input(shape=input_shape)
    encoded = encoder(input_layer, encoder_setup)
    encoded = Conv1D(1, 3, padding='same')(encoded)
    decoded = decoder(encoded, decoder_setup)
    decoded = Conv1D(1, (1), activation='sigmoid', padding='same')(decoded)

    ae = Model(input_layer, decoded)
    return ae


def conv_lstm_ae_1d(input_shape, encoder_setup, lstm_num, decoder_setup):
    assert encoder_setup.shape[1] == 3
    assert decoder_setup.shape[1] == 3

    input_layer = Input(shape=input_shape)
    encoded = encoder(input_layer, encoder_setup)
    encoded = Flatten()(encoded)
    units = encoded.output_shape

    lstm = encoded
    for _ in range(lstm_num):
        lstm = LSTM(units=units)(lstm)

    lstm = Reshape((units, 1))(lstm)
    decoded = decoder(lstm, decoder_setup)
    decoded = Conv1D(1, (1), activation='sigmoid', padding='same')(decoded)

    ae = Model(input_layer, decoded)
    return ae
