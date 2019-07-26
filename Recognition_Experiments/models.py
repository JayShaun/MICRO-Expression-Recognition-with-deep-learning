from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, ZeroPadding2D, \
    LSTM, UpSampling2D, Permute, Reshape, Lambda, RepeatVector, concatenate, Input
from tensorflow.python.keras import backend as K

def VGG_16_5_channels(spatial_size=224, classes=5, channels=5, channel_first=False, weights_path=None):
    model = Sequential()

    if channel_first:
        model.add(ZeroPadding2D((1, 1), input_shape=(channels, spatial_size, spatial_size)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(spatial_size, spatial_size, channels)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 33

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))  # 34
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))  # 35
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax'))  # Dropped

    if weights_path:
        model.load_weights(weights_path)
    model.pop()
    model.add(Dense(classes, activation='softmax'))  # 36

    return model


def VGG_16(spatial_size, classes, channels, channel_first=True, weights_path=None):
    model = Sequential()
    if channel_first:
        model.add(ZeroPadding2D((1, 1), input_shape=(channels, spatial_size, spatial_size)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(spatial_size, spatial_size, channels)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 33

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))  # 34
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))  # 35
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax'))  # Dropped

    if weights_path:
        model.load_weights(weights_path)
    model.pop()
    model.add(Dense(classes, activation='softmax'))  # 36

    return model

def attention_3d_block(inputs, SINGLE_ATTENTION_VECTOR = False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    TIME_STEPS = int(inputs.shape[1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = concatenate([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def temporal_module_with_attention(data_dim, timesteps_TIM=10, classes=5, weights_path=None):
    inputs = Input(shape=(timesteps_TIM, data_dim,))
    attention_mul = attention_3d_block(inputs)
    attention_mul = LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim))(attention_mul)
    output = Dense(128, activation='relu')(attention_mul)
    output = Dense(classes, activation='sigmoid')(output)
    model = Model(inputs=[inputs], outputs=output)

    if weights_path:
        model.load_weights(weights_path)

    return model

def temporal_module(data_dim, timesteps_TIM=10, classes=5, weights_path=None):
    model = Sequential()
    model.add(LSTM(3000, return_sequences=False, input_shape=(timesteps_TIM, data_dim)))
    # model.add(LSTM(3000, return_sequences=False))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes, activation='sigmoid'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def convolutional_autoencoder(spatial_size, channel_first=True):
    model = Sequential()

    # encoder
    if channel_first:
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(3, spatial_size, spatial_size), padding='same'))
    else:
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(spatial_size, spatial_size, 3), padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    # decoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(2))
    model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

    return model


def VGG_16_tim(spatial_size, classes, channels, channel_first=True, weights_path=None):
    model = Sequential()
    if channel_first:
        model.add(ZeroPadding2D((1, 1), input_shape=(channels, spatial_size, spatial_size)))
    else:
        model.add(ZeroPadding2D((1, 1), input_shape=(spatial_size, spatial_size, channels)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 33

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))  # 34
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))  # 35
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax'))  # Dropped

    if weights_path:
        model.load_weights(weights_path)
    model.pop()
    model.add(Dense(classes, activation='softmax'))  # 36

    return model