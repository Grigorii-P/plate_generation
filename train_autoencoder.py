import numpy as np
from os.path import join
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers, optimizers, losses
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


path_npy = '/home/grigorii/Desktop/plates_generator/npy'


def load_data():
    imgs_target = np.load(join(path_npy, 'data.npy'))
    imgs_input = np.load(join(path_npy, 'data_temp.npy'))
    return imgs_target, imgs_input


X, Y = load_data()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
input_dim = len(X[1])

encoding_1 = 1000
encoding_2 = 50

input_ = Input(shape=(input_dim,))
# encoded = Dense(encoding_1, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(input_)
# encoded = Dense(encoding_2, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(encoded)
encoded = Dense(encoding_1, activation='sigmoid')(input_)
encoded = Dense(encoding_2, activation='sigmoid')(encoded)
decoded = Dense(encoding_1, activation='sigmoid')(encoded)
decoded = Dense(input_dim, activation='sigmoid',)(decoded)

autoencoder = Model(input_, decoded)
plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=True)
model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
autoencoder.compile(optimizer=optimizers.Adam(lr=1e-3), loss=losses.binary_crossentropy)

e = 15
batch = 16
autoencoder.fit(x_train, y_train,
                epochs=e,
                batch_size=batch,
                shuffle=True,
                validation_data=(x_test, y_test),
                callbacks=[model_checkpoint])
