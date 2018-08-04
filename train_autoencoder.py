import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import regularizers, optimizers, losses
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from experiments import path_npy, resize, num_train, import_images_train_valid, \
    create_images, create_npy, load_npy, generator


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    set_session(sess)


def train():
    # X, Y = load_npy()
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    # input_dim = len(X[1])

    input_dim = resize[0] * resize[1]

    encoding_1 = 500
    encoding_2 = 50

    input_ = Input(shape=(input_dim,))
    #TODO add regularization
    # encoded = Dense(encoding_1, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(input_)
    # encoded = Dense(encoding_2, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(encoded)
    encoded = Dense(encoding_1, activation='sigmoid')(input_)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(encoding_2, activation='sigmoid')(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(encoding_1, activation='sigmoid')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim, activation='sigmoid',)(decoded)

    autoencoder = Model(input_, decoded)
    plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=True)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    autoencoder.compile(optimizer=optimizers.Adam(lr=1e-3), loss=losses.binary_crossentropy)

    images_train, _, images_dict = import_images_train_valid()
    create_images()
    create_npy()
    validation_inp, validation_trg = load_npy()

    e = 20
    batch = 512

    # autoencoder.fit(x_train, y_train,
    #                 epochs=e,
    #                 batch_size=batch,
    #                 shuffle=True,
    #                 validation_data=(x_test, y_test),
    #                 callbacks=[model_checkpoint])

    autoencoder.fit_generator(generator(batch_size=batch, images_train=images_train, images_dict=images_dict),
                              steps_per_epoch=round(num_train / batch),
                              epochs=e,
                              shuffle=True,
                              validation_data=(validation_inp, validation_trg),
                              verbose=1,
                              callbacks=[model_checkpoint])

    print('\nDONE\n')


if __name__ == '__main__':
    tf_config()
    train()
