import tensorflow as tf
from os.path import join
import cv2
import numpy as np
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model
from keras import regularizers, optimizers, losses
from keras.utils import plot_model
from keras.callbacks import Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from data_processing import path_npy, resize, num_train, import_images_train_valid, \
     create_dataset, load_npy, generator


path_intermediate_result = '/ssd480/grisha/plates_generation/result_during_training'


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    set_session(sess)


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        input = self.test_data
        prediction = self.model.predict(input)
        prediction = np.reshape(prediction, (resize[0], resize[1]))
        prediction = np.multiply(prediction, 255.)
        prediction = np.array(prediction, dtype=np.uint8)
        cv2.imwrite(join(path_intermediate_result, 'result_on_epoch_' + str(epoch) + '.jpg'), prediction)


def get_callback_img():
    # callback image (one sample only)
    img_callback = cv2.imread('img_callback_backward.jpg', 0)
    img_callback = cv2.resize(img_callback, (resize[1], resize[0]))
    img_callback = np.divide(img_callback, 255.)
    img_callback = img_callback.flatten()
    img_callback = np.reshape(img_callback, (1, img_callback.shape[0]))
    img_callback = np.array(img_callback, dtype=np.float32)
    return img_callback


def train():
    create_dataset()

    # Part for IN-MEMORY training (without generator)
    Y, X = load_npy()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

    input_dim = resize[0] * resize[1]

    encoding_1 = 1000
    encoding_2 = 500
    encoding_3 = 50

    input_ = Input(shape=(input_dim,))
    # TODO add regularization
    # encoded = Dense(encoding_1, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(input_)
    # encoded = Dense(encoding_2, activation='sigmoid', activity_regularizer=regularizers.l2(1e-5))(encoded)
    encoded = Dense(encoding_1, activation='sigmoid')(input_)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(encoding_2, activation='sigmoid')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(encoding_3, activation='sigmoid')(encoded)
    encoded = BatchNormalization()(encoded)

    decoded = Dense(encoding_2, activation='sigmoid')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(encoding_1, activation='sigmoid')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(input_dim, activation='sigmoid',)(decoded)

    e = 40
    batch = 512
    lr = 1e-4

    autoencoder = Model(input_, decoded)
    plot_model(autoencoder, to_file='model.png', show_shapes=True, show_layer_names=True)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)
    autoencoder.compile(optimizer=optimizers.Adam(lr=lr), loss=losses.binary_crossentropy)
    img_callback = get_callback_img()

    autoencoder.fit(x_train, y_train,
                    epochs=e,
                    batch_size=batch,
                    shuffle=True,
                    validation_data=(x_test, y_test),
                    callbacks=[model_checkpoint, TestCallback(img_callback)])

    # TODO add EarlyStopping callback
    # TODO add LearningRateScheduler or ReduceLROnPlateau callback
    # Part for OUT-MEMORY training (with generator)
    # images_train, _, images_dict = import_images_train_valid()
    # validation_inp, validation_trg = load_npy()

    # autoencoder.fit_generator(generator(batch_size=batch, images_train=images_train, images_dict=images_dict),
    #                           steps_per_epoch=round(num_train / batch),
    #                           epochs=e,
    #                           shuffle=True,
    #                           validation_data=(validation_inp, validation_trg),
    #                           verbose=1,
    #                           callbacks=[model_checkpoint, TestCallback(img_callback)])
    print('\nDONE\n')


if __name__ == '__main__':
    tf_config()
    train()
