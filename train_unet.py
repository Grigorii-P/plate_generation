import tensorflow as tf
from os.path import join
import cv2
from time import time
import numpy as np
from utils.unet import unet_small, img_cols, img_rows
from keras.callbacks import Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from data_processing import printing, path_npy, create_images, create_npy_unet


path_weights = '/ssd480/grisha/plates_generation/unet_generation.h5'
path_intermediate_result = '/ssd480/grisha/plates_generation/res_during_training_unet'
callback_img = 'img_callback.jpg'

num_train = 200000


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        input = self.test_data
        input = cv2.resize(input, (img_cols, img_rows))
        input = input[np.newaxis, ..., np.newaxis]
        printing('input shape - {}'.format(input.shape))
        prediction = self.model.predict(input)
        prediction = np.multiply(prediction, 255.)
        prediction = np.reshape(prediction, (img_rows, img_cols))
        prediction = np.array(prediction, dtype=np.uint8)
        cv2.imwrite(join(path_intermediate_result, 'result_on_epoch_' + str(epoch) + '.jpg'), prediction)


def tf_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    config.gpu_options.visible_device_list = "0"
    sess = tf.Session(config=config)
    set_session(sess)


def train():
    create_images()
    create_npy_unet()
    X = np.load(join(path_npy, 'input_unet.npy'))
    Y = np.load(join(path_npy, 'target_unet.npy'))
    X = X[:num_train]
    Y = Y[:num_train]

    x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size=0.05, random_state=42)

    # printing('x_train shape - {}'.format(x_train.shape))
    img_callback = cv2.imread(callback_img, 0)
    printing('img_callback shape - {}'.format(img_callback.shape))

    print('Creating and compiling model...')
    model = unet_small()
    model_checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True)

    batch = 128
    epochs = 8
    print('Fitting model...')
    t0 = time()
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch,
              shuffle=True,
              validation_data=(x_test, y_test),
              callbacks=[model_checkpoint, TestCallback(img_callback)])
    printing('Training time - {:.1f}'.format((time()-t0)/60))


if __name__ == '__main__':
    tf_config()
    train()
