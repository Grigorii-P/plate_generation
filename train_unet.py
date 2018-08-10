import tensorflow as tf
from os.path import join
import cv2
from time import time
import numpy as np
from utils.unet import unet_small
from keras.callbacks import Callback, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from data_processing import path_npy, create_images, create_npy_unet


path_weights = '/ssd480/grisha/plates_generation/unet_generation.h5'
path_intermediate_result = '/ssd480/grisha/plates_generation/res_during_training_unet'
callback_img = 'img_callback.jpg'

num_train = 30000


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        input = self.test_data
        prediction = self.model.predict(input)
        prediction = np.multiply(prediction, 255.)
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
    # Part for IN-MEMORY training (without generator)
    create_npy_unet()
    # X = np.load(join(path_npy, 'input_unet.npy'))
    # Y = np.load(join(path_npy, 'target_unet.npy'))
    # X = X[:num_train]
    # Y = Y[:num_train]
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)
    #
    # img_callback = cv2.imread(callback_img)
    #
    # print('Creating and compiling model...')
    # model = unet_small()
    # model_checkpoint = ModelCheckpoint(path_weights, monitor='val_loss', save_best_only=True)
    #
    # batch = 64
    # epochs = 8
    # print('Fitting model...')
    # t0 = time()
    # model.fit(x_train, y_train,
    #           epochs=epochs,
    #           batch_size=batch,
    #           shuffle=True,
    #           validation_data=(x_test, y_test),
    #           callbacks=[model_checkpoint, TestCallback(img_callback)])


if __name__ == '__main__':
    tf_config()
    train()
