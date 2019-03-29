from glob import glob
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from keras import backend as K
from keras import losses

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "3"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, MaxPooling2D
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout, LeakyReLU, PReLU, ReLU
from keras.models import Model
from keras.activations import relu
from keras.optimizers import Adam
from numpy import random
from sklearn.model_selection import KFold
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import math

input_shape = (512, 512, 3)


batch_size = 8
import cv2

def custom_activation(x):
    return K.relu(x, alpha=0.0, max_value=1)

smooth = 1.

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * math.log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true)))) 

# Define our custom loss function
def charbonnier(y_true, y_pred):
    epsilon = 1e-3
    error = y_true - y_pred
    p = K.sqrt(K.square(error) + K.square(epsilon))
    return K.mean(p)



def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return 10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def get_unet(do=0, activation=ReLU):
    inputs = Input(input_shape)
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(inputs)))
    conv1 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(conv1)))
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(pool1)))
    conv2 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv2)))
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(pool2)))
    conv3 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(conv3)))
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(pool3)))
    conv4 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(conv4)))
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(pool4)))
    conv5 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv5)))
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    conv6 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(pool5)))
    conv6 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv6)))
    pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
    
    conv7 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(pool6)))
    conv7 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv7)))
    pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)

    
    conv8 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(pool7)))
    conv8 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv8)))

    
    up9 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv8), conv7], axis=3)
    conv9 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(up9)))
    conv9 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv9)))
    
    up10 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv9), conv6], axis=3)
    conv10 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(up10)))
    conv10 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv10)))
    
    up11 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv10), conv5], axis=3)
    conv11 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(up11)))
    conv11 = Dropout(do)(activation()(Conv2D(512, (3, 3), padding='same')(conv11)))
    

    up12 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv11), conv4], axis=3)
    conv12 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(up12)))
    conv12 = Dropout(do)(activation()(Conv2D(256, (3, 3), padding='same')(conv12)))

    up13 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv12), conv3], axis=3)
    conv13 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(up13)))
    conv13 = Dropout(do)(activation()(Conv2D(128, (3, 3), padding='same')(conv13)))

    up14 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv13), conv2], axis=3)
    conv14 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(up14)))
    conv14 = Dropout(do)(activation()(Conv2D(64, (3, 3), padding='same')(conv14)))

    up15 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv14), conv1], axis=3)
    conv15 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(up15)))
    conv15 = Dropout(do)(activation()(Conv2D(32, (3, 3), padding='same')(conv15)))

    conv16 = Dropout(do)(Conv2D(3, (1, 1), activation='sigmoid')(conv15))

    model = Model(inputs=[inputs], outputs=[conv16])

    model.compile(optimizer=Adam(lr=1e-4), loss=losses.mse , metrics=[PSNRLoss])

    model.summary()

    return model




def read_input(path):
    x = resize(cv2.imread(path)/255., input_shape)
    return np.asarray(x)

def read_gt(path):
    x = resize(cv2.imread(path)/255., input_shape)
    return np.asarray(x)

def gen(data):
    while True:
        # choose random index in features
        # try:
        index= random.choice(list(range(len(data))), batch_size)
        index = list(map(int, index))
        list_images_base = [read_input(data[i][0]) for i in index]
        list_gt_base = [read_gt(data[i][1]) for i in index]
        yield np.array(list_images_base), np.array(list_gt_base)
        # except Exception as e:
        #     print(e)





if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dropout", required=False,
                    help="dropout", type=float, default=0)
    ap.add_argument("-a", "--activation", required=False,
                    help="activation", default="ReLu")

    args = vars(ap.parse_args())


    activation = globals()[args['activation']]

    model_name = "./models/{epoch:04d}-{val_loss:.6f}_Unet_dropout_%s_activation_%s_"%(args['dropout'], args['activation'])

    print("Model : %s"%model_name)

    train_data = list(zip(sorted(glob('./images/train/*.png')), sorted(glob('./images/trainGT/*.png'))))
    val_data = list(zip(sorted(glob('./images/valid/*.png')), sorted(glob('./images/validGT/*.png'))))

    print(len(val_data)//batch_size, len(val_data), batch_size)


    model = get_unet(do=args['dropout'], activation=activation)

    file_path = model_name + "weights.best.hdf5"
    try:
        model.load_weights("./models/model_save_name.hdf5")
        print("loading model ...")
    except:
        pass


    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=30, verbose=1)
    callbacks_list = [checkpoint]  # early

    history = model.fit_generator(gen(train_data), validation_data=gen(val_data), epochs=100, verbose=1,
                         callbacks=callbacks_list, steps_per_epoch= len(train_data),
                                  validation_steps=len(val_data), use_multiprocessing=False, workers=16)




