import glob
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "3"
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
from keras import backend as K
from keras import losses

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

from cv2 import imread
from skimage.transform import resize
import numpy as np
from matplotlib import pyplot as plt
from skimage.morphology import label
from pycocotools import mask as maskUtils
from tqdm import tqdm
import os
import cv2
from keras.layers import ReLU

input_shape = (512, 512, 3)

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


batchsize = 1

def batch(iterable, n=batchsize):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



test_files = glob.glob("./images/test/*.png")
try:
    os.mkdir("./Output/")
except:
    pass

model = get_unet(do=0.1, activation=ReLU)
model.load_weights("./models/model_save_name.hdf5")

for batch_files in tqdm(batch(test_files), total=len(test_files)//batchsize):
    imgs = [resize(imread(image_path)/255., input_shape) for image_path in batch_files]
    imgs = np.array(imgs)
    pred = model.predict(imgs)

    for i, image_path in enumerate(batch_files):
        pred_ = pred[i, :, :, :]
        pred_ = resize(pred_, (1200, 1600,3))
        pred_ = 255.*(pred_ - np.min(pred_))/(np.max(pred_)-np.min(pred_))
        image_base = image_path.split("/")[-1]
        cv2.imwrite("./Output/"+image_base, pred_, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
