# NTIRE-Single-Image-Dehazing-Challenge-CVPR-2019
NTIRE Workshop and Challenges @ CVPR 2019

This is my work at Challenge

We using the 3D-Unet for dehazing the images.  

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

The result is 

    hcilab	(date) 02/20/19	(PSNR)14.17	 (SSIM)0.48

To run the training code using command:

    python trainig.py

The output images should be in Output folder (will be generated when running the code)
