import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf

configProt = tf.ConfigProto()
configProt.gpu_options.allow_growth = True
configProt.allow_soft_placement = True

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 50
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        #large_images = np.zeros(
        #    (batch_size, config.input_width*2, config.input_height*2, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            img = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 0:
                if 'P' in img.mode:  # check if image is a palette type
                    img = img.convert("RGB")  # convert it to RGB
                    img = img.resize((config.input_width*2, config.input_height*2), Image.ANTIALIAS)  # resize it
                    img = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    img = img.resize((config.input_width*2, config.input_height*2), Image.ANTIALIAS)  # regular resize
            large_images[i] = np.array(img) / 255.0
        yield (small_images, large_images)
        counter += batch_size

def SubpixelConv2D(input_shape, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return layers.Lambda(subpixel, output_shape=subpixel_shape, name='subpixel')

def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=False,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    #feature depth transformation

    filter_max = 64
    #bottleneck


    x = inputs
    if num_filters <=filter_max:
        conv = layers.Conv2D(num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             # kernel_regularizer=l2(1e-4)
                             )
        conv2 = layers.Conv2D(num_filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding='same',
                              kernel_initializer='he_normal',
                              # kernel_regularizer=l2(1e-4)
                              )
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = layers.Activation(activation)(x)
            x = conv2(x)
        else:
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = layers.Activation(activation)(x)
            x = conv2(x)
    else:
        conv_bot_in = layers.Conv2D(filter_max,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    # kernel_regularizer=l2(1e-4)
                                    )
        conv_bot_out = layers.Conv2D(num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     # kernel_regularizer=l2(1e-4)
                                     )
        conv = layers.Conv2D(filter_max,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer='he_normal',
                             # kernel_regularizer=l2(1e-4)
                             )
        conv2 = layers.Conv2D(filter_max,
                              kernel_size=kernel_size,
                              strides=strides,
                              padding='same',
                              kernel_initializer='he_normal',
                              # kernel_regularizer=l2(1e-4)
                              )
        if conv_first:
            x = conv_bot_in(x)
            if batch_normalization:
                x = layers.BatchNormalization()(x)
            if activation is not None:
                x = layers.Activation(activation)(x)
            x = conv2(x)
            x = conv_bot_out(x)


    return x



def sr_resnet_simp(input_shape):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 32
    scale_ratio = 8
    num_filters_out = 3 * scale_ratio**2
    inputs = layers.Input(shape=input_shape)

    res_in = layers.Conv2D(num_filters,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         #kernel_regularizer=l2(1e-4)
                         ) (inputs)
    x = resnet_layer(inputs=res_in,
                     num_filters=num_filters,
                     conv_first=True)
    res_out = layers.add([res_in, x])

    res_in2 = layers.Conv2D(num_filters_out,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         #kernel_regularizer=l2(1e-4)
                         )(res_out)
    x = resnet_layer(inputs=res_in2,
                     num_filters=num_filters_out,
                     conv_first=True)




    res_out2 = layers.add([res_in2, x])
    res_in3 = layers.Conv2D(num_filters_out,
                            kernel_size=3,
                            strides=1,
                            padding='same',
                            kernel_initializer='he_normal',
                            # kernel_regularizer=l2(1e-4)
                            )(inputs)
    res_out3 = layers.add([res_in3, res_out2])
    up_samp = SubpixelConv2D([None, config.input_width, config.input_height, num_filters_out], scale=8)(res_out3)
    outputs = layers.Conv2D(3,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         #kernel_regularizer=l2(1e-4)
                             )(up_samp)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


#model = Sequential()

#model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
#                        input_shape=(config.input_width, config.input_height, 3)))
#model.add(layers.Conv2D(32, (3, 3), padding='same',
#                        input_shape=(config.input_width, config.input_height, 32)))\

#model.add(SubpixelConv2D([None, config.input_width, config.input_height,32],scale=2))

#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#model.add(layers.UpSampling2D())
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#model.add(layers.UpSampling2D())
#model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))

model = sr_resnet_simp(input_shape=(config.input_width, config.input_height, 3))


#print(model.summary())
if 1:
    opt = tf.keras.optimizers.Adam(lr=0.001)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss='mse',
                  metrics=[perceptual_distance])

    model.fit_generator(image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=config.num_epochs, callbacks=[
                            ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)
