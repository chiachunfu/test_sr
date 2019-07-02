from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def SubpixelConv2D(input_shape, scale=8):
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
    def subpixel_shape():
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape)



def resnet_layer(inputs,
                 num_filters=16,
                 filter_max=64,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=False,
                 weight_normalization=False,
                 kernel_initializer='he_normal',
                 reg_scale=0
                 ):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters, input/output have same number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """

    input_chan_num = inputs.get_shape()[-1]
    if num_filters <= filter_max:
        conv = Conv2D(num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                             )
        conv2 = Conv2D(num_filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(reg_scale)
                              )
        x = conv(inputs)
        if batch_normalization:
            x = BatchNormalization()(x)
        elif weight_normalization:
            pass
        if activation is not None:
            x = Activation(activation)(x)
        x = conv2(x)

    else: ## bottleneck structure
        conv_bot_in = Conv2D(filter_max,
                             kernel_size=1,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                                    )
        conv_bot_out = Conv2D(num_filters,
                              kernel_size=1,
                              strides=strides,
                              padding='same',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(reg_scale)
                                     )
        conv = Conv2D(filter_max,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                             )
        conv2 = Conv2D(filter_max,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       kernel_initializer=kernel_initializer,
                       kernel_regularizer=l2(reg_scale)
                              )
        x = conv_bot_in(inputs)
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        elif weight_normalization:
            pass
        if activation is not None:
            x = Activation(activation)(x)
        x = conv2(x)
        x = conv_bot_out(x)


    return x



