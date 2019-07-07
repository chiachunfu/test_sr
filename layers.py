from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Lambda, add, multiply,GlobalAveragePooling2D,Dense
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K

class Conv2DWeightNorm(tf.layers.Conv2D):

  def build(self, input_shape):
    self.wn_g = self.add_weight(
        name='wn_g',
        shape=(self.filters,),
        dtype=self.dtype,
        initializer=tf.initializers.ones,
        trainable=True,
    )
    super(Conv2DWeightNorm, self).build(input_shape)
    square_sum = tf.reduce_sum(
        tf.square(self.kernel), [0, 1, 2], keepdims=False)
    inv_norm = tf.rsqrt(square_sum)
    self.kernel = self.kernel * (inv_norm * self.wn_g)


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
    #print(inputs.get_shape())
    if num_filters <= filter_max:
        conv = Conv2DWeightNorm(num_filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                             )
        conv2 = Conv2DWeightNorm(num_filters,
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
        conv_bot_in = Conv2DWeightNorm(filter_max,
                             kernel_size=1,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                                    )
        conv_bot_out = Conv2DWeightNorm(num_filters,
                              kernel_size=1,
                              strides=strides,
                              padding='same',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=l2(reg_scale)
                                     )
        conv = Conv2DWeightNorm(filter_max,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding='same',
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(reg_scale)
                             )
        conv2 = Conv2DWeightNorm(filter_max,
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

    x = add([inputs, x])

    return x

def attention_layer(input, reduction_ratio):
    """
    :param input: input tensor
    :param reduction_ratio: reduction ratio for w1
    :return: output tensor
    """
    input_chan_num = int(input.get_shape()[-1])
    #print(input_chan_num)
    #print(type(input_chan_num))
    #print(tf.divide(input_chan_num, reduction_ratio))
    gap = GlobalAveragePooling2D()(input)
    gap = K.expand_dims(gap,1)
    gap = K.expand_dims(gap,1)

    #print(gap.get_shape())
    down_scale = Dense(int(input_chan_num/reduction_ratio),
                        activation='relu',
                       use_bias=False
                        )(gap)
    up_scale = Dense(input_chan_num ,
                        activation='sigmoid',
                     use_bias=False,
                     )(down_scale)
    up_scale = tf.squeeze(up_scale, [1, 2])
    x = multiply([input, up_scale])

    return input
