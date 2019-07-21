from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Lambda, add, multiply,GlobalAveragePooling2D,Dense, LeakyReLU
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform

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


def SubpixelConv2D(input_shape, scale=8,name='subp'):
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


    return Lambda(subpixel, output_shape=subpixel_shape,name=name)



def BicubicUpscale(scale):
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


    def bicubic(input_img):
        w = input_img.get_shape()[1]
        h = input_img.get_shape()[2]
        return tf.image.resize_bicubic(input_img, [w*scale, h*scale])

        #return tf.image.resize_bicubic(x, output_shape)


    return Lambda(bicubic)

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
            x = LeakyReLU(alpha=0.01)(x)
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
            x = LeakyReLU(alpha=0.01)(x)
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



def icnr_weights(init = tf.glorot_normal_initializer(), scale=2, shape=[3,3,32,4], dtype = tf.float32):
    sess = tf.Session()
    return sess.run(ICNR(init, scale=scale)(shape=shape, dtype=dtype))

class ICNR:
    """ICNR initializer for checkerboard artifact free sub pixel convolution
    Ref:
     [1] Andrew Aitken et al. Checkerboard artifact free sub-pixel convolution
     https://arxiv.org/pdf/1707.02937.pdf)
    Args:
    initializer: initializer used for sub kernels (orthogonal, glorot uniform, etc.)
    scale: scale factor of sub pixel convolution
    """

    def __init__(self, initializer, scale=1):
        self.scale = scale
        self.initializer = initializer

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.scale == 1:
            return self.initializer(shape)

        new_shape = shape[:3] + [shape[3] // (self.scale ** 2)]
        x = self.initializer(new_shape, dtype, partition_info)
        x = tf.transpose(x, perm=[2, 0, 1, 3])
        x = tf.image.resize_nearest_neighbor(x, size=(shape[0] * self.scale, shape[1] * self.scale))
        x = tf.space_to_depth(x, block_size=self.scale)
        x = tf.transpose(x, perm=[1, 2, 0, 3])

        return x


class Subpixel(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 r,
                 padding='valid',
                 data_format=None,
                 strides=(1,1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=r*r*filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.r = r

    def _phase_shift(self, I):
        r = self.r
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize, a, b, int(c/(r*r)),r, r]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (unshifted[0], self.r*unshifted[1], self.r*unshifted[2], int(unshifted[3]/(self.r*self.r)))

    def get_config(self):
        config = super(Conv2D, self).get_config()
        #config.pop('rank')
        #config.pop('dilation_rate')
        config['filters']= int(config['filters'] / self.r*self.r)
        config['r'] = self.r
        return config
