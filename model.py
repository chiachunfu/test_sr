
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from layers import resnet_layer, SubpixelConv2D

def sr_resnet(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 32
    reg_scale = 0
    #scale_ratio = 2
    num_filters_out = max(64, 3 * scale_ratio**2)
    inputs = Input(shape=input_shape)

    x = Conv2D(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(reg_scale)
               )(inputs)
    num_res_layer = 8

    def res_blocks(res_in, num_chans):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans
                         )
        return add([res_in, x])

    for l in range(num_res_layer):

        x = resnet_layer(inputs=x,
                         num_filters=num_filters * 2,
                         )



    pixelshuf_in = Conv2D(num_filters_out,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(reg_scale)
                         )(x)



    #res_out2 = layers.add([res_in2, x])

    pixelshuf_skip_in = Conv2D(num_filters_out,
                               kernel_size=3,
                               strides=1,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=l2(reg_scale)
                               )(inputs)
    up_samp = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                             scale=scale_ratio
                             )(pixelshuf_in)
    up_samp_skip = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                             scale=scale_ratio
                                  )(pixelshuf_skip_in)
    outputs = add([up_samp, up_samp_skip])

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model



