
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, add, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from layers import resnet_layer, SubpixelConv2D, Conv2DWeightNorm, attention_layer

def sr_resnet(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    #scale_ratio = 2
    num_filters_out = max(64, 3 * scale_ratio**2)
    inputs = Input(shape=input_shape)

    x = Conv2DWeightNorm(num_filters,
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
        return x

    def res_scaling_blocks(res_in, num_chans):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         res_scale=0.2
                         )
        return x

    def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         )
        x = attention_layer(x, reduction_ratio)
        return x

    for l in range(num_res_layer):

        #x = res_blocks(x,num_filters)
        x = res_chan_attention_blocks(x,num_filters,4)

    #print(type(x))
    num_filters2 = 256
    if 1:
        x = Conv2DWeightNorm(256,
                             kernel_size=1,
                             strides=1,
                             padding='same',
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(reg_scale)
                             )(x)
        for l in range(4):

            #x = res_blocks(x,num_filters)
            x = res_chan_attention_blocks(x,num_filters2,4)


    pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                         kernel_initializer='he_normal',
                         kernel_regularizer=l2(reg_scale)
                         )(x)



    #res_out2 = layers.add([res_in2, x])

    pixelshuf_skip_in = Conv2DWeightNorm(num_filters_out,
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



