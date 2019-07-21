
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Input, add, Lambda, Dense, Flatten, LeakyReLU, UpSampling2D
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Sequential, Model
from layers import resnet_layer, SubpixelConv2D, Conv2DWeightNorm, attention_layer, BicubicUpscale, icnr_weights, Subpixel
from tensorflow.keras import backend as K
from tensorflow.keras.applications import ResNet50, VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import re

def sr_resnet(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    #scale_ratio = 2
    num_filters_out = max(64, 3 * scale_ratio**2)
    inputs = Input(shape=input_shape)
    #test_initializer = RandomUniform(minval=-0.005, maxval=0.005,seed=None)
    test_initializer = 'he_normal'
    x = Conv2DWeightNorm(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer=test_initializer,
               kernel_regularizer=l2(reg_scale)
               )(inputs)
    num_res_layer = 8

    def res_blocks(res_in, num_chans):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         kernel_initializer=test_initializer
                         )
        return x

    def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         kernel_initializer=test_initializer
                         )
        x = attention_layer(x, 4)
        return x

    for l in range(num_res_layer):

        #x = res_blocks(x,num_filters)
        x = res_chan_attention_blocks(x,num_filters,4)

    #print(type(x))
    num_filters2 = 256

    x = Conv2DWeightNorm(256,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=test_initializer,
                         kernel_regularizer=l2(reg_scale)
                         )(x)
    for l in range(0):

        #x = res_blocks(x,num_filters)
        x = res_chan_attention_blocks(x,num_filters2,16)


    pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                                    kernel_initializer=test_initializer,
                                    kernel_regularizer=l2(reg_scale)
                         )(x)

    up_samp = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                             scale=scale_ratio,
                             name='sub_1'

                             )(pixelshuf_in)

    #res_out2 = layers.add([res_in2, x])
    if 1:
        pixelshuf_skip_in = Conv2DWeightNorm(num_filters_out,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                             kernel_initializer=test_initializer,
                                             kernel_regularizer=l2(reg_scale)
                                   )(inputs)

        up_samp_skip = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                                 scale=scale_ratio,
                                  name='sub_2'
                                      )(pixelshuf_skip_in)
    else:
        up_samp_skip = BicubicUpscale(8)(inputs)
    res_scale = 0.2
    if res_scale >= 0:
        up_samp = Lambda(lambda x: x * res_scale)(up_samp)
    outputs = add([up_samp, up_samp_skip])
    #outputs = up_samp_skip

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def sr_prosr_rcan(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    scale_ratio = 2
    #num_filters_out = max(64, 3 * scale_ratio**2)
    num_filters_out = 3 * 2**2
    inputs = Input(shape=input_shape)
    x_in = inputs
    for i in range(3):
        x = Conv2DWeightNorm(num_filters,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg_scale)
                   )(x_in)
        num_res_layer = 8

        def res_blocks(res_in, num_chans):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            return x

        def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            x = attention_layer(x, 4)
            return x

        for l in range(num_res_layer):

            #x = res_blocks(x,num_filters)
            x = res_chan_attention_blocks(x,num_filters,4)

        pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=l2(reg_scale)
                                        )(x)
        up_samp = SubpixelConv2D([None, pixelshuf_in.get_shape()[1], pixelshuf_in.get_shape()[2], num_filters_out],
                                 scale=scale_ratio,
                                 name='subp_'+str(i)
                                 )(pixelshuf_in)
        #up_samp = Subpixel(3, 3, 2, padding='same')(pixelshuf_in)

        #print(up_samp.get_shape())
        up_samp_skip = BicubicUpscale(2**(i+1))(inputs)

        x_in = add([up_samp, up_samp_skip], name="out_"+str(i))

    outputs = x_in

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            if scale_ratio == 3:
                w = icnr_weights(scale=3, shape=c.shape)
            else:
                w = icnr_weights(scale=2, shape=c.shape)
            layer.set_weights([w, b])
    return model


def sr_prosr_rcan_test(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    scale_ratio = 2
    #num_filters_out = max(64, 3 * scale_ratio**2)
    num_filters_out = 3 * 2**2
    inputs = Input(shape=input_shape)
    x_in = inputs
    for i in range(1):
        x = Conv2DWeightNorm(num_filters,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg_scale)
                   )(x_in)
        num_res_layer = 8

        def res_blocks(res_in, num_chans):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            return x

        def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            x = attention_layer(x, 4)
            return x

        for l in range(num_res_layer):

            #x = res_blocks(x,num_filters)
            x = res_chan_attention_blocks(x,num_filters,4)

        pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=l2(reg_scale)
                                        )(x)
        up_samp = SubpixelConv2D([None, pixelshuf_in.get_shape()[1], pixelshuf_in.get_shape()[2], num_filters_out],
                                 scale=scale_ratio,
                                 name='subp_'+str(i)
                                 )(pixelshuf_in)
        #up_samp = Subpixel(3, 3, 2, padding='same')(pixelshuf_in)

        #print(up_samp.get_shape())
        up_samp_skip = BicubicUpscale(2**(i+1))(inputs)

        x_in = add([up_samp, up_samp_skip], name="out_"+str(i))

    outputs = x_in

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            if scale_ratio == 3:
                w = icnr_weights(scale=3, shape=c.shape)
            else:
                w = icnr_weights(scale=2, shape=c.shape)
            layer.set_weights([w, b])
    return model

def sr_prosr_rcan_upsample(input_shape,scale_ratio):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    scale_ratio = 2
    #num_filters_out = max(64, 3 * scale_ratio**2)
    num_filters_out = 3
    inputs = Input(shape=input_shape)
    x_in = inputs
    for i in range(3):
        x = Conv2DWeightNorm(num_filters,
                   kernel_size=3,
                   strides=1,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg_scale)
                   )(x_in)
        num_res_layer = 8

        def res_blocks(res_in, num_chans):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            return x

        def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
            x = resnet_layer(inputs=res_in,
                             num_filters=num_chans
                             )
            x = attention_layer(x, 4)
            return x

        for l in range(num_res_layer):

            #x = res_blocks(x,num_filters)
            x = res_chan_attention_blocks(x,num_filters,4)

        pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                                        kernel_size=3,
                                        strides=1,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        kernel_regularizer=l2(reg_scale)
                                        )(x)
        up_samp = UpSampling2D(pixelshuf_in)
        #up_samp = Subpixel(3, 3, 2, padding='same')(pixelshuf_in)

        #print(up_samp.get_shape())
        up_samp_skip = BicubicUpscale(2**(i+1))(inputs)

        x_in = add([up_samp, up_samp_skip])

    outputs = x_in

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    for layer in model.layers:
        if type(layer) == Subpixel:
            c, b = layer.get_weights()
            if scale_ratio == 3:
                w = icnr_weights(scale=3, shape=c.shape)
            else:
                w = icnr_weights(scale=2, shape=c.shape)
            layer.set_weights([w, b])
    return model


def sr_discriminator(input_shape, num_filters=32):
    inputs = Input(shape=input_shape)
    filter_scale = [1, 2, 4, 8, 4]
    x = inputs
    for ratio in filter_scale:
        x = Conv2D(num_filters * ratio,
                   kernel_size=3,
                   strides=2,
                   padding='same',
                   kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
    x = Dense(num_filters)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    outputs = Activation('sigmoid')(x)
    #outputs = Dense(1, activation='')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def sr_gan_test(input_shape, gen_model, dis_model,resnet_model):
    inputs = Input(shape=input_shape)
    gen_out = gen_model(inputs)
    gen_feat = resnet_model(preprocess_resnet(gen_out))
    #print(gen_model.layers)
    dis_model.trainable=False

    dis_out = dis_model(gen_out)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[dis_out, gen_feat, gen_out])
    return model

def resnet_model(input_shape):
    inputs = Input(shape=input_shape)
    # Get the vgg network. Extract features from last conv layer
    res = ResNet50(weights="imagenet")
    res.outputs = [res.layers[51].input]
    #print(res.layers[51].input.shape)
    # Create model and compile
    model = Model(inputs=inputs, outputs=res(inputs))
    model.trainable = False
    return model

def vgg19_model(input_shape):
    inputs = Input(shape=input_shape)
    # Get the vgg network. Extract features from last conv layer
    #vgg = VGG19(weights="imagenet",include_top=False,input_shape=input_shape)
    vgg = VGG19(weights="imagenet")
    vgg.outputs = [vgg.layers[20].output]
    print(vgg.layers[20].input.shape)
    # Create model and compile
    model = Model(inputs=inputs, outputs=vgg(inputs))
    model.trainable = False
    return model

def preprocess_resnet(x):
    """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
    if isinstance(x, np.ndarray):
        return preprocess_input((x)*255.)
    else:
        return Lambda(lambda x: preprocess_input(x * 255.0))(x)


def sr_resnet_test(input_shape,scale_ratio,resnet_model):
    #inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    num_filters = 64
    reg_scale = 0
    #scale_ratio = 2
    num_filters_out = max(64, 3 * scale_ratio**2)
    inputs = Input(shape=input_shape)
    #test_initializer = RandomUniform(minval=-0.005, maxval=0.005,seed=None)
    test_initializer = 'he_normal'
    x = Conv2DWeightNorm(num_filters,
               kernel_size=3,
               strides=1,
               padding='same',
               kernel_initializer=test_initializer,
               kernel_regularizer=l2(reg_scale)
               )(inputs)
    num_res_layer = 8

    def res_blocks(res_in, num_chans):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         kernel_initializer=test_initializer
                         )
        return x

    def res_chan_attention_blocks(res_in, num_chans, reduction_ratio):
        x = resnet_layer(inputs=res_in,
                         num_filters=num_chans,
                         kernel_initializer=test_initializer
                         )
        x = attention_layer(x, 4)
        return x

    for l in range(num_res_layer):

        #x = res_blocks(x,num_filters)
        x = res_chan_attention_blocks(x,num_filters,4)

    #print(type(x))
    num_filters2 = 256

    x = Conv2DWeightNorm(256,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         kernel_initializer=test_initializer,
                         kernel_regularizer=l2(reg_scale)
                         )(x)
    for l in range(0):

        #x = res_blocks(x,num_filters)
        x = res_chan_attention_blocks(x,num_filters2,16)


    pixelshuf_in = Conv2DWeightNorm(num_filters_out,
                         kernel_size=3,
                         strides=1,
                         padding='same',
                                    kernel_initializer=test_initializer,
                                    kernel_regularizer=l2(reg_scale)
                         )(x)

    up_samp = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                             scale=scale_ratio
                             )(pixelshuf_in)

    #res_out2 = layers.add([res_in2, x])
    if 0:
        pixelshuf_skip_in = Conv2DWeightNorm(num_filters_out,
                                   kernel_size=3,
                                   strides=1,
                                   padding='same',
                                             kernel_initializer=test_initializer,
                                             kernel_regularizer=l2(reg_scale)
                                   )(inputs)

        up_samp_skip = SubpixelConv2D([None, input_shape[0], input_shape[1], num_filters_out],
                                 scale=scale_ratio
                                      )(pixelshuf_skip_in)
    else:
        up_samp_skip = BicubicUpscale(8)(inputs)
    res_scale = 0.2
    if res_scale >= 0:
        up_samp = Lambda(lambda x: x * res_scale)(up_samp)
    outputs = add([up_samp, up_samp_skip])
    gen_feat = resnet_model(preprocess_resnet(outputs))
    #outputs = up_samp_skip
    # Instantiate model.
    model = Model(inputs=inputs, outputs=[outputs, gen_feat])
    return model

def sr_combine(input_shape, gen_model, resnet_model):
    inputs = Input(shape=input_shape)
    gen_out = gen_model(inputs)
    gen_feat = resnet_model(preprocess_resnet(gen_out))

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[ gen_out, gen_feat])
    return model




def sr_x2_check(input_shape, gen_x2, gen_x4, gen_x8):
    inputs = Input(shape=input_shape)
    gen_out = gen_x2(inputs)
    gen_out2 = gen_x4(gen_out)
    gen_out3 = gen_x8(gen_out2)

        #print(i, layer.name, layer.output)
    #intermediate_layer_model = Model(inputs=x,
    #                                 outputs=gen_model.get_layer("out_0").output)
    #intermediate_output = intermediate_layer_model.predict(x,steps=1)
    model = Model(inputs=inputs, outputs=[gen_out3, gen_out, gen_out2])
    #model = Model(inputs=inputs, outputs=gen_out)
    return model


def sr_x2_check_val(input_shape, gen_x2, gen_x4, gen_x8):
    inputs = Input(shape=input_shape)
    gen_out = gen_x2(inputs)
    gen_out2 = gen_x4(gen_out)
    gen_out3 = gen_x8(gen_out2)

        #print(i, layer.name, layer.output)
    #intermediate_layer_model = Model(inputs=x,
    #                                 outputs=gen_model.get_layer("out_0").output)
    #intermediate_output = intermediate_layer_model.predict(x,steps=1)
    model = Model(inputs=inputs, outputs=gen_out3)
    #model = Model(inputs=inputs, outputs=gen_out)
    return model