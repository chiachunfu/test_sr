import random
import glob
import subprocess
import os
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
from model import sr_resnet, sr_prosr_rcan,sr_discriminator, sr_gan_test, resnet_model, preprocess_resnet, vgg19_model, sr_resnet_test, sr_combine, sr_x2_check,sr_prosr_rcan_test,sr_x2_check2,sr_prosr_rcan_upsample, denoise_resnet
import re
from tensorflow.keras import backend as K


configProt = tf.ConfigProto()
configProt.gpu_options.allow_growth = True
configProt.allow_soft_placement = True

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 2000
config.batch_size = 32
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
scale = 8
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


def image_transform(img, type):
    if type == 0:
        pass
    elif type == 4:
        img = img.transpose(0)
    elif type == 5:
        img = img.transpose(1)
    else:
        img = img.rotate(90 * type)
    return img

def get_all_imgs(img_dir):
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    small_images = np.zeros(
        (len(input_filenames), config.input_width, config.input_height, 3))
    large_images = np.zeros(
        (len(input_filenames), config.output_width, config.output_height, 3))
    for i, f in enumerate(input_filenames):
        small_images[i] = np.array(Image.open(f)) / 255.0
        large_images[i] = np.array(Image.open(f.replace("-in.jpg","-out.jpg"))) / 255.0
    return (small_images, large_images)

def train_image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros(
            (batch_size, config.input_width*1, config.input_height*1, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        #carnation_cnt = 0
        carnation_check = 1
        while carnation_check:
            carnation_cnt = 0
            for i in range(batch_size):
                f = input_filenames[counter+i]
                if re.search('-carnation',f):
                    carnation_cnt += 1
            if carnation_cnt >= int(batch_size / 4):
                temp = input_filenames[counter:len(input_filenames)]
                random.shuffle(temp)
                input_filenames[counter:len(input_filenames)] = temp
            else:
                carnation_check = 0
        for i in range(batch_size):
            type = random.randint(0, 5) #augment option
            img = input_filenames[counter + i]
            #print(img)
            small_img = Image.open(img)
            small_img = image_transform(small_img, type)
            add_blur = np.random.choice(2,1,p=[0.75, 0.25])[0]
            if add_blur and 0:
                blur_radius = np.random.choice(4,1,p=[0.6, 0.25, 0.1, 0.05])[0] / 2 + 0.5
                small_img = small_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            small_images[i] = np.array(small_img) / 255.0
            img = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 1:
                if 'P' in img.mode:  # check if image is a palette type
                    img = img.convert("RGB")  # convert it to RGB
                    img = img.resize((config.input_width*1, config.input_height*1), Image.BICUBIC)  # resize it
                    large_image = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = img.resize((config.input_width*1, config.input_height*1), Image.BICUBIC)  # regular resize
            large_image = image_transform(large_image, type)

            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images)
        counter += batch_size

def train_image_generator_test(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros(
            (batch_size, config.input_width*scale, config.input_height*scale, 3))
        gtx1_images = np.zeros(
            (batch_size, config.input_width * 1, config.input_height * 1, 3))
        gtx2_images = np.zeros(
            (batch_size, config.input_width * 2, config.input_height * 2, 3))
        gtx4_images = np.zeros(
            (batch_size, config.input_width * 4, config.input_height * 4, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0
        #carnation_cnt = 0
        carnation_check = 1
        while carnation_check:
            carnation_cnt = 0
            for i in range(batch_size):
                f = input_filenames[counter+i]
                if re.search('-carnation',f):
                    carnation_cnt += 1
            if carnation_cnt >= int(batch_size / 4):
                temp = input_filenames[counter:len(input_filenames)]
                random.shuffle(temp)
                input_filenames[counter:len(input_filenames)] = temp
            else:
                carnation_check = 0
        for i in range(batch_size):
            type = random.randint(0, 5) #augment option
            img = input_filenames[counter + i]
            #print(img)
            small_img = Image.open(img)
            small_img = image_transform(small_img, type)
            small_images[i] = np.array(small_img) / 255.0
            img = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 1:
                if 'P' in img.mode:  # check if image is a palette type
                    img = img.convert("RGB")  # convert it to RGB
                    img = img.resize((config.input_width*scale, config.input_height*scale), Image.NEAREST)  # resize it
                    large_image = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = img.resize((config.input_width*scale, config.input_height*scale), Image.NEAREST)  # regular resize
            large_image = image_transform(large_image, type)
            gtx1 = large_image.resize((config.input_width*1, config.input_height*1), Image.NEAREST)
            gtx2 = large_image.resize((config.input_width*2, config.input_height*2), Image.NEAREST)
            gtx4 = large_image.resize((config.input_width*4, config.input_height*4), Image.NEAREST)
            large_images[i] = np.array(large_image) / 255.0
            gtx1_images[i] = np.array(gtx1) / 255.0
            gtx2_images[i] = np.array(gtx2) / 255.0
            gtx4_images[i] = np.array(gtx4) / 255.0
        yield (small_images, large_images, gtx1_images, gtx2_images, gtx4_images)
        counter += batch_size


class DataGenerator():
    def __init__(self, img_dir, is_train=False):
        self.input_filenames = glob.glob(img_dir + "/*-in.jpg")
        #print(len(self.input_filenames), img_dir)
        self.counter = 0
        self.total_file_cnt = len(self.input_filenames)
        self.is_train = is_train
        if self.is_train:
            random.shuffle(self.input_filenames)
    def batch_gen(self, batch_size):

        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros(
            (batch_size, config.input_width * scale, config.input_height * scale, 3))
        #random.shuffle(input_filenames)
        if self.counter + batch_size >= self.total_file_cnt:
            self.counter = 0
            if self.is_train:
                print("reset")
                random.shuffle(self.input_filenames)
        for i in range(batch_size):
            ttype = random.randint(0, 5) #augment option
            #print(self.counter)
            img = self.input_filenames[self.counter + i]
            #print(i, self.counter, img)
            small_img = Image.open(img)
            if self.is_train:
                small_img = image_transform(small_img, ttype)
            small_images[i] = np.array(small_img) / 255.0
            large_image = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 0:
                if 'P' in img.mode:  # check if image is a palette type
                    img = img.convert("RGB")  # convert it to RGB
                    img = img.resize((config.input_width*scale, config.input_height*scale), Image.ANTIALIAS)  # resize it
                    large_image = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = img.resize((config.input_width*scale, config.input_height*scale), Image.ANTIALIAS)  # regular resize
            if self.is_train:
                large_image = image_transform(large_image, ttype)

            large_images[i] = np.array(large_image) / 255.0
        #print(self.counter)
        self.counter += batch_size
        #yield (small_images, large_images)
        return (small_images, large_images)
    def shuffle(self):
        random.shuffle(self.input_filenames)
        self.counter = 0


def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros(
            (batch_size, config.input_width*1, config.input_height*1, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0

        for i in range(batch_size):
            type = random.randint(0, 5) #augment option
            img = input_filenames[counter + i]
            #print(img)
            small_img = Image.open(img)
            small_images[i] = np.array(small_img) / 255.0
            large_image = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 1:
                if 'P' in large_image.mode:  # check if image is a palette type
                    large_image = large_image.convert("RGB")  # convert it to RGB
                    large_image = large_image.resize((config.input_width*1, config.input_height*1), Image.BICUBIC)  # resize it
                    large_image = large_image.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = large_image.resize((config.input_width*1, config.input_height*1), Image.BICUBIC)  # regular resize
            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def image_generator_test(batch_size, img_dir, model):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        large_images = np.zeros(
            (batch_size, config.input_width*scale, config.input_height*scale, 3))
        random.shuffle(input_filenames)
        if counter+batch_size >= len(input_filenames):
            counter = 0

        for i in range(batch_size):
            type = random.randint(0, 5) #augment option
            img = input_filenames[counter + i]
            #print(img)
            small_img = Image.open(img)
            small_images[i] = np.array(small_img) / 255.0
            large_image = Image.open(img.replace("-in.jpg", "-out.jpg"))

            large_images[i] = np.array(large_image) / 255.0
        feat = model.predict(preprocess_resnet(large_images))
        yield (small_images, large_images, feat)
        counter += batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

def psnr(y_true, y_pred):
    """Calculate psnr"""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def psnr_v2(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303



class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(scale, axis=0).repeat(scale, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)

def LogImage(model,in_sample_images, out_sample_images):
    preds = model.predict(in_sample_images)
    in_resized = []
    for arr in in_sample_images:
        # Simple upsampling
        in_resized.append(arr.repeat(scale, axis=0).repeat(scale, axis=1))
    wandb.log({
        "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for
                     i, o in enumerate(preds)]
    }, commit=False)


#model = sr_prosr_rcan(input_shape=(config.input_width, config.input_height, 3),scale_ratio=scale)


#print(model.summary())


# Define custom loss
def custom_loss():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        r_del = K.abs(y_pred[:,:,:,0] - y_true[:,:,:,0])
        g_del = K.abs(y_pred[:,:,:,1] - y_true[:,:,:,1]) * 2.0
        b_del = K.abs(y_pred[:,:,:,2] - y_true[:,:,:,2])
        #return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
        return K.mean(r_del+g_del+b_del)

    # Return a function
    return loss

def lr_scheduler(lr, epoch, decay_freq=50, decay_factor=0.5):
    pow = epoch // decay_freq
    lr_new = lr * (decay_factor ** pow)
    return max(1e-7, lr_new)


# Gradients of output wrt model weights
def get_gradients(model, weights_list, input_imgs):
    gradients = K.gradients(model.output, weights_list)
    #print(gradients)
    # Wrap the model input tensor and the gradient tensors in a callable function
    f = K.function([model.input], gradients)

    ## Random input image
    #x = np.random.rand(1,100,100,3)

    # Call the function to get the gradients of the model output produced by this image, wrt the model weights
    return f([input_imgs])

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad



def perceptual_distance_np(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[ :, :, 0] + y_pred[ :, :, 0]) / 2
    r = y_true[ :, :, 0] - y_pred[ :, :, 0]
    g = y_true[ :, :, 1] - y_pred[ :, :, 1]
    b = y_true[ :, :, 2] - y_pred[ :, :, 2]

    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

#for l in model.layers:
    #print(type(l))
val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)
if 1:
    model = sr_prosr_rcan_upsample(input_shape=(config.input_width, config.input_height, 3),scale_ratio=1)

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)
    print(model.summary())
    model = load_model('best_denoise.h5')
    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer=opt, loss='mae',
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator1 = image_generator(config.batch_size, val_dir)
    #in_sample_images, out_sample_images = next(val_generator)

    checkpoint = ModelCheckpoint('best_denoise.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=1000, callbacks=[
                        #epochs = config.num_epochs, callbacks = [
                        checkpoint],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator1)
