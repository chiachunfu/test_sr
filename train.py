import random
import glob
import subprocess
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from model import sr_resnet, sr_prosr_rcan,sr_discriminator, sr_gan_test, resnet_model, preprocess_resnet
import re


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
            (batch_size, config.input_width*scale, config.input_height*scale, 3))
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
                    img = img.resize((config.input_width*scale, config.input_height*scale), Image.ANTIALIAS)  # resize it
                    large_image = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = img.resize((config.input_width*scale, config.input_height*scale), Image.ANTIALIAS)  # regular resize
            large_image = image_transform(large_image, type)

            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images)
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
        #if self.counter + batch_size >= self.total_file_cnt:
        #    self.counter = 0
        #    if self.is_train:
        #        print("reset")
        #random.shuffle(self.input_filenames)
        fidx = np.random.randint(len(self.input_filenames),size=config.batch_size)  # augment option
        #print(len(self.input_filenames))
        for i in range(batch_size):
            ttype = random.randint(0, 5) #augment option
            #print(self.counter)
            img = self.input_filenames[fidx[i]]
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
        #self.counter += batch_size
        yield (small_images, large_images)
        #return (small_images, large_images)


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
        yield (small_images, large_images)
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
#for l in model.layers:
    #print(type(l))
val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)
if 0:
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)



    model.fit_generator(train_image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=config.num_epochs, callbacks=[
                            ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)

else:
    #all_train_input_imgs, all_train_output_imgs = get_all_imgs(train_dir)
    all_val_input_imgs, all_val_output_imgs = get_all_imgs(val_dir)
    if 1:
        res = resnet_model(input_shape=(config.output_width, config.output_height, 3))
        res.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                              , loss='mse'
                              , metrics=['accuracy'])

    discriminator = sr_discriminator(input_shape=(config.output_width, config.output_height, 3))
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                          , loss='binary_crossentropy'
                          , metrics=['accuracy'])

    generator = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)
    generator.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])




    gan = sr_gan_test((config.input_width, config.input_height, 3), generator, discriminator,res)
    gan.compile(
        loss=['binary_crossentropy', 'mse'],
        loss_weights=[1, 1],
        optimizer=tf.keras.optimizers.Adam(lr=0.0002,decay=0.9)
    )
    ## train generator for a couple of epochs
    if 0:
        generator.fit_generator(train_image_generator(config.batch_size, train_dir),
                                steps_per_epoch=config.steps_per_epoch,
                                epochs=5, callbacks=[
                ImageLogger(), WandbCallback()],
                                validation_steps=config.val_steps_per_epoch,
                                validation_data=val_generator)
        generator.save_weights("trained_generator_{}X_epoch{}.h5".format(scale, 0))

    #print(gan.summary())
    #print(discriminator.summary())
    train_generator = DataGenerator(train_dir,is_train=True)
    val_generator = DataGenerator(val_dir,is_train=False)
    all_dis_loss = []
    all_gen_loss = []
    all_gen_mae_loss = []
    all_gen_dis_loss = []
    itr_for_disc = 1
    for itr in range(config.num_epochs * config.steps_per_epoch):
        input_imgs, output_imgs = next(train_generator.batch_gen(config.batch_size))
        gen_imgs = generator.predict(input_imgs)
        #discriminator.trainable = True
        real_img_loss = discriminator.train_on_batch(output_imgs, np.ones(config.batch_size)*0.9)
        fake_img_loss = discriminator.train_on_batch(gen_imgs, np.zeros(config.batch_size)*0.1)
        dis_loss = 0.5 * np.add(real_img_loss,fake_img_loss)
        #print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss)
        #discriminator.trainable = False
        if itr > itr_for_disc:
            if itr == itr_for_disc+1:
                discriminator.save_weights("trained_discriminator_{}X_epoch{}.h5".format(scale, 0))
                print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss)

            input_imgs, output_imgs = next(train_generator.batch_gen(config.batch_size))

            real_feat = res.predict(preprocess_resnet(output_imgs))
            gen_loss = gan.train_on_batch(input_imgs,[np.ones(config.batch_size), real_feat])
            #print("gan loss: ", gen_loss)
            all_dis_loss.append(dis_loss[0])
            all_gen_loss.append(gen_loss[0])
            all_gen_mae_loss.append(gen_loss[2])
            all_gen_dis_loss.append(gen_loss[1])
            if (itr+1) % 1 == 32:

                print(itr, np.mean(np.array(all_dis_loss)), np.mean(np.array(all_gen_loss)), np.mean(np.array(all_gen_mae_loss)), np.mean(np.array(all_gen_dis_loss)))
                all_dis_loss = []
                all_gen_loss = []
                all_gen_mae_loss = []
                all_gen_dis_loss = []

            if (itr+1) % 512 == 0 or (itr + 1) < itr_for_disc+10:
                results = generator.evaluate(input_imgs, output_imgs, config.batch_size)

                #print("train performance", generator.evaluate(all_train_input_imgs, all_train_output_imgs, config.batch_size))
                results = generator.evaluate(all_val_input_imgs, all_val_output_imgs, config.batch_size)
                print("val performance", results)
                #LogImage(generator, in_sample_images, out_sample_images)
                #wandb.log(results)
                if (itr + 1) % (3 * 1000) == 0:
                    """Save the generator and discriminator networks"""
                    generator.save_weights("trained_new_generator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))
                    discriminator.save_weights("trained_new_discriminator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))

        elif (itr + 1) % 100 == 0:
            print(itr, "real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss)
            #results = discriminator.evaluate(gen_imgs, output_imgs, config.batch_size)
            #print("val performance", results)

