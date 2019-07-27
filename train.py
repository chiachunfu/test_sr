import random
import glob
import subprocess
import os
from PIL import Image, ImageFilter
import numpy as np
import tensorflow as tf
from model import sr_resnet, sr_prosr_rcan,sr_discriminator, sr_gan_test, resnet_model, preprocess_resnet, vgg19_model, sr_resnet_test, sr_resnet_bilin, sr_combine, sr_x2_check,sr_prosr_rcan_test,sr_x2_check2,sr_prosr_rcan_upsample, sr_resnet84, dbpn,sr_resnet_simp,sr_x2_concat
import re
from tensorflow.keras import backend as K


configProt = tf.ConfigProto()
configProt.gpu_options.allow_growth = True
configProt.allow_soft_placement = True

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres')
config = run.config

config.num_epochs = 500
config.batch_size = 16
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256
scale = 4
val_dir = 'data/test'
train_dir = 'data/train'
train_test_dir = 'data/train_new2'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size


def img_augmentation():
    print('augmenting...')
    import shutil
    shutil.copytree('./data/train', './data/train_new')
    input_filenames = glob.glob('./data/train_new' + "/*-in.jpg")
    for f in input_filenames:
        small_img = Image.open(f)
        small_img_rot90 = small_img.rotate(90)
        small_img_rot180 = small_img.rotate(180)
        small_img_rot270 = small_img.rotate(270)
        small_img_rot90_flip0 = small_img_rot90.transpose(0)
        small_img_rot90_flip1 = small_img_rot90.transpose(1)
        small_img_rot180_flip0 = small_img_rot180.transpose(0)
        small_img_rot180_flip1 = small_img_rot180.transpose(1)
        small_img_rot270_flip0 = small_img_rot270.transpose(0)
        small_img_rot270_flip1 = small_img_rot270.transpose(1)

        small_img_rot90.save(f.replace('-in.jpg', '1-in.jpg'))
        small_img_rot180.save(f.replace('-in.jpg', '2-in.jpg'))
        small_img_rot270.save(f.replace('-in.jpg', '3-in.jpg'))
        small_img_rot90_flip0.save(f.replace('-in.jpg', '4-in.jpg'))
        small_img_rot90_flip1.save(f.replace('-in.jpg', '5-in.jpg'))
        small_img_rot180_flip0.save(f.replace('-in.jpg', '6-in.jpg'))
        small_img_rot180_flip1.save(f.replace('-in.jpg', '7-in.jpg'))
        small_img_rot270_flip0.save(f.replace('-in.jpg', '8-in.jpg'))
        small_img_rot270_flip1.save(f.replace('-in.jpg', '9-in.jpg'))

        f_l = f.replace('-in.jpg', '-out.jpg')
        large_img = Image.open(f_l)
        large_img_rot90 = large_img.rotate(90)
        large_img_rot180 = large_img.rotate(180)
        large_img_rot270 = large_img.rotate(270)
        large_img_rot90_flip0 = large_img_rot90.transpose(0)
        large_img_rot90_flip1 = large_img_rot90.transpose(1)
        large_img_rot180_flip0 = large_img_rot180.transpose(0)
        large_img_rot180_flip1 = large_img_rot180.transpose(1)
        large_img_rot270_flip0 = large_img_rot270.transpose(0)
        large_img_rot270_flip1 = large_img_rot270.transpose(1)

        large_img_rot90.save(f_l.replace('-out.jpg', '1-out.jpg'))
        large_img_rot180.save(f_l.replace('-out.jpg', '2-out.jpg'))
        large_img_rot270.save(f_l.replace('-out.jpg', '3-out.jpg'))
        large_img_rot90_flip0.save(f_l.replace('-out.jpg', '4-out.jpg'))
        large_img_rot90_flip1.save(f_l.replace('-out.jpg', '5-out.jpg'))
        large_img_rot180_flip0.save(f_l.replace('-out.jpg', '6-out.jpg'))
        large_img_rot180_flip1.save(f_l.replace('-out.jpg', '7-out.jpg'))
        large_img_rot270_flip0.save(f_l.replace('-out.jpg', '8-out.jpg'))
        large_img_rot270_flip1.save(f_l.replace('-out.jpg', '9-out.jpg'))


def img_augmentation2():
    print('augmenting...')
    import shutil
    shutil.copytree('./data/train', './data/train_new2')
    input_filenames = glob.glob('./data/train_new2' + "/*-in.jpg")
    for f in input_filenames:
        small_img = Image.open(f)
        lr = Image.open(f)
        # lr_rs = lr.resize((256,256))
        hr = Image.open(f.replace("-in.jpg", "-out.jpg"))
        lr_rs = lr.resize((256, 256), Image.BILINEAR)
        pd_lr_rs = perceptual_distance_np(np.array(hr, dtype='float32'), np.array(lr_rs, dtype='float32'))
        if pd_lr_rs > 90:
            continue

        small_img_rot90 = small_img.rotate(90)
        small_img_rot180 = small_img.rotate(180)
        small_img_rot270 = small_img.rotate(270)
        small_img_rot90_flip0 = small_img_rot90.transpose(0)
        small_img_rot90_flip1 = small_img_rot90.transpose(1)
        small_img_rot180_flip0 = small_img_rot180.transpose(0)
        small_img_rot180_flip1 = small_img_rot180.transpose(1)
        small_img_rot270_flip0 = small_img_rot270.transpose(0)
        small_img_rot270_flip1 = small_img_rot270.transpose(1)

        small_img_rot90.save(f.replace('-in.jpg', '1-in.jpg'))
        small_img_rot180.save(f.replace('-in.jpg', '2-in.jpg'))
        small_img_rot270.save(f.replace('-in.jpg', '3-in.jpg'))
        small_img_rot90_flip0.save(f.replace('-in.jpg', '4-in.jpg'))
        small_img_rot90_flip1.save(f.replace('-in.jpg', '5-in.jpg'))
        small_img_rot180_flip0.save(f.replace('-in.jpg', '6-in.jpg'))
        small_img_rot180_flip1.save(f.replace('-in.jpg', '7-in.jpg'))
        small_img_rot270_flip0.save(f.replace('-in.jpg', '8-in.jpg'))
        small_img_rot270_flip1.save(f.replace('-in.jpg', '9-in.jpg'))

        f_l = f.replace('-in.jpg', '-out.jpg')
        large_img = Image.open(f_l)
        large_img_rot90 = large_img.rotate(90)
        large_img_rot180 = large_img.rotate(180)
        large_img_rot270 = large_img.rotate(270)
        large_img_rot90_flip0 = large_img_rot90.transpose(0)
        large_img_rot90_flip1 = large_img_rot90.transpose(1)
        large_img_rot180_flip0 = large_img_rot180.transpose(0)
        large_img_rot180_flip1 = large_img_rot180.transpose(1)
        large_img_rot270_flip0 = large_img_rot270.transpose(0)
        large_img_rot270_flip1 = large_img_rot270.transpose(1)

        large_img_rot90.save(f_l.replace('-out.jpg', '1-out.jpg'))
        large_img_rot180.save(f_l.replace('-out.jpg', '2-out.jpg'))
        large_img_rot270.save(f_l.replace('-out.jpg', '3-out.jpg'))
        large_img_rot90_flip0.save(f_l.replace('-out.jpg', '4-out.jpg'))
        large_img_rot90_flip1.save(f_l.replace('-out.jpg', '5-out.jpg'))
        large_img_rot180_flip0.save(f_l.replace('-out.jpg', '6-out.jpg'))
        large_img_rot180_flip1.save(f_l.replace('-out.jpg', '7-out.jpg'))
        large_img_rot270_flip0.save(f_l.replace('-out.jpg', '8-out.jpg'))
        large_img_rot270_flip1.save(f_l.replace('-out.jpg', '9-out.jpg'))



def image_transform_rot_flip(img, rot_type, flip_type):
    if rot_type == 0:
        pass
    else:
        img = img.rotate(90 * rot_type)

    if flip_type == 0:
        pass
    elif flip_type == 1:
        img = img.transpose(0)
    elif flip_type == 2:
        img = img.transpose(1)

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
            #rot_type = random.randint(0, 3) #augment option
            #flip_type = random.randint(0, 2) #augment option
            #flip_type = random.randint(0, 2) #augment option
            img = input_filenames[counter + i]
            #color_shuffle = random.randint(0, 1) #augment option
            #is_syn = np.random.choice(2,1,p=[0.75, 0.25])[0]
            #print(img)
            #if not is_syn:
            small_img = Image.open(img)
            #small_img = image_transform_rot_flip(small_img, rot_type, flip_type)
            ##if color_shuffle:
            #    rgb = [0,1,2]
            #    np.random.shuffle(rgb)
            #add_blur = np.random.choice(2,1,p=[0.75, 0.25])[0]
            #if add_blur and 0:
            #    blur_radius = np.random.choice(4,1,p=[0.6, 0.25, 0.1, 0.05])[0] / 2 + 0.5
            #    small_img = small_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            large_image = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if not scale == 8:
                if 'P' in large_image.mode:  # check if image is a palette type
                    large_image = large_image.convert("RGB")  # convert it to RGB
                    large_image = large_image.resize((config.input_width*scale, config.input_height*scale), Image.BILINEAR)  # resize it
                    large_image = large_image.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = large_image.resize((config.input_width*scale, config.input_height*scale), Image.BILINEAR)  # regular resize
            #large_image = image_transform_rot_flip(large_image, rot_type, flip_type)
            #if is_syn:
            #    blur_radius = random.randint(0, 9) / 10
            #    small_img = large_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            #    small_img = small_img.resize((config.input_width,config.input_height),Image.NEAREST)

            #if color_shuffle:
            #    rgb = [0,1,2]
            #    np.random.shuffle(rgb)
            #    small_img = np.array(small_img)[:,:,rgb]
            #    large_image = np.array(large_image)[:,:,rgb]

            small_images[i] = np.array(small_img) / 255.0

            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images)
        counter += batch_size


def train_image_generator_x2(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        x2_images = np.zeros(
            (batch_size, config.input_width*2, config.input_height*2, 3))
        large_images = np.zeros(
            (batch_size, config.input_width * scale, config.input_height * scale, 3))
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
            #rot_type = random.randint(0, 3) #augment option
            #flip_type = random.randint(0, 2) #augment option
            #flip_type = random.randint(0, 2) #augment option
            img = input_filenames[counter + i]
            #color_shuffle = random.randint(0, 1) #augment option
            #is_syn = np.random.choice(2,1,p=[0.75, 0.25])[0]
            #print(img)
            #if not is_syn:
            small_img = Image.open(img)
            #small_img = image_transform_rot_flip(small_img, rot_type, flip_type)
            ##if color_shuffle:
            #    rgb = [0,1,2]
            #    np.random.shuffle(rgb)
            #add_blur = np.random.choice(2,1,p=[0.75, 0.25])[0]
            #if add_blur and 0:
            #    blur_radius = np.random.choice(4,1,p=[0.6, 0.25, 0.1, 0.05])[0] / 2 + 0.5
            #    small_img = small_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            large_image = Image.open(img.replace("-in.jpg", "-out.jpg"))
            x2_image = large_image.resize((config.input_width * 2, config.input_height * 2),
                                          Image.BILINEAR)  # regular resize

            if not scale == 8:
                if 'P' in large_image.mode:  # check if image is a palette type
                    large_image = large_image.convert("RGB")  # convert it to RGB
                    large_image = large_image.resize((config.input_width*scale, config.input_height*scale), Image.BILINEAR)  # resize it
                    large_image = large_image.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = large_image.resize((config.input_width*scale, config.input_height*scale), Image.BILINEAR)  # regular resize
            #large_image = image_transform_rot_flip(large_image, rot_type, flip_type)
            #if is_syn:
            #    blur_radius = random.randint(0, 9) / 10
            #    small_img = large_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            #    small_img = small_img.resize((config.input_width,config.input_height),Image.NEAREST)

            #if color_shuffle:
            #    rgb = [0,1,2]
            #    np.random.shuffle(rgb)
            #    small_img = np.array(small_img)[:,:,rgb]
            #    large_image = np.array(large_image)[:,:,rgb]

            small_images[i] = np.array(small_img) / 255.0

            large_images[i] = np.array(large_image) / 255.0
            x2_images[i] = np.array(x2_image) / 255.0
        yield (small_images, large_images,x2_images)
        counter += batch_size



def train_image_generator1(batch_size, files):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = files
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
            add_blur = np.random.choice(2,1,p=[0.75, 0.25])[0]
            if add_blur and 0 :
                blur_radius = np.random.choice(4,1,p=[0.6, 0.25, 0.1, 0.05])[0] / 2 + 0.5
                small_img = small_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            small_images[i] = np.array(small_img) / 255.0
            img = Image.open(img.replace("-in.jpg", "-out.jpg"))
            if 1:
                if 'P' in img.mode:  # check if image is a palette type
                    img = img.convert("RGB")  # convert it to RGB
                    img = img.resize((config.input_width*scale, config.input_height*scale), Image.BICUBIC)  # resize it
                    large_image = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                    # convert back to palette
                else:
                    large_image = img.resize((config.input_width*scale, config.input_height*scale), Image.BICUBIC)  # regular resize
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
            if not scale == 8:
                large_image = large_image.resize((config.input_height*scale , config.input_height * scale),Image.BICUBIC)
            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images)
        counter += batch_size



def image_generator_x2(batch_size, img_dir):
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
        x2_images = np.zeros(
            (batch_size, config.input_width * scale, config.input_height * scale, 3))
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
            x2_image = large_image.resize((config.input_height * 2, config.input_height * 2), Image.BICUBIC)
            x2_images[i] = np.array(x2_image) / 255.0
            if not scale == 8:
                large_image = large_image.resize((config.input_height*scale , config.input_height * scale),Image.BICUBIC)
            large_images[i] = np.array(large_image) / 255.0
        yield (small_images, large_images, x2_images)
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

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


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
    #y_true *= 255
    #y_pred *= 255
    rmean = (y_true[ :, :, 0] + y_pred[ :, :, 0]) / 2
    r = y_true[ :, :, 0] - y_pred[ :, :, 0]
    g = y_true[ :, :, 1] - y_pred[ :, :, 1]
    b = y_true[ :, :, 2] - y_pred[ :, :, 2]

    return np.mean(np.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

#for l in model.layers:
    #print(type(l))
val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)
if 0:
    model = sr_resnet84(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)


    checkpoint = ModelCheckpoint('best_resnet84_x8_no_aug.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-7)
    #model.fit(X_train, Y_train, callbacks=[reduce_lr])
    model.fit_generator(train_image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        #steps_per_epoch=1,
                        epochs=config.num_epochs, callbacks=[
                        #epochs = config.num_epochs, callbacks = [
                        checkpoint,reduce_lr],
                        #ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)
elif 0:
    #img_augmentation2()
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=2)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])
    print(model.summary())
    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)

    checkpoint = ModelCheckpoint('best_resnet_x2_aug_new.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-7)
    # model.fit(X_train, Y_train, callbacks=[reduce_lr])
    model.fit_generator(train_image_generator(config.batch_size, train_test_dir),
                        steps_per_epoch=(len(glob.glob(train_test_dir + "/*-in.jpg") )// config.batch_size),
                        #steps_per_epoch=1,
                        epochs=config.num_epochs, callbacks=[
            # epochs = config.num_epochs, callbacks = [
                        checkpoint, reduce_lr],
                        # ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)

elif 1:
    #img_augmentation2()
    model_x2 = sr_resnet_bilin(input_shape=(config.input_width, config.input_height, 3), scale_ratio=2)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
    model_x2.load_weights('best_resnet_x2_aug_new.h5')
    model_x2.trainable = False
    # DONT ALTER metrics=[perceptual_distance]
    model_x2.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])
    model_x4 = sr_resnet_bilin(input_shape=(config.input_width*2, config.input_height*2, 3), scale_ratio=2)
    model_x4.load_weights('best_resnet_x2_aug_new.h5')
    model_x4.compile(optimizer='adam', loss=custom_loss(),
                     metrics=[perceptual_distance, psnr, psnr_v2])
    #model_x8 = sr_resnet_bilin(input_shape=(config.input_width*2, config.input_height*2, 3), scale_ratio=2)
    #model_x8.load_weights('best_resnet_x2_aug_new.h5')
    #model_x8.compile(optimizer='adam', loss=custom_loss(),
    #                 metrics=[perceptual_distance, psnr, psnr_v2])
    #tf.keras.backend.clear_session()

    model = sr_x2_concat((config.input_width, config.input_height, 3),
                         model_x2,model_x4)
    model.compile(optimizer='adam',
                  loss=[custom_loss(),custom_loss()],
                  loss_weights=[1,1e-8],
                  metrics=[perceptual_distance, psnr, psnr_v2])
    print(model.summary())
    val_generator = image_generator_x2(config.batch_size, val_dir)
    in_sample_images, out_sample_images, out_sample_imagesx2 = next(val_generator)

    checkpoint = ModelCheckpoint('best_resnet_combined_aug_new.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-7)
    # model.fit(X_train, Y_train, callbacks=[reduce_lr])
    model.fit_generator(train_image_generator_x2(config.batch_size, train_test_dir),
                        #steps_per_epoch=(len(glob.glob(train_test_dir + "/*-in.jpg") )// config.batch_size),
                        steps_per_epoch=1,
                        epochs=config.num_epochs,
                        callbacks=[],
            # epochs = config.num_epochs, callbacks = [
                        #checkpoint, reduce_lr],
                        # ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)

elif 1:
    #img_augmentation2()
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=2)

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])
    print(model.summary())
    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)

    checkpoint = ModelCheckpoint('best_resnet_x2_aug_new.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=1e-7)
    # model.fit(X_train, Y_train, callbacks=[reduce_lr])
    model.fit_generator(train_image_generator(config.batch_size, train_test_dir),
                        steps_per_epoch=(len(glob.glob(train_test_dir + "/*-in.jpg") )// config.batch_size),
                        #steps_per_epoch=1,
                        epochs=config.num_epochs, callbacks=[
            # epochs = config.num_epochs, callbacks = [
                        checkpoint, reduce_lr],
                        # ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)
elif 1:
    model = dbpn(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    opt = tf.keras.optimizers.Adam(lr=0.0001,decay=0.9)

    print(model.summary())

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss='mae',
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)


    checkpoint = ModelCheckpoint('best_dbpn2_x2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    model.fit_generator(train_image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        #steps_per_epoch=1,
                        epochs=config.num_epochs, callbacks=[
                        #epochs = config.num_epochs, callbacks = [
                        checkpoint],
                        #ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)
elif 0:
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)
    input_filenames = glob.glob("./data/train" + "/*-in.jpg")

    dist_arr = []
    for f in input_filenames:
        lr_img = np.array(Image.open(f).resize((256, 256), Image.ANTIALIAS), dtype='float32') / 255.0
        hr_img = np.array(Image.open(f.replace('-in.jpg', '-out.jpg')), dtype='float32') / 255.0
        dist_arr.append(perceptual_distance_np(lr_img, hr_img))

    g80 = [input_filenames[i] for i, dist in enumerate(dist_arr) if dist > 80]

    model.fit_generator(train_image_generator1(config.batch_size, g80),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=10, callbacks=[
                        #epochs = config.num_epochs, callbacks = [
                        ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)
elif 0:
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)

    # DONT ALTER metrics=[perceptual_distance]
    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    in_sample_images, out_sample_images = next(val_generator)



    model.fit_generator(train_image_generator(config.batch_size, train_dir),
                        steps_per_epoch=config.steps_per_epoch,
                        epochs=1, callbacks=[
                            ImageLogger(), WandbCallback()],
                        validation_steps=config.val_steps_per_epoch,
                        validation_data=val_generator)

    res = resnet_model(input_shape=(config.output_width, config.output_height, 3))
    res.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                , loss='mse'
                )



elif 0:
    res = resnet_model(input_shape=(config.output_width, config.output_height, 3))
    res.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                , loss='mse'
                )
    model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)
    #model = sr_resnet(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)

    model.compile(optimizer='adam', loss=custom_loss(),
                  metrics=[perceptual_distance, psnr, psnr_v2])

    opt = tf.keras.optimizers.Adam(lr=0.001,decay=0.9)
    combine = sr_combine((config.input_width, config.input_height, 3),model, res)
    # DONT ALTER metrics=[perceptual_distance]
    combine.compile(optimizer='adam', loss=['mae', 'mse'], loss_weights=[0.94, 0.06]
                  ,metrics=[perceptual_distance, psnr, psnr_v2])
    #model.compile(optimizer='adam', loss=custom_loss(),
    #              metrics=[perceptual_distance, psnr, psnr_v2])

    val_generator = image_generator(config.batch_size, val_dir)
    train_generator = train_image_generator(config.batch_size, train_dir)
    #in_sample_images, out_sample_images = next(val_generator)
    all_val_input_imgs, all_val_output_imgs = get_all_imgs(val_dir)
    all_content_loss = []
    all_pixel_l1_loss = []
    #print(model.summary())

    for itr in range(2000000):
        input_imgs, output_imgs = next(train_generator)
        real_feat = res.predict(output_imgs)
        #gen_loss = model.train_on_batch(input_imgs, output_imgs  )
        gen_loss = combine.train_on_batch(input_imgs, [output_imgs, real_feat]  )
        all_pixel_l1_loss.append(gen_loss)
        #all_content_loss.append(gen_loss[1])
        if (itr + 1) % 128 == 0:
            # print("fake_img_loss: ", fake_img_loss)
            print(itr, gen_loss)

            #print(itr, np.mean(np.array(all_pixel_l1_loss)), np.mean(np.array(all_content_loss)))
            #print(itr, np.mean(np.array(all_pixel_l1_loss)))
            all_pixel_l1_loss = []
            #all_content_loss = []


        if (itr + 1) % 512 == 0 and 1:
            # results = generator.evaluate(input_imgs, output_imgs, config.batch_size)

            # print("train performance", generator.evaluate(all_train_input_imgs, all_train_output_imgs, config.batch_size))
            #for _ in range(20):
                #val_input_imgs, val_output_imgs = next(val_generator)
                #val_real_feat = res.predict(preprocess_resnet(val_output_imgs))

                #model.evaluate(val_input_imgs, [val_output_imgs, val_real_feat], config.batch_size)
            model.evaluate(all_val_input_imgs, all_val_output_imgs, config.batch_size)
            # print("val performance", results)
            # LogImage(generator, in_sample_images, out_sample_images)
            # wandb.log(results)


elif 0:
    #all_train_input_imgs, all_train_output_imgs = get_all_imgs(train_dir)
    all_val_input_imgs, all_val_output_imgs = get_all_imgs(val_dir)
    #print(len(all_val_input_imgs))
    gan_lr = 1e-4
    disc_lr = 1e-3
    if 1:
        feat_matching = resnet_model(input_shape=(config.output_width, config.output_height, 3))
        feat_matching.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                              , loss='mse'
                    )
    else:
        feat_matching = vgg19_model(input_shape=(config.output_width, config.output_height, 3))
        feat_matching.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                    , loss='mse'
                    )
    discriminator = sr_discriminator(input_shape=(config.output_width, config.output_height, 3))
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=disc_lr, decay=0.9)
                          , loss='binary_crossentropy'
                          )
    generator = sr_prosr_rcan(input_shape=(config.input_width, config.input_height, 3), scale_ratio=scale)
    #print(generator.summary())
    generator.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])



    gan = sr_gan_test((config.input_width, config.input_height, 3), generator, discriminator,feat_matching)
    #print(gan.summary())
    gan.compile(
        loss=['binary_crossentropy', 'mse', 'mae'],
        loss_weights=[5e-4, 1, 1e-2],
        optimizer=tf.keras.optimizers.Adam(lr=gan_lr,beta_1=0.9,beta_2=0.999)
    )
    ## train generator for a couple of epochs
    if 0:
        generator.fit_generator(train_image_generator(config.batch_size, train_dir),
                                steps_per_epoch=config.steps_per_epoch,
                                epochs=2000, callbacks=[
                ImageLogger(), WandbCallback()],
                                validation_steps=config.val_steps_per_epoch,
                                validation_data=val_generator)
        generator.save_weights("trained_generator_{}X_epoch{}.h5".format(scale, 0))
    generator.load_weights("gen_mode_prosr.h5")

    dist_arr = []
    mae_arr = []
    input_filenames = glob.glob(train_dir + "/*-in.jpg")
    for f in input_filenames:
        fname = f
        lr = np.array(Image.open(f), dtype='float32')/255.0
        hr = np.array(Image.open(f.replace('-in.jpg', '-out.jpg')), dtype='float32')/255.0

        result = generator.evaluate(np.expand_dims(lr,axis=0),np.expand_dims(hr,axis=0))
        dist_arr.append(result[1])
        mae_arr.append(result[0])
        #print()
    min_dist = np.min(np.array(dis_arr))
    max_dist = np.max(np.array(dis_arr))
    print("min, max dist: ", min_dist, max_dist)
    grt_60 = 0
    for i, f in input_filenames:

        print(i, dist_arr[i])
        print(i, mae_arr[i])
        if dist_arr[i] > 60:
            grt_60 +=1
    print("greater than 60: ",grt_60)
    #print(gan.summary())
    #print(discriminator.summary())
    disc_weights_list = discriminator.trainable_weights
    gen_weights_list = generator.trainable_weights
    gan_weights_list = gan.trainable_weights

    train_generator = train_image_generator(config.batch_size, train_dir)
    val_generator = image_generator(config.batch_size, val_dir)
    all_dis_loss = []
    all_real_loss = []
    all_fake_loss = []
    all_gen_loss = []
    all_gen_mae_loss = []
    all_gen_feat_loss = []
    all_gen_dis_loss = []
    itr_for_disc = 1
    for itr in range(config.num_epochs * config.steps_per_epoch):
        if (itr + 1) % config.steps_per_epoch == 0:
            K.set_value(gan.optimizer.lr, lr_scheduler(gan_lr, (itr+1), decay_freq=50e3))
            K.set_value(discriminator.optimizer.lr, lr_scheduler(disc_lr, (itr+1) // config.steps_per_epoch))

        input_imgs, output_imgs = next(train_generator)
        gen_imgs = generator.predict(input_imgs)
        real_img_loss = discriminator.train_on_batch(output_imgs, np.ones(config.batch_size))
        fake_img_loss = discriminator.train_on_batch(gen_imgs, np.ones(config.batch_size))
        dis_loss = 0.5 * np.add(real_img_loss,fake_img_loss)
        #print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss, dis_loss)

        #discriminator.trainable = False
        if itr > itr_for_disc or 1:
            if itr == itr_for_disc+1 and 0:
                discriminator.save_weights("trained_discriminator_{}X_epoch{}.h5".format(scale, 0))
                print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss, dis_loss)

            #input_imgs, output_imgs = next(train_generator.batch_gen(config.batch_size))

            real_feat = feat_matching.predict(preprocess_resnet(output_imgs))
            results = generator.evaluate(input_imgs, output_imgs)
            gen_loss = gan.train_on_batch(input_imgs,[np.ones(config.batch_size), real_feat, output_imgs])
            print("batch1: ", gen_loss)
            results = generator.evaluate(input_imgs, output_imgs)
            gen_loss = gan.train_on_batch(input_imgs, [np.ones(config.batch_size), real_feat, output_imgs])
            print("batch2: ", gen_loss)
            results = generator.evaluate(input_imgs, output_imgs)
            gen_loss = gan.train_on_batch(input_imgs, [np.ones(config.batch_size), real_feat, output_imgs])
            print("batch3: ", gen_loss)
            results = generator.evaluate(input_imgs, output_imgs)
            gen_loss = gan.train_on_batch(input_imgs, [np.ones(config.batch_size), real_feat, output_imgs])
            print("batch4: ", gen_loss)
            results = generator.evaluate(input_imgs, output_imgs)

            #real_img_loss = discriminator.train_on_batch(output_imgs, np.ones(config.batch_size) * 0.9)
            gen_imgs = generator.predict(input_imgs)
            #fake_img_loss = discriminator.evaluate(gen_imgs, np.ones(config.batch_size)*0.9)

            #print("gan loss: ", gen_loss)
            all_dis_loss.append(dis_loss)
            all_real_loss.append(real_img_loss)
            all_fake_loss.append(fake_img_loss)
            all_gen_loss.append(gen_loss[0])
            all_gen_mae_loss.append(gen_loss[3])
            all_gen_feat_loss.append(gen_loss[2])
            all_gen_dis_loss.append(gen_loss[1])

            if (itr+1) % 128 == 0:
                #print("fake_img_loss: ", fake_img_loss)

                print(itr, np.mean(np.array(all_dis_loss)), np.mean(np.array(all_gen_loss)), np.mean(np.array(all_gen_mae_loss)), np.mean(np.array(all_gen_feat_loss)), np.mean(np.array(all_gen_dis_loss)))


            if (itr+1) % 512 == 0:
                #results = generator.evaluate(input_imgs, output_imgs, config.batch_size)
                test = np.array(get_weight_grad(generator, input_imgs, output_imgs))
                print(itr, np.mean(test[1]), np.absolute(np.mean(test[-1])))
                #print("train performance", generator.evaluate(all_train_input_imgs, all_train_output_imgs, config.batch_size))
                #results = generator.evaluate(all_val_input_imgs, all_val_output_imgs, config.batch_size)
                results = generator.evaluate(all_val_input_imgs, all_val_output_imgs)
                print("val performance", results)
                LogImage(generator, in_sample_images, out_sample_images)
                wandb.log({"real img loss": np.mean(np.array(all_real_loss)),
                           "fake img loss": np.mean(np.array(all_fake_loss)),
                           "first layer gradients": np.absolute(np.mean(test[0])),
                           "last layer gradients": np.absolute(np.mean(test[-1])),
                           "pixel l1 loss": np.mean(np.array(all_gen_mae_loss)),
                           "content l2 loss": np.mean(np.array(all_gen_feat_loss)),
                           "discriminator crossentropy loss": np.mean(np.array(all_gen_dis_loss)),
                           "val perceptual dist": results[1],

                           })
                all_dis_loss = []
                all_gen_loss = []
                all_gen_mae_loss = []
                all_gen_feat_loss = []
                all_gen_dis_loss = []
                all_real_loss = []
                all_fake_loss = []
                #wandb.log(results)
                if (itr + 1) % 2000 == 0:
                    """Save the generator and discriminator networks"""
                    generator.save_weights("trained_new_generator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))
                    discriminator.save_weights("trained_new_discriminator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))

else:
    #all_train_input_imgs, all_train_output_imgs = get_all_imgs(train_dir)
    all_val_input_imgs, all_val_output_imgs = get_all_imgs(val_dir)
    print(len(all_val_input_imgs))
    gan_lr = 1e-3
    disc_lr = 1e-3
    if 0:
        if 1:
            feat_matching = resnet_model(input_shape=(config.output_width, config.output_height, 3))
            feat_matching.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                                  , loss='mse'
                        )
        else:
            feat_matching = vgg19_model(input_shape=(config.output_width, config.output_height, 3))
            feat_matching.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001, decay=0.9)
                        , loss='mse'
                        )
        discriminator = sr_discriminator(input_shape=(config.output_width, config.output_height, 3))
        discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=disc_lr, decay=0.9)
                              , loss='binary_crossentropy'
                              )
    generatorx1 = sr_prosr_rcan_upsample(input_shape=(config.input_width, config.input_height, 3), scale_ratio=2)
    generatorx1.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])
    print(generatorx1.summary())
    generatorx2 = sr_prosr_rcan_test(input_shape=(config.input_width, config.input_height, 3), scale_ratio=2)
    #print(generatorx2.summary())
    generatorx2.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])
    generatorx4 = sr_prosr_rcan_test(input_shape=(config.input_width*2, config.input_height*2, 3), scale_ratio=2)
    #print(generatorx4.summary())
    generatorx4.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])
    generatorx8 = sr_prosr_rcan_test(input_shape=(config.input_width*4, config.input_height*4, 3), scale_ratio=2)
    #print(generatorx8.summary())
    generatorx8.compile(optimizer='adam', loss='mae', metrics=[perceptual_distance, psnr, psnr_v2])

    test = sr_x2_check((config.input_width, config.input_height, 3), generatorx1, generatorx2,generatorx4,generatorx8)
    #test2 = sr_x2_check2((config.input_width, config.input_height, 3), generatorx2,generatorx4,generatorx8)
    #print(test.summary())
    test.compile(
        loss=['mae', 'mae', 'mae', 'mae'],
        loss_weights=[1, 1, 1, 1],
        metrics=[perceptual_distance],
        optimizer=tf.keras.optimizers.Adam(lr=gan_lr,beta_1=0.9,beta_2=0.999)
    )

    #test2.compile(
    #    loss='mae',
    #    metrics=[perceptual_distance],
    #    optimizer=tf.keras.optimizers.Adam(lr=gan_lr, beta_1=0.9, beta_2=0.999)
    #)
    #print(test2.summary())

    ## train generator for a couple of epochs
    if 0:
        generator.fit_generator(train_image_generator(config.batch_size, train_dir),
                                steps_per_epoch=config.steps_per_epoch,
                                epochs=2000, callbacks=[
                ImageLogger(), WandbCallback()],
                                validation_steps=config.val_steps_per_epoch,
                                validation_data=val_generator)
        generator.save_weights("trained_generator_{}X_epoch{}.h5".format(scale, 0))
    #generator.load_weights("gen_mode_prosr.h5")

        #print(gan.summary())
        #print(discriminator.summary())
        disc_weights_list = discriminator.trainable_weights
        gen_weights_list = generator.trainable_weights
        gan_weights_list = gan.trainable_weights

    train_generator = train_image_generator_test(config.batch_size, train_dir)
    val_generator = image_generator(config.batch_size, val_dir)
    all_x1_dist = []
    all_x2_dist = []
    all_x4_dist = []
    all_x8_dist = []
    all_gen_loss = []
    all_x8_loss = []
    all_x4_loss = []
    all_x2_loss = []
    all_x1_loss = []
    itr_for_disc = 1
    for itr in range(config.num_epochs * config.steps_per_epoch):
        if (itr + 1) % config.steps_per_epoch == 0:
            K.set_value(test.optimizer.lr, lr_scheduler(gan_lr, (itr+1), decay_freq=50e3))
            #K.set_value(discriminator.optimizer.lr, lr_scheduler(disc_lr, (itr+1) // config.steps_per_epoch))

        input_imgs, output_imgs, gtx1_imgs, gtx2_imgs, gtx4_imgs = next(train_generator)
        #gen_imgs = generator.predict(input_imgs)
        #real_img_loss = discriminator.train_on_batch(output_imgs, np.ones(config.batch_size)*0.9)
        #fake_img_loss = discriminator.train_on_batch(gen_imgs, np.ones(config.batch_size)*0.1)
        #dis_loss = 0.5 * np.add(real_img_loss,fake_img_loss)
        #print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss, dis_loss)

        #discriminator.trainable = False
        #if itr == itr_for_disc+1 and 0:
        #    discriminator.save_weights("trained_discriminator_{}X_epoch{}.h5".format(scale, 0))
        #    print("real_img_loss: ", real_img_loss, "; fake_img_loss: ", fake_img_loss, dis_loss)

        #input_imgs, output_imgs = next(train_generator.batch_gen(config.batch_size))

        #real_feat = feat_matching.predict(preprocess_resnet(output_imgs))
        gen_loss = test.train_on_batch(input_imgs,[output_imgs, gtx1_imgs, gtx2_imgs, gtx4_imgs])
        #print(output_imgs.shape)
        #results = test2.evaluate(input_imgs, output_imgs)

        #real_img_loss = discriminator.train_on_batch(output_imgs, np.ones(config.batch_size) * 0.9)
        #gen_imgs = generator.predict(input_imgs)
        #fake_img_loss = discriminator.evaluate(gen_imgs, np.ones(config.batch_size)*0.9)

        #print("gan loss: ", gen_loss)
        #all_dis_loss.append(dis_loss)
        #all_real_loss.append(real_img_loss)
        #all_fake_loss.append(fake_img_loss)
        print(gen_loss)
        all_gen_loss.append(gen_loss[0])
        all_x8_loss.append(gen_loss[1])
        all_x4_loss.append(gen_loss[4])
        all_x2_loss.append(gen_loss[3])
        all_x1_loss.append(gen_loss[2])
        all_x1_dist.append(gen_loss[6])
        all_x8_dist.append(gen_loss[5])
        all_x4_dist.append(gen_loss[8])
        all_x2_dist.append(gen_loss[7])
        if (itr+1) % 128 == 0:
            #print("fake_img_loss: ", fake_img_loss)

            print(itr, np.mean(np.array(all_gen_loss)), np.mean(np.array(all_x8_loss)), np.mean(np.array(all_x4_loss)), np.mean(np.array(all_x2_loss)),  np.mean(np.array(all_x1_loss)),
                  np.mean(np.array(all_x8_dist)), np.mean(np.array(all_x4_dist)), np.mean(np.array(all_x2_dist)), np.mean(np.array(all_x1_dist)))


        if (itr+1) % 512 == 0:
            #results = generator.evaluate(input_imgs, output_imgs, config.batch_size)
            #print("train performance", generator.evaluate(all_train_input_imgs, all_train_output_imgs, config.batch_size))
            #results = generator.evaluate(all_val_input_imgs, all_val_output_imgs, config.batch_size)
            #results = test2.evaluate(all_val_input_imgs, all_val_output_imgs)
            #print("val performance", results)
            #LogImage(test2, in_sample_images, out_sample_images)
            wandb.log({"x1 loss": np.mean(np.array(all_x1_loss)),
                       "x2 loss": np.mean(np.array(all_x2_loss)),
                       "x4 loss": np.mean(np.array(all_x4_loss)),
                       "x8 loss": np.mean(np.array(all_x8_loss)),
                       "x1 dist": np.mean(np.array(all_x1_dist)),
                       "x2 dist": np.mean(np.array(all_x2_dist)),
                       "x4 dist": np.mean(np.array(all_x4_dist)),
                       "x8 dist": np.mean(np.array(all_x8_dist)),
                       #"val perceptual dist": results[1],

                       })
            all_x1_dist = []
            all_x2_dist = []
            all_x4_dist = []
            all_x8_dist = []
            all_gen_loss = []
            all_x8_loss = []
            all_x4_loss = []
            all_x2_loss = []
            all_x1_loss = []
            #wandb.log(results)
            if (itr + 1) % 2000 == 0:
                """Save the generator and discriminator networks"""
                pass
                generator.save_weights("trained_new_generator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))
                discriminator.save_weights("trained_new_discriminator_{}X_epoch{}.h5".format(scale, itr // config.steps_per_epoch))


