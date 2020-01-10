import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image
import os
from sklearn.utils import shuffle
import cv2
import glob
import imageio
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from model import *
from warping import *

input_path="/kaggle/input/fulldata/fulldata/lee_full/"
input_land_path="/kaggle/input/fulldata/fulldata/lee_full_land/"
output_path="/kaggle/input/fulldata/fulldata/iu_full/"
output_land_path="/kaggle/input/fulldata/fulldata/iu_full_land/"



images_input = os.listdir(input_path)
images_input_land = os.listdir(input_land_path)
images_output = os.listdir(output_path)
images_output_land = os.listdir(output_land_path)
images_input.sort()
images_input_land.sort()
images_output.sort()
images_output_land.sort()

train_data_input=[]
for i in range(len(images_input)):
    img = cv2.imread(input_path+images_input[i])
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    train_data_input.append(img)   
        
train_data_input_land=[]
for i in range(len(images_input_land)):
    img = cv2.imread(input_land_path+images_input_land[i])
    img = cv2.resize(img,(64,64))
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    train_data_input_land.append(img)    
    
train_data_output=[]
for i in range(len(images_output)):
    img = cv2.imread(output_path+images_output[i])
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    train_data_output.append(img)   
        
train_data_output_land=[]
for i in range(len(images_output_land)):
    img = cv2.imread(output_land_path+images_output_land[i])
    img = cv2.resize(img,(64,64))
    b, g, r = cv2.split(img)
    img = cv2.merge([r,g,b])
    train_data_output_land.append(img)

    data_input = np.array(train_data_input)
data_input = data_input/255

data_input_land = np.array(train_data_input_land)
data_input_land= data_input_land/255

data_output = np.array(train_data_output)
data_output= data_output/255

data_output_land = np.array(train_data_output_land)
data_output_land= data_output_land/255

data_input = shuffle(data_input, random_state=12)
data_input_land = shuffle(data_input_land, random_state=12)
data_output = shuffle(data_output, random_state=12)
data_output_land = shuffle(data_output_land, random_state=12)



warp_data_input = []
for i in range(len(data_input)):
    warp_data_input.append(random_warp(data_input[i])[0])
warp_data_input = np.array(warp_data_input)

warp_data_output = []
for i in range(len(data_output)):
    warp_data_output.append(random_warp(data_output[i])[0])
warp_data_output = np.array(warp_data_output)

plt.subplot(2,2,1)
plt.imshow(data_input[115])
plt.subplot(2,2,2)
plt.imshow(warp_data_input[115])
plt.subplot(2,2,3)
plt.imshow(data_output[115])
plt.subplot(2,2,4)
plt.imshow(warp_data_output[115])


encode = encoder()
decode_input = decoder()
decode_output = decoder()
generator_input = autoencoder(encode,decode_input)
generator_output = autoencoder(encode,decode_output)
#generator_input.load_weights("/kaggle/input/fulldata/generator_best_input.h5")
generator_output.load_weights("/kaggle/input/fulldata/generator_best_output (16).h5")


generator_optimizer = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.5)
generator_input.compile(loss='mse', optimizer=generator_optimizer)
generator_output.compile(loss='mse', optimizer=generator_optimizer)


generator_output.fit(warp_data_output[:8000], data_output[:8000],
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(warp_data_output[8000:], data_output[ 8000:]),
                verbose=2,
                callbacks=[ModelCheckpoint(filepath='generator_best_output.h5',
                save_weights_only=True, save_best_only=True, verbose=2)])
generator_input.fit(warp_data_input[:3500], data_input[:3500],
                epochs=500,
                batch_size=256,
                shuffle=True,
                validation_data=(warp_data_input[3500:], data_input[3500:]),
                verbose=2,
                callbacks=[ModelCheckpoint(filepath='generator_best_input.h5',
                save_weights_only=True, save_best_only=True, verbose=2)])


generator_input.load_weights("/kaggle/working/generator_best_input.h5")
generator_output.load_weights("/kaggle/working/generator_best_output.h5")
generator_optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
generator_input.compile(loss='mse', optimizer=generator_optimizer)
generator_output.compile(loss='mse', optimizer=generator_optimizer)

warp_pre = generator_output.predict(warp_data_output)
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(4,6,1+(i*6))
    plt.imshow(data_output[i])
    plt.axis("off")
    plt.subplot(4,6,2+(i*6))
    plt.imshow(warp_data_output[i])
    plt.axis("off")
    plt.subplot(4,6,3+(i*6))
    plt.imshow(warp_pre[i])
    plt.axis("off")    
    plt.subplot(4,6,4+(i*6))
    plt.imshow(data_output[i+1])
    plt.axis("off")
    plt.subplot(4,6,5+(i*6))
    plt.imshow(warp_data_output[i+1])
    plt.axis("off")
    plt.subplot(4,6,6+(i*6))
    plt.imshow(warp_pre[i+1])
    plt.axis("off")