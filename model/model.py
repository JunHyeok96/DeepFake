import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.layers import  Flatten, Conv2D, MaxPooling2D, Dense, Dropout, Softmax, Conv2DTranspose, BatchNormalization
from tensorflow_examples.models.pix2pix import pix2pix


def encoder():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same' , input_shape=[64, 64, 3]))
    model.add(layers.Conv2D(48, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(192, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(200))
    return model

def decoder():
    model = tf.keras.Sequential()
    dropout = 0.4 
    depth = 64 *4
    dim = 8
    model.add(layers.Dense(dim*dim*depth, input_dim=200))
    model.add(layers.BatchNormalization(momentum=0.9)) 
    model.add(layers.Activation('relu'))
    model.add(layers.Reshape((dim, dim, depth))) 
    model.add(layers.Dropout(dropout)) 
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(int(depth/2), 5, padding='same'))
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(int(depth/4), 5, padding='same')) 
    model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation('relu')) 
    model.add(layers.Conv2DTranspose(int(depth/8), 5, padding='same')) 
    model.add(layers.BatchNormalization(momentum=0.9)) 
    model.add(layers.Activation('relu'))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2DTranspose(3, 5, padding='same'))
    model.add(layers.Activation('tanh'))
    return model              

def autoencoder(encoder , decoder):
    model = tf.keras.Sequential()
    model = tf.keras.models.clone_model(encoder)
    model.add(decoder)
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def build_adversarial_model(gen_model, dis_model):
    model = tf.keras.Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model


def vgg16_encoder(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu", input_shape=input_shape))
    model.add(Conv2D(64, (3,3), padding ="same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(128, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2)))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(256, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block3_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block4_pool'))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(Conv2D(512, (3,3), padding = "same", activation = "relu"))
    model.add(MaxPooling2D((2,2), strides=(2, 2), name = 'block5_pool'))
    return model


def fcn_decoder(input_shape, model):
    encoder = model
    layer_names  = ['block3_pool', 'block4_pool', 'block5_pool']
    layers = [encoder.get_layer(name).output for name in layer_names]
    fcn_model = tf.keras.Model(inputs=encoder.input, outputs=layers)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    skip = fcn_model(x)
    vgg_last_layer = skip[-1]
    x = vgg_last_layer
    x =  Conv2D(4096, (7,7), padding = "same", activation = "relu")(x)
    x = Conv2D(3, (1,1), padding = "same", activation = "relu")(x)
    x = Conv2DTranspose(512, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([skip[1], x])
    x = Conv2DTranspose(256, 4, (2,2), padding = "same", activation = "relu")(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Concatenate()([skip[0], x])
    x = BatchNormalization()(x)
    x = Conv2DTranspose(3, 16, (8,8), padding = "same", activation = "relu")(x)
    model =  tf.keras.Model(inputs = inputs , outputs = x)
    
    return model
