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
