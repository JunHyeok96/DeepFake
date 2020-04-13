import os
import cv2
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 128
IMG_HEIGHT = 128

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def data_load():
    list_ds_src_img = tf.data.Dataset.list_files("./dataset/iu_img/*", shuffle=False)
    list_ds_src_land= tf.data.Dataset.list_files("./dataset/iu_land/*", shuffle=False)

    list_ds_dst_img = tf.data.Dataset.list_files("./dataset/lee_img/*", shuffle=False)
    list_ds_dst_land= tf.data.Dataset.list_files("./dataset/lee_land/*", shuffle=False)
    # Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
    src_ds = list_ds_src_img.map(process_path, num_parallel_calls=AUTOTUNE)
    src_land_ds = list_ds_src_land.map(process_path, num_parallel_calls=AUTOTUNE)

    dst_ds = list_ds_dst_img.map(process_path, num_parallel_calls=AUTOTUNE)
    dst_land_ds = list_ds_dst_land.map(process_path, num_parallel_calls=AUTOTUNE)


    src_dataset = tf.data.Dataset.zip((src_ds, src_land_ds))
    dst_dataset = tf.data.Dataset.zip((dst_ds, dst_land_ds))

    return src_dataset, dst_dataset

