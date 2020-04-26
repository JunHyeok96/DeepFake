import os
import cv2
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_WIDTH = 64
IMG_HEIGHT = 64

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img

def data_load():
    list_ds_src_img = tf.data.Dataset.list_files("./dataset/아이유/img/*", shuffle=False)
    list_ds_src_land= tf.data.Dataset.list_files("./dataset/아이유/land/*", shuffle=False)

    list_ds_dst_img = tf.data.Dataset.list_files("./dataset/이준혁/img/*", shuffle=False)
    list_ds_dst_land= tf.data.Dataset.list_files("./dataset/이준혁/land/*", shuffle=False)
    # Set `num_parallel__calls` so multiple images are loaded/processed in parallel.
    src_ds = list_ds_src_img.map(process_path, num_parallel_calls=AUTOTUNE)
    src_land_ds = list_ds_src_land.map(process_path, num_parallel_calls=AUTOTUNE)

    dst_ds = list_ds_dst_img.map(process_path, num_parallel_calls=AUTOTUNE)
    dst_land_ds = list_ds_dst_land.map(process_path, num_parallel_calls=AUTOTUNE)


    src_dataset = tf.data.Dataset.zip((src_ds, src_land_ds))
    dst_dataset = tf.data.Dataset.zip((dst_ds, dst_land_ds))

    return src_dataset, dst_dataset

