import cv2 
import dlib
import numpy as np
import tensorflow as tf
from data_loader import data_load
from IPython.display import clear_output
from model.model import fcn_decoder,vgg16_encoder
import datetime
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
%matplotlib inline

IMG_WIDTH = 128
IMG_HEIGHT = 128

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
       gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)])
    except RuntimeError as e:
        print(e)
        
        
encoder = vgg16_encoder((IMG_HEIGHT, IMG_WIDTH,3))
decoder_src = fcn_decoder((IMG_HEIGHT, IMG_WIDTH,3), encoder)
decoder_dst = fcn_decoder((IMG_HEIGHT, IMG_WIDTH,3), encoder)

decoder_src.load_weights("model_h5/fcn/src2.h5")
decoder_dst.load_weights("model_h5/fcn/dst2.h5")


def swapRGB2BGR(img):
    r, g, b = cv2.split(img)
    rgb = cv2.merge([b,g,r])
    return rgb

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('result_video/lee_result.mp4', fourcc, 30.0, (480,270))
cap = cv2.VideoCapture("lee_video.mp4")
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

index = 0 

while(True):
    ret, img = cap.read()
    
    img = swapRGB2BGR(img)
    land_mark = img.copy()
    land_mark -=land_mark
    dets = detector(img, 0)
    
    if len(dets)<1 :
        print("검출 x")
        continue
        
    for k, d in enumerate(dets):
        # k 얼굴 인덱스
        # d 얼굴 좌표
        
        # 인식된 좌표에서 랜드마크 추출 
        shape = predictor(img, d)
        # num_parts(랜드마크 구조체)를 하나씩 루프를 돌린다.
        x = [shape.part(i).x for i in range(28, shape.num_parts)] 
        y = [shape.part(i).y for i in range(28, shape.num_parts)] 

    crop_face = img[np.min(y):np.max(y),np.min(x): np.max(x),  :]
    
    for x_value, y_value in zip(x,y):
        cv2.circle(land_mark, (x_value, y_value), 2, (0, 0, 255), -1)
        
    land_mark = land_mark[np.min(y):np.max(y),np.min(x): np.max(x),  :]
    land_mark = cv2.resize(land_mark, (128,128))
    land_mark =  swapRGB2BGR(land_mark)
    pred_img =  decoder_src(land_mark[tf.newaxis,...]/255)[0].numpy()
    pred_img = cv2.resize(pred_img, (np.max(x)-np.min(x),np.max(y)-np.min(y)))
    img = img/255
    img[np.min(y):np.max(y),np.min(x): np.max(x),  :] = pred_img
    img = swapRGB2BGR(img)
    img = cv2.resize(img, (480,270))
    cv2.imshow("", img)
    
    video = np.uint8(img*240)
    out.write(video)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    index+=1
        
cap.release()
cv2.destroyAllWindows()
out.release()