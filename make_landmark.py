import cv2 
import dlib
import numpy as np

def swapRGB2BGR(img):
    r, g, b = cv2.split(img)
    rgb = cv2.merge([b,g,r])
    return rgb

VIDEO_PATH = "dataset_video/src/video.mp4"
IMG_PATH="dataset/src/img/"
LAND_PATH = "dataset/src/land/"
LAND_CROP_SIZE = 18 #  you can choose 0, 18, 28   This determines how much the landmark image will be cut

cap = cv2.VideoCapture(VIDEO_PATH)
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
        print("not detected")
        continue
    try:   
        for k, d in enumerate(dets):
            
            shape = predictor(img, d)
            x = [shape.part(i).x for i in range(LAND_CROP_SIZE, shape.num_parts)]   
            y = [shape.part(i).y for i in range(LAND_CROP_SIZE, shape.num_parts)] 

        if np.max(y)- np.min(y) <128 or np.max(x) - np.min(x) <128:
            continue
        crop_face = img[np.min(y):np.max(y),np.min(x): np.max(x),  :]
        
        print(crop_face.shape)
        for x_value, y_value in zip(x,y):
            cv2.circle(land_mark, (x_value, y_value), 2, (0, 0, 255), -1)
        land_mark = land_mark[np.min(y):np.max(y),np.min(x): np.max(x),  :]
        crop_face = cv2.resize(crop_face, (128,128))
        land_mark = cv2.resize(land_mark, (128,128))
    except:
        continue
    cv2.imwrite(IMG_PATH+"{:04d}.png".format(index),swapRGB2BGR(crop_face)) # IMG_PATH
    cv2.imwrite(LAND_PATH+"{:04d}.png".format(index), land_mark)            # LAND_PATH
    cv2.imshow("", swapRGB2BGR(crop_face))      
    cv2.imshow("2", land_mark)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    index+=1
        
cap.release()
cv2.destroyAllWindows()