import cv2 
import dlib
import numpy as np

# 카메라에 접근하기 위해 VideoCapture 객체를 생성
def swapRGB2BGR(img):
    r, g, b = cv2.split(img)
    rgb = cv2.merge([b,g,r])
    return rgb


cap = cv2.VideoCapture("lee_video.mp4")
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()

# 얼굴 인식용 클래스 생성 (기본 제공되는 얼굴 인식 모델 사용)
detector = dlib.get_frontal_face_detector()
# 인식된 얼굴에서 랜드마크 찾기위한 클래스 생성 
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
    
    print(crop_face.shape)
    for x_value, y_value in zip(x,y):
        cv2.circle(land_mark, (x_value, y_value), 2, (0, 0, 255), -1)
        
    land_mark = land_mark[np.min(y):np.max(y),np.min(x): np.max(x),  :]
    crop_face = cv2.resize(crop_face, (128,128))
    land_mark = cv2.resize(land_mark, (128,128))
    
    cv2.imwrite("dataset/lee_img/{:04d}.png".format(index),swapRGB2BGR(crop_face))
    cv2.imwrite("dataset/lee_land/{:04d}.png".format(index), land_mark)
    
    cv2.imshow("", swapRGB2BGR(crop_face))
    cv2.imshow("2", land_mark)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
    index+=1
        
cap.release()
cv2.destroyAllWindows()