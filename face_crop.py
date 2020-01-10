import cv2 

# 카메라에 접근하기 위해 VideoCapture 객체를 생성
cap = cv2.VideoCapture("lee2.mp4")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):

    # 이미지를 캡쳐
    ret, img = cap.read()

    # 캡쳐되지 않은 경우 처리
    if ret == False:
        break;
    try:        
        faces = face_cascade.detectMultiScale(img, 1.3,5)
        for (x,y,w,h) in faces:
            cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
            img= img[y:y+h, x:x+w]     
            img = cv2.resize(img, (64,64))
        #index+=1
        #cv2.imwrite('./data/{:04d}.png'.format(index),img)
        cv2.imshow("VideoFrame", img)

    except:
        continue
        
    # ESC 키누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
# VideoCapture 객체를 메모리 해제하고 모든 윈도우 창을 종료합니다.
cap.release()
cv2.destroyAllWindows()