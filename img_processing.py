import cv2
import io
import os
import numpy as np
import cv2 as cv
from PIL import Image
from base64 import encodebytes

def detect_faces(img_path):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    image = cv.imread(img_path)
    grayImage = cv.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayImage, 1.03, 5)

    return faces.shape[0]


def preprocessing_crop(img_path):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv.imread(img_path)
    height, width, channels = img.shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 얼굴을 검출할 그레이스케일 이미지를 준비해놓습니다.
    # 이미지에서 얼굴을 검출합니다.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴받습니다.
    for (x,y,w,h) in faces:
        # 가로는 w기준으로, x1 : w : x1 으로 자른다
        # 가로는 h기준으로, y_up : h : y_down 으로 자른다
        x1 = int(0.53 * (w / 2))
        x2 = x + w + x1
        y_up = int(0.88 * (h / 2))
        y_down = int(1.4 * (h / 2))
        # 패딩을 얼마나 줄 것인지를 각각 구한다.
        # 왼쪽
        if x1 - x > 0:
            left = x1 - x + 1
        else:
            left = 0
        # 오른쪽
        if x + w + x1 > width:
            right = x + w + x1 - width + 1
        else:
            right = 0
        # 위
        if y_up - y > 0:
            up = y_up - y + 1
        else:
            up = 0
        # 아래
        if y + h + y_down > height:
            down = y + h + y_down - height + 1
        else:
            down = 0
        px = img[int(width/2), 1].tolist() # img의 (width/2, 1) 의 좌표 색상으로 패딩을 준다.
        constant = cv2.copyMakeBorder(img, up, down, left, right, cv2.BORDER_CONSTANT, value=px) # 위 아래 왼 오 // 이미지에 패딩을 준다.
        gray2 = cv.cvtColor(constant, cv.COLOR_BGR2GRAY)
        faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5) # 패딩을 준 이미지에서 다시 얼굴을 검출합니다.
        # 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴받습니다.
        for (x,y,w,h) in faces2:
            if y > y_up:
                starty = y - y_up
            else:
                starty = y_up - y
            if x > x1:
                startx = x - x1
            else:
                startx = x1 - x
            endy = y + h + y_down
            endx = x + w + x1
            cropped = constant[starty:endy, startx:endx]
            cv2.imwrite(img_path, cropped) # 최종 이미지를 저장할 경로 지정

def transform_encoded_image(result_image_png):
    file = np.fromfile(result_image_png)
    pil_img = Image.open(io.BytesIO(file))
    img_resize = pil_img.resize((int(pil_img.width), int(pil_img.height*9/7))) # 이미지 크기 조절
    img_resize = img_resize.convert("RGB")
    byte_arr = io.BytesIO()
    img_resize.save(byte_arr, format='PNG')
    encoded_img = encodebytes(byte_arr.getvalue())
    return encoded_img

def delete_image(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist")