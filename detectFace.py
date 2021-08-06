import numpy as np
import cv2 as cv
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

# 이목구비 지정

RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))
LEFT_CHEEK = [4, 3, 2, 1, 31, 50, 49, 48]
RIGHT_CHEEK = [12, 13, 14, 15, 35, 52, 53, 54]

img = cv.imread('face.jpg')
imgOriginal = img.copy()
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask = np.zeros_like(img)
faces = detector(imgGray)

for face in faces:  # 얼굴 인식
    x1, y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    imgOriginal = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    landmarks = predictor(imgGray, face)

    myPoints = []
    for n in range(68):  # 얼굴에서 이목구비 인식
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x, y])  # 68개 포인트 좌표값 저장

# 좌표값 저장
with open("pixel.txt", "w") as file:
    vstr = ''
    sep = ' '

    for a in myPoints:
        for b in a:
            vstr = vstr + str(b) + sep
        vstr = vstr.rstrip(sep) # 마지막 추가되는 sep 삭제
        vstr = vstr + '\n'

    file.writelines(vstr)

print(myPoints)