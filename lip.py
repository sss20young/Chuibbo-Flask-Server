import cv2 as cv
import numpy as np
import dlib

""" 이목구비 좌표 """
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
LEFT_SHADOW = [36, 17, 18, 19, 20, 21, 39, 38, 37]
RIGHT_SHADOW = [45, 26, 25, 24, 23, 22, 42, 43, 44]
UP_MOUTH = [48, 49, 50, 51, 52, 53, 54, 63, 62, 61]
DOWN_MOUTH = [48, 60, 59, 58, 57, 56, 55, 54, 64, 65, 66, 67]
SNOT_HIGHLIGHTER = [39, 27, 42, 35, 33, 31]
LEFT_PUPIL = [37, 38, 40, 41]
RIGHT_PUPIL = [43, 44, 46, 47]
# ALL = [RIGHT_EYEBROW, LEFT_EYEBROW, RIGHT_EYE, LEFT_EYE, NOSE, MOUTH_OUTLINE, MOUTH_INNER, JAWLINE, RIGHT_CHEEK,LEFT_CHEEK]

class makeUp:
    def __init__(self):
        self.img = None
        self.make = None

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("../data/shape_predictor_68_face_landmarks.dat")

    def readImg(self):
        self.img = cv.imread('static/Input.jpg')

    def makeUpFeatures(self, r, g, b, size, index, strong):

        def createBox(img, points, scale=5, masked=False, cropped=True):
            if masked:
                global mask
                mask = cv.fillPoly(self.mask, [points], (255, 255, 255))  # 정확한 영역 추출을 위해 rectangle 말고 fillpoly 사용
                img = cv.bitwise_and(img, mask)
                # cv.imshow("ssg", img)
            if cropped:
                bbox = cv.boundingRect(points)  # x, y, 넓이, 높이 반환
                x, y, w, h = bbox
                imgCrop = img[y:y + h, x:x + w]
                imgCrop = cv.resize(imgCrop, (0, 0), None, scale, scale)
                return imgCrop
            else:
                return mask

        imgOriginal = self.img.copy()
        imgGray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.mask = np.zeros_like(self.img)
        faces = self.detector(imgGray)

        if index == 1:
            indices = [LEFT_SHADOW, RIGHT_SHADOW]
        elif index == 2:
            indices = [UP_MOUTH, DOWN_MOUTH]
        elif index == 3:
            indices = [LEFT_CHEEK, RIGHT_CHEEK]
        elif index == 4:
            indices = [LEFT_PUPIL, RIGHT_PUPIL]

        """ 얼굴 인식 """
        for face in faces:
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            # imgOriginal = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 12)
            landmarks = self.predictor(imgGray, face)

            myPoints = []
            for n in range(68):  # 얼굴에서 이목구비 인식
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                myPoints.append([x, y])  # 68개 포인트 좌표값 저장
                # cv.circle(imgOriginal, (x, y), 3, (50, 50, 255), cv.FILLED)
                # cv.putText(imgOriginal, str(n), (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1)

            for index in indices:
                # 이목구비 영역 지정
                myPoints = np.array(myPoints)  # 넘파이 배열로 변경
                imgFeatures = createBox(self.img, myPoints[index], 3, masked=True, cropped=False)

                # print(np.array(myPoints))
                # print(np.array(imgFeatures))

                # 이목구비 초기화
                imgColorFeatures = np.zeros_like(imgFeatures)

                # 선택 영역 색칠
                imgColorFeatures[:] = b, g, r
                imgColorFeatures = cv.bitwise_and(imgFeatures, imgColorFeatures)
                imgColorFeatures = cv.boxFilter(imgColorFeatures, ddepth=-1, ksize=size)  # 자연스럽게 블러 처리
                """ 입술&눈동자 11,11 / 볼 80,80 / 섀도우 45,45 / 콧대 하이라이터 100,100"""
                # imgOriginalGray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)  # 채널 똑같이 하기 위해 조정
                # imgOriginalGray = cv.cvtColor(imgOriginalGray, cv.COLOR_GRAY2BGR)  # 채널 똑같이 하기 위해 조정
                imgColorFeatures = cv.addWeighted(imgOriginal, 1, imgColorFeatures, strong, 0)  # 원본에 메이크업 추가 0.07

                # cv.imshow('BGR', imgColorFeatures)
                # cv.imshow('ORG', imgOriginal)
                cv.imwrite('static/Output.jpg', imgColorFeatures)
                print(index, "저장...")
                # cv.imshow('Lips', imgFeatures)
                # print(myPoints)

        # cv.waitKey()
        # cv.destroyAllWindows()

if __name__ == '__main__':
    make = makeUp()
    make.readImg()
    make.makeUpFeatures(size=(11,11))