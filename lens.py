import cv2 as cv
import numpy as np

img = cv.imread('static/Input.jpg')
print(img.shape)
rows, cols = img.shape[:2]

# ---① 설정 값 셋팅
exp = 0.8     # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
scale = 1           # 변환 영역 크기 (0 ~ 1)

# 매핑 배열 생성 ---②
mapy, mapx = np.indices((rows, cols),dtype=np.float32)

# 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경 ---③
mapx = 2*mapx/(cols-1)-1
mapy = 2*mapy/(rows-1)-1

# 직교좌표를 극 좌표로 변환 ---④
r, theta = cv.cartToPolar(mapx, mapy) # 원점으로부터 거리, 사잇각

# 왜곡 영역만 중심확대/축소 지수 적용 ---⑤
r[r< scale] = r[r<scale] **exp

# 극 좌표를 직교좌표로 변환 ---⑥
mapx, mapy = cv.polarToCart(r, theta)

# 중심점 기준에서 좌상단 기준으로 변경 ---⑦
mapx = ((mapx + 1)*cols-1)/2
mapy = ((mapy + 1)*rows-1)/2
# 재매핑 변환
distorted = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)
imgColorFeatures = cv.addWeighted(img, 1, distorted, 0.9, 0)

cv.imshow('imgColorFeatures', imgColorFeatures)
cv.imshow('distorted', distorted)
cv.waitKey()
cv.destroyAllWindows()