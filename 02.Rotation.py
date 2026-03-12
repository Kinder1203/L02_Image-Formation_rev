import cv2
import numpy as np

# 1. 이미지 로드
img = cv2.imread("images/rose.png")
if img is None:
    raise FileNotFoundError("images/rose.png 파일을 찾을 수 없습니다.")

h, w = img.shape[:2]
center = (w // 2, h // 2)

# 2. 회전 및 스케일 변환 행렬 생성 (30도 회전, 0.8배 축소)
# getRotationMatrix2D는 2x3 Affine 행렬을 반환함
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 3. 평행 이동 결합 (Translation)
# 반환된 2x3 행렬의 3번째 열([0, 2]와 [1, 2])이 x, y 이동을 담당함
M[0, 2] += 80  # x축으로 +80px 이동
M[1, 2] -= 40  # y축으로 -40px 이동

# 4. Affine Transform 적용
# 빈 공간은 검은색(기본값)으로 채워짐
transformed_img = cv2.warpAffine(img, M, (w, h))

# 5. 결과 저장 및 출력
cv2.imwrite("outputs/transformed_rose.png", transformed_img)
