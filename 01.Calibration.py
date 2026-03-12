import cv2
import numpy as np
import glob
import os

# -----------------------------
# 0. 설정 및 변수 초기화
# -----------------------------
# 체크보드 내부 코너 개수 (만약 계속 실패하면 이 숫자를 사진에 맞게 수정해야 합니다!)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

# 경로 설정 (현재 실행 위치 기준)
print("▶ 현재 실행 디렉토리:", os.getcwd())
search_path = "images/calibration_images/left*.jpg"
images = glob.glob(search_path)
print(f"▶ 찾은 이미지 개수: {len(images)}장 (경로: {search_path})\n")

img_size = None
success_count = 0

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"이미지 읽기 에러: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너 검출 성공 여부 출력
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        success_count += 1
        print(f"⭕ 성공: {fname}")
    else:
        print(f"❌ 실패 (코너 못 찾음): {fname}")

print(f"\n▶ 최종적으로 코너 검출에 성공한 유효 이미지: {success_count}장")

# -----------------------------
# 2. 카메라 캘리브레이션 및 검증
# -----------------------------
if success_count == 0:
    print("\n[🚨 치명적 에러 분석 🚨]")
    if len(images) == 0:
        print("원인: 이미지를 한 장도 찾지 못했습니다.")
        print("해결책: 폴더 이름이 오타가 없는지, 바탕화면의 computer_vison/2 폴더 안에 images 폴더가 정확히 있는지 확인하세요.")
    else:
        print("원인: 이미지는 찾았지만 코너 검출에 100% 실패했습니다.")
        print("해결책: left01.jpg 사진을 열어보고 내부 코너(검은색과 흰색이 교차하는 십자가 지점) 개수를 직접 세어보세요.")
        print("가로 교차점 개수와 세로 교차점 개수를 세어서 코드 10번째 줄의 CHECKERBOARD = (가로, 세로) 로 수정해야 합니다.")
else:
    print("\n▶ 캘리브레이션 연산을 시작합니다...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    print("\nCamera Matrix K:")
    print(K)

    print("\nDistortion Coefficients:")
    print(dist)
    
    # 왜곡 보정 테스트 (성공한 첫 번째 이미지 대상)
    test_img = cv2.imread(images[0])
    undistorted_img = cv2.undistort(test_img, K, dist, None, K)
    cv2.imwrite("outputs/undistorted_test.jpg", undistorted_img)
    print("\n▶ 완료: 왜곡 보정 테스트 이미지가 'outputs/undistorted_test.jpg'로 저장되었습니다.")