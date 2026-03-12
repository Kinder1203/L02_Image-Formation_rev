import cv2 
import numpy as np 
import glob 
import os 

# -----------------------------
# 0. 설정 및 변수 초기화
# -----------------------------
# 체크보드의 내부 코너 개수 설정 (가로 교차점 9개, 세로 교차점 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 물리적 크기 설정 (단위: mm)
square_size = 25.0

# 서브픽셀(SubPixel) 정밀도를 높이기 위한 반복 알고리즘 종료 조건 설정
# (알고리즘이 오차 0.001 이하에 도달하거나, 최대 30번 반복하면 정밀 탐색을 종료함)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 실제 세계 좌표계(World coordinates)를 저장할 빈 Numpy 배열 생성
# 평면이므로 Z축은 0으로 가정하며, 크기는 (총 코너 개수 54개, x/y/z 3차원)인 배열 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)

# np.mgrid를 사용하여 (0,0), (1,0) ... 형태의 2D 격자 좌표를 생성한 뒤, 배열의 X, Y 열 위치에 할당
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 생성된 기본 격자 좌표에 실제 체크보드 한 칸의 크기(25.0mm)를 곱하여 실제 물리적 거리 스케일 반영
objp *= square_size

# 모든 이미지에서 성공적으로 찾은 3D 실제 좌표(objp) 배열들을 모아둘 빈 리스트
objpoints = []
# 모든 이미지에서 성공적으로 찾은 2D 이미지 픽셀 좌표들을 모아둘 빈 리스트
imgpoints = []

# 현재 파이썬 스크립트가 실행되고 있는 작업 디렉토리 경로를 터미널에 출력
print("▶ 현재 실행 디렉토리:", os.getcwd())

# 찾고자 하는 캘리브레이션용 이미지 파일들의 경로 및 패턴 지정 (left01.jpg, left02.jpg 등)
search_path = "images/calibration_images/left*.jpg"

# glob을 사용하여 지정된 패턴에 맞는 모든 이미지 파일의 경로를 리스트 형태로 가져옴
images = glob.glob(search_path)

# 검색된 이미지 파일의 총 개수와 해당 경로 패턴을 화면에 출력하여 확인
print(f"▶ 찾은 이미지 개수: {len(images)}장 (경로: {search_path})\n")

# 입력 이미지의 해상도(가로, 세로 픽셀 크기)를 저장할 변수 초기화 (캘리브레이션 함수 호출 시 필요함)
img_size = None
# 코너 검출에 성공한 이미지의 개수를 카운트하기 위한 변수를 0으로 초기화
success_count = 0

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
# 찾은 이미지 파일 경로 리스트를 순회하며 하나씩 처리하는 반복문 시작
for fname in images:
    # OpenCV를 사용하여 파일 경로(fname)로부터 이미지를 읽어와 img 변수에 할당
    img = cv2.imread(fname)

    # 이미지를 정상적으로 읽어오지 못한 경우(파일이 깨졌거나 경로 오류 등) 예외 처리
    if img is None:
        # 에러 메시지 출력
        print(f"이미지 읽기 에러: {fname}")
        # 아래 코드를 무시하고 다음 이미지 파일로 넘어감
        continue

    # 코너 검출 알고리즘은 흑백 이미지에서 수행해야 하므로, 원본 컬러 이미지(BGR)를 흑백(Grayscale)으로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 크기(img_size)가 아직 설정되지 않았다면 (첫 번째 유효 이미지를 처리할 때 1회만 실행됨)
    if img_size is None:
        # 흑백 이미지의 shape(세로, 가로) 차원을 뒤집어서 (가로, 세로) 튜플 형태로 img_size에 저장
        img_size = gray.shape[::-1]

    # 흑백 이미지에서 체크보드의 내부 코너 위치를 찾는 함수 호출
    # (ret: 검출 성공 여부 boolean, corners: 찾은 2D 코너 픽셀 좌표 배열 반환)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 체크보드 코너를 모두 성공적으로 찾은 경우
    if ret == True:
        # 3D 공간 상의 정해진 실제 좌표 배열(objp)을 objpoints 리스트에 추가 (2D 픽셀 좌표와 쌍을 맞추기 위함)
        objpoints.append(objp)

        # 찾은 코너 좌표의 정밀도를 픽셀 이하(SubPixel) 단위로 미세하게 조정하여 캘리브레이션 정확도를 높임
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 정밀화된 2D 이미지 좌표를 imgpoints 리스트에 추가
        imgpoints.append(corners2)

        # 성공적으로 코너를 찾았으므로 성공 카운트를 1 증가
        success_count += 1

        # 어떤 파일이 성공했는지 화면에 출력
        print(f"⭕ 성공: {fname}")
        
    # 코너를 찾지 못한 경우 (이미지가 잘렸거나, 조명이 나쁘거나, 체크보드 크기 설정이 틀린 경우)
    else:
        # 실패한 파일 이름을 화면에 출력하여 사용자에게 알림
        print(f"❌ 실패 (코너 못 찾음): {fname}")

# 전체 이미지 검출 반복문을 마친 후, 최종적으로 코너를 찾아낸 유효 이미지의 총 개수를 출력
print(f"\n▶ 최종적으로 코너 검출에 성공한 유효 이미지: {success_count}장")

# -----------------------------
# 2. 카메라 캘리브레이션 및 검증
# -----------------------------
# 코너 검출에 성공한 이미지가 단 한 장도 없는 경우, 캘리브레이션을 진행할 수 없으므로 에러 처리
if success_count == 0:
    # 에러 분석 헤더 출력
    print("\n[🚨 치명적 에러 분석 🚨]")
    
    # 이미지 파일 자체를 폴더에서 찾지 못한 경우
    if len(images) == 0:
        print("원인: 이미지를 한 장도 찾지 못했습니다.")
        print("해결책: 폴더 이름이 오타가 없는지, 바탕화면의 computer_vison/2 폴더 안에 images 폴더가 정확히 있는지 확인하세요.")
        
    # 이미지는 찾았으나 코너 검출에서 전부 실패한 경우 (보통 CHECKERBOARD 크기 오입력)
    else:
        print("원인: 이미지는 찾았지만 코너 검출에 100% 실패했습니다.")
        print("해결책: left01.jpg 사진을 열어보고 내부 코너(검은색과 흰색이 교차하는 십자가 지점) 개수를 직접 세어보세요.")
        print("가로 교차점 개수와 세로 교차점 개수를 세어서 코드 10번째 줄의 CHECKERBOARD = (가로, 세로) 로 수정해야 합니다.")

# 코너 검출에 성공한 이미지가 1장 이상이어서 정상적으로 캘리브레이션이 가능한 경우
else:
    # 캘리브레이션 연산 시작 알림 출력
    print("\n▶ 캘리브레이션 연산을 시작합니다...")

    # 모아둔 3D 대응 좌표(objpoints)와 2D 픽셀 좌표(imgpoints)를 이용해 카메라 내부 파라미터와 왜곡 계수를 계산
    # 반환값: ret(재투영 오차), K(내부 파라미터 3x3 행렬), dist(왜곡 계수 1x5 배열), rvecs(회전 벡터), tvecs(이동 벡터)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # 계산된 카메라 내부 파라미터 행렬(Camera Matrix K) 출력 알림
    print("\nCamera Matrix K:")
    # 실제 K 행렬 값 출력 (초점 거리 fx, fy 및 주점 px, py 포함)
    print(K)

    # 계산된 렌즈 왜곡 계수(Distortion Coefficients) 출력 알림
    print("\nDistortion Coefficients:")
    # 실제 dist 값 출력 (방사 왜곡 k1, k2, k3 및 접선 왜곡 p1, p2 포함)
    print(dist)
    
    # 결과 이미지를 저장할 'outputs' 폴더가 없을 경우, 에러를 방지하기 위해 자동으로 폴더 생성
    os.makedirs("outputs", exist_ok=True)
    
    # 계산된 파라미터가 잘 작동하는지 첫 번째 유효 이미지를 불러와서 보정 테스트 진행
    test_img = cv2.imread(images[0])
    
    # cv2.undistort 함수를 사용하여 렌즈 왜곡이 있는 원본 이미지(test_img)를 평평하게 펴주는 보정 연산 수행
    undistorted_img = cv2.undistort(test_img, K, dist, None, K)
    
    # 왜곡이 보정된 최종 결과 이미지를 'outputs' 폴더 안에 'undistorted_test.jpg'라는 이름으로 저장
    cv2.imwrite("outputs/undistorted_test.jpg", undistorted_img)
    
    # 테스트 이미지가 성공적으로 저장되었음을 터미널에 출력하여 프로그램 정상 종료 알림
    print("\n▶ 완료: 왜곡 보정 테스트 이미지가 'outputs/undistorted_test.jpg'로 저장되었습니다.")