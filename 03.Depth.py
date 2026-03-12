import cv2 
import numpy as np 
from pathlib import Path 

# -----------------------------
# 0. 설정 및 데이터 로드
# -----------------------------
# 결과를 저장할 'outputs' 폴더 경로를 Path 객체로 지정
output_dir = Path("./outputs")
# 폴더가 존재하지 않으면 부모 디렉토리까지 포함하여 자동으로 생성 (exist_ok=True로 이미 있어도 에러 무시)
output_dir.mkdir(parents=True, exist_ok=True)

# OpenCV를 사용하여 왼쪽, 오른쪽 카메라에서 각각 촬영된 컬러 스테레오 이미지를 불러옴
left_color = cv2.imread("images/left.png")
right_color = cv2.imread("images/right.png")

# 경로 오류나 파일 누락으로 이미지를 정상적으로 불러오지 못한 경우 에러 발생
if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다. 'images' 폴더 내 파일명과 확장자(.png)를 확인하세요.")

# 카메라 캘리브레이션을 통해 미리 얻어진 내부/외부 파라미터 값 설정
f = 700.0 # 카메라의 초점 거리 (Focal length), 단위: 픽셀
B = 0.12 # 두 카메라 렌즈 중심 사이의 물리적 거리인 베이스라인 (Baseline), 단위: 미터(m)

# 깊이(Depth)를 분석하고 비교할 세 가지 특정 물체의 관심 영역(ROI, Region of Interest) 좌표 설정
# 포맷: "물체이름": (시작 x좌표, 시작 y좌표, 가로 너비 w, 세로 높이 h)
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# StereoBM(Block Matching) 알고리즘은 흑백 이미지에서 픽셀 밝기 패턴을 비교하므로 BGR 컬러를 Grayscale로 변환
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity(시차) 계산
# -----------------------------
# OpenCV의 Block Matching 스테레오 객체 생성
# numDisparities: 탐색할 최대 시차 범위 (반드시 16의 배수여야 함, 여기서는 64 픽셀까지 탐색)
# blockSize: 매칭에 사용할 픽셀 블록의 크기 (홀수여야 하며, 여기서는 15x15 윈도우 사용)
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

# 왼쪽 이미지와 오른쪽 이미지를 비교하여 각 픽셀이 얼만큼 가로로 이동했는지(Disparity) 계산
disparity_16 = stereo.compute(gray_left, gray_right)

# OpenCV의 compute 함수 결과는 내부 메모리 효율을 위해 실제 시차 값에 16이 곱해진 16비트 정수형(int16)으로 반환됨
# 따라서 정확한 픽셀 단위 시차를 얻으려면 실수형(float32)으로 변환 후 16.0으로 나누어 정규화해야 함
disparity = disparity_16.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth(깊이) 계산
# 공식: Z = (f * B) / d
# -----------------------------
# 계산된 Depth 값을 저장할 빈(0으로 채워진) 배열을 disparity와 동일한 크기와 타입(float32)으로 생성
depth_map = np.zeros_like(disparity, dtype=np.float32)

# 시차(disparity)가 0 이하인 경우는 매칭에 실패했거나 물체가 너무 멀리 있어 거리가 무한대(Z=∞)인 노이즈 픽셀임
# ZeroDivisionError(0으로 나누기 에러)를 방지하기 위해 0보다 큰 유효한 픽셀 위치만 True로 가지는 마스크 생성
valid_mask = disparity > 0

# 유효한 픽셀 위치(valid_mask가 True인 곳)에만 Z = (f * B) / d 공식을 적용하여 실제 미터(m) 단위 깊이 도출
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
# 각 물체별 계산 결과를 저장할 딕셔너리 초기화
results = {}

# 설정해둔 3개의 관심 영역(ROI)을 순회하며 데이터 추출
for name, (x, y, w, h) in rois.items():
    # 전체 disparity 맵과 depth 맵에서 해당 물체의 사각형 영역(y~y+h, x~x+w)만 Numpy 슬라이싱으로 잘라냄
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    
    # 해당 영역 내에서 에러 없이 정상적으로 계산된 픽셀 마스크 추출
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    # ROI 내에 유효한(계산된) 픽셀이 하나라도 존재하는 경우
    if np.any(roi_valid):
        # 유효한 픽셀들만의 disparity 평균값과 depth 평균값을 np.mean으로 계산
        mean_disp = np.mean(roi_disp[roi_valid])
        mean_depth = np.mean(roi_depth[roi_valid])
    else:
        # 매칭이 전혀 안 된 영역일 경우 평균값을 0.0으로 예외 처리
        mean_disp = 0.0
        mean_depth = 0.0
        
    # 물체 이름(name)을 키(key)로 하여 계산된 평균값 저장
    results[name] = {"disparity": mean_disp, "depth": mean_depth}

# -----------------------------
# 4. 결과 출력 및 해석
# -----------------------------
print("=== ROI 별 평균 Disparity 및 Depth ===")
# 딕셔너리에 저장된 결과물(물체별 시차 및 깊이)을 보기 좋게 포맷팅하여 출력 (소수점 제한)
for name, vals in results.items():
    print(f"- {name}: Disparity = {vals['disparity']:.2f} px, Depth = {vals['depth']:.3f} m")

# 물체 중 평균 Disparity 값이 가장 큰(즉, 카메라 렌즈에 가장 가까운) 물체의 이름을 찾음
closest_obj = max(results, key=lambda k: results[k]['disparity'])
# 물체 중 평균 Disparity 값이 가장 작은(즉, 카메라 렌즈에서 가장 멀리 있는) 물체의 이름을 찾음
farthest_obj = min(results, key=lambda k: results[k]['disparity'])

# 시차와 깊이의 반비례 관계를 바탕으로 최종 결론 출력
print(f"\n[해석 결론]")
print(f"Disparity가 가장 큰(Depth가 가장 작은) {closest_obj}가 카메라에서 가장 가깝습니다.")
print(f"Disparity가 가장 작은(Depth가 가장 큰) {farthest_obj}가 카메라에서 가장 멉니다.")

# -----------------------------
# 5. 시각화를 위한 정규화 및 저장 (제공된 스켈레톤 코드 기반)
# -----------------------------
# 원본 시차 데이터를 손상시키지 않기 위해 복사본 생성
disp_tmp = disparity.copy()
# 0 이하의 비정상 값은 계산에서 제외하기 위해 NaN(Not a Number) 처리
disp_tmp[disp_tmp <= 0] = np.nan

# 모든 값이 NaN일 경우 시각화가 불가능하므로 에러 발생
if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

# 극단적인 노이즈를 제외하기 위해 하위 5%와 상위 95%의 값을 최소/최대 기준으로 설정
d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

# 분모가 0이 되는 것을 방지
if d_max <= d_min:
    d_max = d_min + 1e-6

# Min-Max 정규화를 통해 값을 0.0 ~ 1.0 사이의 비율로 변환
disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

# 화면 출력을 위해 8비트(0~255) 배열 구조 생성
disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
# 정규화된 비율 데이터에 255를 곱해 최종 픽셀 밝기값 매핑
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

# 흑백 이미지를 온도 분포 형태의 컬러맵(COLORMAP_JET: 빨간색일수록 높은 값)으로 변환
disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# 깊이(Depth) 맵도 동일한 방식으로 정규화 및 시각화 준비
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]
    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)
    
    # 깊이 맵은 시차 맵과 반대로 거리가 멀수록(값이 클수록) 어둡게 보이도록 반전(Invert)
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

# 반전된 깊이 맵 데이터를 컬러맵으로 변환
depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 결과물에 ROI 영역을 시각적으로 표시하기 위해 원본 이미지 복사
left_vis = left_color.copy()
right_vis = right_color.copy()

# 각 ROI 영역별로 초록색 사각형 테두리와 물체 이름을 이미지 위에 그림
for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 최종 완성된 3장의 결과 이미지(컬러 시차 맵, 컬러 깊이 맵, ROI 마킹 이미지)를 outputs 폴더에 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)