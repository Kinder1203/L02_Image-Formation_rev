import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기 (경로 수정)
left_color = cv2.imread("images/left.png")
right_color = cv2.imread("images/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")

# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
gray_left = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
# numDisparities는 16의 배수여야 함
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity_16 = stereo.compute(gray_left, gray_right)

# 16배 스케일업 되어 있으므로 float로 변환 후 16.0으로 나눔 (실제 픽셀 시차)
disparity = disparity_16.astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)

# disparity가 0 이하인 부분(매칭 실패 혹은 무한대 거리)은 계산 제외
valid_mask = disparity > 0
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    # ROI 영역 슬라이싱
    roi_disp = disparity[y:y+h, x:x+w]
    roi_depth = depth_map[y:y+h, x:x+w]
    roi_valid = valid_mask[y:y+h, x:x+w]
    
    if np.any(roi_valid):
        mean_disp = np.mean(roi_disp[roi_valid])
        mean_depth = np.mean(roi_depth[roi_valid])
    else:
        mean_disp = 0.0
        mean_depth = 0.0
        
    results[name] = {"disparity": mean_disp, "depth": mean_depth}

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI 별 평균 Disparity 및 Depth ===")
for name, vals in results.items():
    print(f"- {name}: Disparity = {vals['disparity']:.2f} px, Depth = {vals['depth']:.3f} m")

# 논리적 해석 (가장 가까운 물체와 먼 물체 찾기)
closest_obj = max(results, key=lambda k: results[k]['disparity'])
farthest_obj = min(results, key=lambda k: results[k]['disparity'])

print(f"\n[해석 결론]")
print(f"Disparity가 가장 큰(Depth가 가장 작은) {closest_obj}가 카메라에서 가장 가깝습니다.")
print(f"Disparity가 가장 작은(Depth가 가장 큰) {farthest_obj}가 카메라에서 가장 멉니다.")

# -----------------------------
# 5. disparity 시각화 ~ 9. 출력 (제공된 스켈레톤 그대로 유지)
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

left_vis = left_color.copy()
right_vis = right_color.copy()

for name, (x, y, w, h) in rois.items():
    cv2.rectangle(left_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(left_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.rectangle(right_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(right_vis, name, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 저장
cv2.imwrite(str(output_dir / "disparity_map.png"), disparity_color)
cv2.imwrite(str(output_dir / "depth_map.png"), depth_color)
cv2.imwrite(str(output_dir / "roi_left.png"), left_vis)