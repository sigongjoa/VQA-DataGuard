import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_approximated_polygons(image):
    """
    주어진 segmentation 이미지(image)를 이용하여 approximated polygons(근사화된 다각형) 리스트를 반환합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        if gray.max() <= 1:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    binary_inv = 255 - binary
    contours, _ = cv2.findContours(binary_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approximated_polygons = []
    for cnt in contours:
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approximated_polygons.append(approx)
    return approximated_polygons

def get_largest_polygon(polygons):
    """
    근사화된 다각형 리스트(polygons) 중 면적이 가장 큰 다각형을 선택하여 (N,2) 배열로 반환합니다.
    """
    if not polygons:
        return None
    areas = [cv2.contourArea(poly.reshape(-1,1,2).astype(np.float32)) for poly in polygons]
    idx = np.argmax(areas)
    return polygons[idx].reshape(-1,2)

def get_radial_list(poly):
    """
    주어진 다각형(poly, shape: (N,2))에 대해, 
    다각형의 중심(centroid)에서 각 꼭짓점까지의 거리를 계산합니다.
    
    각 꼭짓점을 중심을 기준으로 한 각도 순서대로 정렬한 후,
    1D 계수(거리) 시퀀스를 반환합니다.
    
    Returns:
        np.array: 정렬된 계수 시퀀스 (N,)
    """
    pts = poly.reshape(-1, 2)
    center = np.mean(pts, axis=0)
    # 각 꼭짓점과 중심 사이의 거리
    radials = np.linalg.norm(pts - center, axis=1)
    # 각 꼭짓점의 각도 계산 (정렬 순서를 일관되게 하기 위함)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sorted_indices = np.argsort(angles)
    return radials[sorted_indices]

def dtw_distance_1d(seq1, seq2):
    """
    두 1D 시퀀스(seq1, seq2) 사이의 DTW (Dynamic Time Warping) 거리를 계산합니다.
    """
    N = len(seq1)
    M = len(seq2)
    dtw = np.full((N+1, M+1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, N+1):
        for j in range(1, M+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # insertion
                                   dtw[i, j-1],    # deletion
                                   dtw[i-1, j-1])  # match
    return dtw[N, M]



def draw_polygons_on_image(image, polygons, color=(0, 255, 0), thickness=2):
    """
    이미지 복사본에 폴리곤 리스트(컨투어)를 그립니다.
    """
    img_copy = image.copy()
    cv2.drawContours(img_copy, polygons, -1, color, thickness)
    return img_copy


def draw_center_lines(image, polygons, line_color=(0, 0, 255), thickness=2):
    """
    각 폴리곤에 대해, 폴리곤의 중심(centroid)에서 각 꼭짓점까지 직선을 그립니다.
    
    Args:
        image (np.array): 원본 이미지 (BGR)
        polygons (list): 근사화된 폴리곤 리스트 (각각 np.array)
        line_color (tuple): 선 색상 (B, G, R)
        thickness (int): 선 두께
    Returns:
        np.array: 선이 그려진 이미지
    """
    img_copy = image.copy()
    for poly in polygons:
        pts = poly.reshape(-1, 2)
        center = np.mean(pts, axis=0).astype(int)
        center_pt = tuple(center)
        for pt in pts:
            pt = tuple(pt)
            cv2.line(img_copy, center_pt, pt, line_color, thickness)
    return img_copy


def visualize_segmentation_center_lines(masks, silhouette):
    """
    두 segmentation 이미지(masks, silhouette)에 대해,
    원본 이미지에서 각각의 근사화된 다각형에 대해 중심(centroid)에서 꼭짓점까지 선을 그린 결과와
    가장 큰 다각형의 중심 기준 계수(거리) 시퀀스로 계산한 DTW Loss를 함께 시각화합니다.
    
    Args:
        masks (np.array): BGR segmentation 이미지 (마스크)
        silhouette (np.array): BGR segmentation 이미지 (실루엣; render_mesh_to_image_silhouette 결과)
    """
    # --- 1. 근사화된 다각형 추출 ---
    masks_polygons = get_approximated_polygons(masks)
    silhouette_polygons = get_approximated_polygons(silhouette)
    
    # --- 2. 원본 이미지에 중심선 그리기 (각각) ---
    masks_center_lines_img = draw_center_lines(masks, masks_polygons, line_color=(255, 0, 0), thickness=2)
    silhouette_center_lines_img = draw_center_lines(silhouette, silhouette_polygons, line_color=(255, 0, 0), thickness=2)
    
    # --- 3. DTW Loss 계산 ---
    largest_mask_poly = get_largest_polygon(masks_polygons)
    largest_silhouette_poly = get_largest_polygon(silhouette_polygons)
    
    if largest_mask_poly is None or largest_silhouette_poly is None:
        dtw_loss = None
        print("두 segmentation 중 하나에서 유효한 다각형을 찾을 수 없습니다.")
    else:
        mask_radials = get_radial_list(largest_mask_poly)
        silhouette_radials = get_radial_list(largest_silhouette_poly)
        dtw_loss = dtw_distance_1d(mask_radials, silhouette_radials)
        print("두 segmentation 계수 간 DTW Loss:", dtw_loss)
    
    # --- 4. matplotlib 시각화를 위해 BGR→RGB 변환 ---
    masks_center_lines_rgb = cv2.cvtColor(masks_center_lines_img, cv2.COLOR_BGR2RGB)
    silhouette_center_lines_rgb = cv2.cvtColor(silhouette_center_lines_img, cv2.COLOR_BGR2RGB)
    
    # --- 5. 결과 시각화 ---
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Mask with Center Lines")
    plt.imshow(masks_center_lines_rgb)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    if dtw_loss is not None:
        plt.title("Silhouette with Center Lines\n(DTW Loss: {:.2f})".format(dtw_loss))
    else:
        plt.title("Silhouette with Center Lines")
    plt.imshow(silhouette_center_lines_rgb)
    plt.axis("off")
    
    plt.suptitle("Segmentation Center Lines & DTW Loss", fontsize=16)
    plt.show()
