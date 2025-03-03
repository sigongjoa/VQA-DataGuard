import torch
import numpy as np
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import matplotlib.pyplot as plt

def segment_image(image, processor, model, target_size=None):
    """
    주어진 이미지에 대해 Mask2Former 모델을 사용하여 panoptic segmentation을 수행합니다.
    
    Parameters:
        image: 입력 이미지. numpy array 또는 PIL 이미지.
        processor: 사전 학습된 AutoImageProcessor.
        model: 사전 학습된 Mask2FormerForUniversalSegmentation 모델.
        target_size (tuple, optional): segmentation 후 결과의 타겟 사이즈 (높이, 너비).
                                     제공되지 않을 경우, image의 크기를 사용합니다.
    
    Returns:
        segmentation: 후처리된 panoptic segmentation 결과.
    """
    # target_size가 제공되지 않은 경우, 이미지의 크기를 사용
    if target_size is None:
        # numpy 배열인 경우 shape가 (H, W, C)
        if hasattr(image, "shape"):
            target_size = (image.shape[0], image.shape[1])
        else:
            # PIL 이미지인 경우 size가 (W, H)이므로 반전
            target_size = image.size[::-1]
    
    # 이미지 전처리 및 tensor 변환
    inputs = processor(images=image, return_tensors="pt")
    
    # 모델 예측 (추론 모드)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 후처리: panoptic segmentation 결과 생성
    segmentation = processor.post_process_panoptic_segmentation(outputs, target_sizes=[target_size])
    
    return segmentation
    
def draw_top_segment(segmentation, image):
    """
    segmentation: 딕셔너리, keys: 'segmentation' (tensor)와 'segments_info' (리스트)
    image: 원본 이미지 (NumPy 배열, [H, W, 3]), 값 범위 0~255
    """
    # segmentation['segmentation']는 (H, W) tensor -> NumPy 배열로 변환
    seg_map = segmentation['segmentation'].cpu().numpy()
    segments_info = segmentation['segments_info']
    
    # 가장 높은 score를 가진 segment 찾기
    top_seg = max(segments_info, key=lambda seg: seg["score"])
    score = top_seg["score"]
    seg_id = top_seg["id"]
    label_id = top_seg["label_id"]
    # 미리 정의한 mapping을 이용해 label 이름 가져오기 (없으면 숫자 그대로 사용)
    
    # 해당 segment에 대한 boolean mask 생성
    mask = seg_map == seg_id

    # 원본 이미지 정규화 (0~1)
    image_norm = image.astype(np.float32) / 255.0
    
    # 임의의 색상 선택 (오버레이에 사용할 색상)
    color = 0
    
    # 색상 오버레이 생성: mask 영역에만 선택한 색상 적용, 나머지 영역은 흰색으로 설정
    overlay = np.ones_like(image_norm)  # 기본 배경을 흰색으로 (1.0)
    overlay[mask] = color
    
    return overlay
