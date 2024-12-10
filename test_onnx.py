import torch
import numpy as np
import onnxruntime as ort
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from maskdino.utils.device_utils import get_device
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from detectron2.data import transforms as T

def setup_cfg(weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)

    cfg.merge_from_file(r"D:\models\MaskDINO\configs\coco\instance-segmentation\maskdino_R50_bs16_50ep_3s.yaml")
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    device = get_device()
    cfg.MODEL.DEVICE = device.type

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2

    cfg.freeze()
    return cfg

def load_image(image_path, cfg):
    """DefaultPredictor의 이미지 전처리 방식을 정확히 구현한 함수"""
    # 이미지 로드 (BGR 포맷)
    original_image = cv2.imread(image_path)
    
    # BGR -> RGB 변환 (필요한 경우)
    if cfg.INPUT.FORMAT == "RGB":
        original_image = original_image[:, :, ::-1]
    
    # 원본 이미지 크기 저장
    height, width = original_image.shape[:2]
    
    # Detectron2의 ResizeShortestEdge 변환 적용
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
        cfg.INPUT.MAX_SIZE_TEST
    )
    image = cv2.resize(original_image, (1024, 1024))
    
    # [H, W, C] -> [C, H, W] 변환 및 float32로 변환
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    
    # 배치 차원 추가 [C, H, W] -> [1, C, H, W]
    image = image[None]
    
    return image, original_image

def visualize_predictions(original_image, pred_masks, pred_logits, pred_boxes, threshold=0.5, output_prefix=""):
    """예측 결과를 시각화하고 저장"""
    # 마스크의 형태를 원본 이미지 크기로 조정
    h, w = original_image.shape[:2]
    resized_masks = []
    for mask in pred_masks[0]:  # 첫 번째 배치의 마스크들
        mask = cv2.resize((mask > threshold).astype(np.uint8), (w, h))
        resized_masks.append(mask)
    
    # 신뢰도 점수 계산
    scores = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()[0, :, 0]
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 1. 원본 이미지
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # 2. 마스크와 바운딩 박스 오버레이
    plt.subplot(132)
    overlay = original_image.copy()
    mask_overlay = np.zeros_like(original_image)
    
    # 상위 5개 마스크 선택
    top_indices = np.argsort(scores)[::-1]
    colors = [(np.random.random(3) * 0.6 + 0.4) for _ in range(len(top_indices))]
    
    for idx, color in zip(top_indices, colors):
        if scores[idx] > threshold:
            # 마스크 오버레이
            mask = resized_masks[idx]
            mask_overlay[mask > 0] = np.array(color) * 255
            
            # # 바운딩 박스
            box = pred_boxes[0, idx] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            # cv2.rectangle(overlay, (x1, y1), (x2, y2), color * 255, 2)
            
            # 신뢰도 점수 표시
            cv2.putText(overlay, f"{scores[idx]:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color * 255, 2)
    
    overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Predictions Overlay")
    plt.axis('off')
    
    # 3. 신뢰도 점수 그래프
    plt.subplot(133)
    plt.bar(range(len(top_indices)), scores[top_indices])
    plt.title("Top 5 Confidence Scores")
    plt.xlabel("Mask Index")
    plt.ylabel("Confidence")
    
    # 결과 저장
    output_dir = Path("visualization_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / f"{output_prefix}_results.png")
    plt.close()

def test_onnx_model(image_path):
    # 설정 로드
    weights_path = r"D:\models\MaskDINO\training\output2\model_0019999.pth"
    cfg = setup_cfg(weights_path)
    
    # ONNX 런타임 세션 생성
    session = ort.InferenceSession("maskdino_model.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 입력 이미지 준비
    image, original_image = load_image(image_path, cfg)
    ort_inputs = {session.get_inputs()[0].name: image.numpy()}
    
    # ONNX 모델로 추론
    ort_outputs = session.run(None, ort_inputs)
    pred_logits, pred_masks, pred_boxes = ort_outputs
    
    # ONNX 모델 결과 시각화
    visualize_predictions(original_image, pred_masks, pred_logits, pred_boxes, output_prefix="onnx")
    
    # 결과 출력
    print("ONNX Model Output Shapes:")
    print(f"pred_logits shape: {pred_logits.shape}")
    print(f"pred_masks shape: {pred_masks.shape}")
    print(f"pred_boxes shape: {pred_boxes.shape}")
    
    # PyTorch 모델 설정 및 로드
    device = get_device()
    model = build_model(cfg)
    model.eval()
    
    # PyTorch 모델로 추론
    with torch.no_grad():
        torch_input = image.to(device)
        features = model.backbone(torch_input)
        torch_outputs, _ = model.sem_seg_head(features)
    
    # PyTorch 모델 결과 시각화
    visualize_predictions(
        original_image,
        torch_outputs['pred_masks'].cpu().numpy(),
        torch_outputs['pred_logits'].cpu().numpy(),
        torch_outputs['pred_boxes'].cpu().numpy(),
        output_prefix="pytorch"
    )
    
    # PyTorch 출력 형태 출력
    print("\nPyTorch Model Output Shapes:")
    print(f"pred_logits shape: {torch_outputs['pred_logits'].shape}")
    print(f"pred_masks shape: {torch_outputs['pred_masks'].shape}")
    print(f"pred_boxes shape: {torch_outputs['pred_boxes'].shape}")

if __name__ == "__main__":
    # 테스트할 이미지 경로 지정
    image_path = r"D:\models\MaskDINO\datasets\TEST_sfa\20002864_0.png"  # 실제 테스트 이미지 경로로 변경하세요
    test_onnx_model(image_path)
