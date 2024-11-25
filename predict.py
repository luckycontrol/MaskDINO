import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from maskdino import add_maskdino_config
import cv2
import numpy as np
from pathlib import Path

def setup_cfg(weights_path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)

    cfg.merge_from_file(r"D:\models\MaskDINO\configs\coco\instance-segmentation\maskdino_R50_bs16_50ep_3s.yaml")

    cfg.MODEL.WEIGHTS = weights_path

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    return cfg

def predict_image(predictor, image_path, output_path=None):
    img = cv2.imread(image_path)

    outputs = predictor(img)

    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()
    scores = instances.scores.numpy()
    confidence_threshold = 0.5

    if len(masks) > 0:
        mask_overlay = np.zeros_like(img)
        for mask, score in zip(masks, scores):
            if score >= confidence_threshold:  # 신뢰도가 높은 마스크만 처리
                mask = mask.astype(np.uint8)
                mask_overlay[mask > 0] = [0, 255, 0]  # Green color for masks
        
        # Blend the mask with original image
        alpha = 0.5
        result_image = cv2.addWeighted(img, 1, mask_overlay, alpha, 0)
    else:
        result_image = img.copy()

    if output_path:
        cv2.imwrite(output_path, result_image)
    
    return result_image

def main():
    weights_path = r"D:\models\MaskDINO\output\model_0004999.pth"

    cfg = setup_cfg(weights_path)

    predictor = DefaultPredictor(cfg)

    test_image_dir = r"D:\SFA_TEST"
    output_dir = "predictions"

    Path(output_dir).mkdir(exist_ok=True)

    for image_path in Path(test_image_dir).glob("*.png"):
        print(f"{image_path} 처리중..")
        output_path = Path(output_dir) / f"pred_{image_path.name}"

        try:
            result = predict_image(predictor, str(image_path), str(output_path))
            print(f"{output_path} 에 저장")
        except Exception as e:
            print(f"{image_path}: {e}")

if __name__ == "__main__":
    main()