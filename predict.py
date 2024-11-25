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

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get("custom_train"), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = out.get_image()[:, :, ::-1]

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