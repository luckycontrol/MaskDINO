import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from maskdino.utils.device_utils import get_device
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
import torch.nn.functional as F

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

def convert_to_onnx():
    # 모델 설정 및 가중치 로드
    weights_path = r"D:\models\MaskDINO\training\output2\model_0019999.pth"
    cfg = setup_cfg(weights_path)
    model = build_model(cfg)
    model.eval()

    # 더미 입력 생성 (입력 크기는 모델에 맞게 조정 필요)
    dummy_tensor = torch.randn(1, 3, 1024, 1024, device=get_device())
    
    # ONNX 내보내기를 위한 forward wrapper 생성
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            # 백본을 통한 특징 추출
            features = self.model.backbone(x)
            # sem_seg_head를 통한 예측
            outputs, _ = self.model.sem_seg_head(features)
            
            return outputs["pred_logits"], outputs["pred_masks"], outputs["pred_boxes"]
    
    wrapped_model = ModelWrapper(model)
    
    # ONNX 내보내기
    torch.onnx.export(
        wrapped_model,
        dummy_tensor,
        "maskdino_model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred_logits', 'pred_masks', 'pred_boxes'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'pred_logits': {0: 'batch_size'},
            'pred_masks': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    convert_to_onnx()