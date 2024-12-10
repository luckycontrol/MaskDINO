import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from maskdino import add_maskdino_config
from maskdino.utils.device_utils import get_device
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
import detectron2.data.transforms as T
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
from detectron2.data.transforms import ResizeShortestEdge

class MaskDINOONNXWrapper(torch.nn.Module):
    def __init__(self, model, cfg):
        super(MaskDINOONNXWrapper, self).__init__()
        self.model = model
        self.cfg = cfg
        self.min_size = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST
    
    def forward(self, x):
        # x is a tensor of shape (B, 3, H, W)
        # Apply resizing
        original_size = x.shape[-2:]
        # Calculate the scale factor for resizing
        min_original_size = float(torch.min(torch.tensor(original_size)))
        max_original_size = float(torch.max(torch.tensor(original_size)))
        scale_factor = self.min_size / min_original_size
        # Calculate the new size
        if max_original_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_original_size
        new_size = (int(scale_factor * original_size[0]), int(scale_factor * original_size[1]))
        # Resize the image tensor
        resized_x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
        # Convert to BGR if necessary
        if self.cfg.INPUT.FORMAT == "BGR":
            resized_x = resized_x[:, [2, 1, 0], :, :]
        # Create input list
        inputs = []
        for i in range(resized_x.shape[0]):
            image = resized_x[i]
            height, width = image.shape[1], image.shape[2]
            inputs.append({"image": image, "height": height, "width": width})
        # Forward pass
        predictions = self.model(inputs)
        instances = predictions[0]['instances'].to("cpu")
        # Collect outputs
        pred_masks = instances.pred_masks.numpy()
        pred_boxes = instances.pred_boxes.tensor.detach().numpy()
        scores = instances.scores.detach().numpy()
    
        # Return as a tuple
        return pred_masks, pred_boxes, scores

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
    
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Create dummy input tensor with batch size 1
    dummy_input = torch.randn(1, 3, 1024, 1024, requires_grad=True).to(cfg.MODEL.DEVICE)
    
    # Create the wrapper
    wrapper = MaskDINOONNXWrapper(model, cfg)
    
    # Export the wrapper
    torch.onnx.export(
        wrapper,
        dummy_input,
        "maskdino_model.onnx",
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['pred_masks', 'pred_boxes', 'scores'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'pred_masks': {0: 'batch_size'},
            'pred_boxes': {0: 'batch_size'},
            'scores': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    convert_to_onnx()