import copy
import logging
import numpy as np
import torch
import albumentations as A
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode, Instances
import cv2
import pycocotools.mask as mask_util
from pathlib import Path
import json

class AlbumentationMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.mask_on = True
        self.img_format = cfg.INPUT.FORMAT
        self.output_dir = Path("augmented_data")
        self.output_dir.mkdir(exist_ok=True)

        self.coco_output = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "object"}
            ]
        }

        self.image_id = 0
        self.annotation_id = 0

        self.transform = A.Compose([
            A.HorizontalFlip()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
           mask_params=A.MaskParams(mask_rules=[]))

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        if not self.is_train:
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            return dataset_dict

        # Prepare masks and boxes
        masks = []
        boxes = []
        category_ids = []
        
        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if "segmentation" in anno:
                    # Convert segmentation to binary mask
                    if isinstance(anno["segmentation"], dict):
                        # RLE format
                        mask = mask_util.decode(anno["segmentation"])
                    else:
                        # Polygon format
                        mask = np.zeros(image.shape[:2], dtype=np.uint8)
                        for seg in anno["segmentation"]:
                            pts = np.array(seg).reshape((-1, 2))
                            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
                    masks.append(mask)
                    boxes.append(anno["bbox"])
                    category_ids.append(anno["category_id"])

        # Apply augmentation
        if masks and boxes:
            transformed = self.transform(
                image=image,
                masks=masks,
                bboxes=boxes,
                category_ids=category_ids
            )

            image = transformed["image"]
            transformed_masks = transformed["masks"]
            transformed_boxes = transformed["bboxes"]

            # Save augmented image and create COCO annotations
            image_filename = f"augmented_{self.image_id}.jpg"
            cv2.imwrite(str(self.output_dir / image_filename), image)

            # Add image info to COCO format
            self.coco_output["images"].append({
                "id": self.image_id,
                "file_name": image_filename,
                "height": image.shape[0],
                "width": image.shape[1]
            })

            # Create new annotations
            new_annotations = []
            for mask, bbox, cat_id in zip(transformed_masks, transformed_boxes, category_ids):
                # Convert mask to RLE format
                rle = mask_util.encode(np.asarray(mask, order="F"))
                area = float(mask_util.area(rle))
                
                annotation = {
                    "id": self.annotation_id,
                    "image_id": self.image_id,
                    "category_id": cat_id,
                    "segmentation": rle,
                    "area": area,
                    "bbox": [float(x) for x in bbox],
                    "iscrowd": 0
                }
                new_annotations.append(annotation)
                self.coco_output["annotations"].append(annotation)
                self.annotation_id += 1

            # Update dataset dict with augmented data
            dataset_dict["annotations"] = new_annotations
            self.image_id += 1

            # Save COCO annotations periodically
            if self.image_id % 100 == 0:
                self._save_coco_annotations()

        # Convert to tensor format for training
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        
        if len(new_annotations) > 0:
            instances = utils.annotations_to_instances(
                new_annotations, image.shape[:2], mask_format="bitmask"
            )
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def _save_coco_annotations(self):
        """COCO 형식의 어노테이션을 JSON 파일로 저장"""
        output_file = self.output_dir / "coco_annotations.json"
        with open(output_file, 'w') as f:
            json.dump(self.coco_output, f)

    def __del__(self):
        """소멸자에서 최종 COCO 어노테이션 저장"""
        self._save_coco_annotations()

