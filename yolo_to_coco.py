import json
import numpy as np
from pathlib import Path
from PIL import Image
import glob

def yolo_to_coco_segmentation(image_folder):
    # COCO 형식의 기본 구조 생성
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 카테고리 정보 추가 (예시)
    categories = [{"id": 0, "name": "object", "supercategory": "none"}]
    coco_format["categories"] = categories
    
    # 이미지 ID 초기화
    image_id = 1
    annotation_id = 1
    
    # PNG 파일 목록 가져오기
    png_files = glob.glob(str(Path(image_folder) / "*.png"))
    
    for png_file in png_files:
        # 이미지 파일 경로
        image_path = Path(png_file)
        # 라벨 파일 경로
        label_path = Path(image_folder) / "labels" / f"{image_path.stem}.txt"
        
        if not label_path.exists():
            print(f"Warning: No label file for {image_path.name}")
            continue
            
        # 이미지 크기 읽기
        with Image.open(image_path) as img:
            width, height = img.size
        
        # 이미지 정보 추가
        coco_format["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path.name
        })
        
        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if not line.strip():
                continue
                
            # YOLO 형식 파싱
            values = line.strip().split()
            class_id = int(values[0])
            coordinates = list(map(float, values[1:]))
            
            # x,y 좌표쌍으로 변환
            points = []
            for i in range(0, len(coordinates), 2):
                points.append([coordinates[i], coordinates[i+1]])
                
            # COCO 형식으로 변환
            # YOLO는 normalized 좌표(0~1)를 사용하므로 실제 픽셀 좌표로 변환
            segmentation = []
            for point in points:
                segmentation.extend([
                    point[0] * width,
                    point[1] * height
                ])
                
            # 바운딩 박스 계산
            x_coords = [p[0] * width for p in points]
            y_coords = [p[1] * height for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # annotation 추가
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id,
                "segmentation": [segmentation],
                "area": bbox_width * bbox_height,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            }
            
            coco_format["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    return coco_format

# 사용 예시
folder_path = r"D:\SFA_TEST"  # A 폴더 경로
coco_data = yolo_to_coco_segmentation(folder_path)

# JSON 파일로 저장
output_path = Path(folder_path) / "coco_annotations.json"
with open(output_path, 'w') as f:
    json.dump(coco_data, f, indent=2)
