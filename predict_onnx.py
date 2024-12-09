import onnxruntime
import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지 전처리
    image = cv2.resize(image, (1024, 1024))
    image = image.astype(np.float32) / 255.0
    
    # 정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # 차원 변경 (B, C, H, W)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    
    return image

def visualize_results(image, masks, scores, labels, output_path):
    # 원본 이미지로 변환
    image = image[0].transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image = (image * 255).astype(np.uint8)
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # 마스크 시각화
    if len(masks.shape) == 4:  # [batch, num_instances, height, width]
        masks = masks[0]  # 첫 번째 배치만 사용
    
    if len(scores.shape) > 1:
        scores = scores[0]  # 첫 번째 배치만 사용
    
    if len(labels.shape) > 1:
        labels = labels[0]  # 첫 번째 배치만 사용
    
    # 각 마스크에 대해 시각화
    for idx, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        if score > 0.5:  # confidence threshold
            plt.contour(mask, colors='r', alpha=0.5)
            plt.text(10, 20 + idx * 20, f'Class {label}: {score:.2f}', color='white', 
                    bbox=dict(facecolor='red', alpha=0.5))
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    # ONNX 모델 로드
    onnx_path = "maskdino_model.onnx"
    session = onnxruntime.InferenceSession(onnx_path)
    
    # 모델 정보 출력
    print("Input Info:")
    for input in session.get_inputs():
        print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")
    
    print("\nOutput Info:")
    for output in session.get_outputs():
        print(f"Name: {output.name}, Shape: {output.shape}, Type: {output.type}")
    
    # 테스트 이미지 경로
    image_path = r"D:\models\MaskDINO\datasets\TEST_sfa\20002864_0.png"  # 실제 테스트할 이미지 경로로 변경해주세요
    
    # 이미지 전처리
    input_image = preprocess_image(image_path)
    
    # float32 타입으로 명시적 변환
    input_image = input_image.astype(np.float32)
    
    # 입력 이름 가져오기
    input_name = session.get_inputs()[0].name
    
    # 추론 실행
    outputs = session.run(None, {input_name: input_image})
    
    # 결과 후처리 (예시 - 실제 모델의 출력 형식에 따라 수정 필요)
    masks = outputs[0]  # 마스크
    scores = outputs[1]  # 점수
    labels = outputs[2]  # 레이블
    
    # 결과 시각화 및 저장
    output_path = "result.png"
    visualize_results(input_image, masks, scores, labels, output_path)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main()
