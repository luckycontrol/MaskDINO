import torch

def get_device():
    """
    사용 가능한 최적의 디바이스를 반환합니다.
    Returns:
        torch.device: 사용할 디바이스
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cpu")

def move_to_device(model, device=None):
    """
    모델을 지정된 디바이스로 이동합니다.
    Args:
        model: 이동할 PyTorch 모델
        device: 대상 디바이스 (None인 경우 자동 선택)
    Returns:
        이동된 모델
    """
    if device is None:
        device = get_device()
    return model.to(device)