@echo off
docker run --gpus all ^
    --shm-size=12g ^
    --ipc=host ^
    -v d:/models/MaskDINO/datasets:/maskdino/datasets ^
    -v d:/models/MaskDINO/weights:/maskdino/weights ^
    -v d:/models/MaskDINO/output:/maskdino/output ^
    maskdino python train_net.py %*