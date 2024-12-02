@echo off
docker run --gpus all ^
    -v d:/models/MaskDINO/datasets:/maskdino/datasets ^
    -v d:/models/MaskDINO/weights:/maskdino/weights ^
    -v d:/models/MaskDINO/output:/maskdino/output ^
    maskdino:latest python train_net.py %*