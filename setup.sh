#!/bin/bash

pip install -r requirements.txt
MODELS_DIR=../../models
mkdir $MODELS_DIR/flashface
mkdir $MODELS_DIR/face_detection
wget -O $MODELS_DIR/flashface/flashface.ckpt https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/flashface.ckpt?download=true
wget -O $MODELS_DIR/vae/sd-v1-vae.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/sd-v1-vae.pth?download=true
wget -O $MODELS_DIR/clip/openai-clip-vit-large-14.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/openai-clip-vit-large-14.pth?download=true
wget -O $MODELS_DIR/clip/bpe_simple_vocab_16e6.txt.gz https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/bpe_simple_vocab_16e6.txt.gz?download=true
wget -O $MODELS_DIR/face_detection/retinaface_resnet50.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth?download=true