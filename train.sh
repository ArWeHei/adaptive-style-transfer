#!/bin/bash
CUDA_VISIBLE_DEVICES=9 python3 main.py \
--model_name=model_roerich_new \
--batch_size=1 \
--phase=train \
--image_size=768 \
--lr=0.0002 \
--dsr=0.8 \
--ptcd=../val_large \
--ptad=../nicholas-roerich
