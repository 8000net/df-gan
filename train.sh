#!/bin/sh

echo "Training DeepFace GAN Detection Model"
python dfgan.py --mode train --batch_size 64 
