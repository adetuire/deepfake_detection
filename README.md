# Deepfake detection

I trained a classifier to detect manipulated images using JPEG error level analysis and compared it with my original CNN on 900 images held out from v2 training.

At the same decision threshold, v2 detected 267 of 450 fake images, up from 190. It also flagged 166 of 450 real images, up from 122. ROC AUC increased from 0.599 to 0.669. The original model’s training file list was unavailable, so overlap with its training set is unknown.

[Browse the v2 code and evaluation](deepfake-model-v2-trained/) · [View the results](deepfake-model-v2-trained/run/results.json) · [Download the trained model release](https://github.com/adetuire/deepfake_detection/releases/tag/v2.0.0)
