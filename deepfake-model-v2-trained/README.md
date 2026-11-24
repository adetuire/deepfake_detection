# Deepfake detector v2

A compact binary image classifier trained from scratch on JPEG error level analysis (ELA) features. It summarizes residual intensity by region, color channel and position within an 8×8 JPEG block. The original CNN and v2 use the same quality-95 ELA transform. The original CNN takes the full residual image. V2 takes a 300-element summary, normalizes it using training data, then applies a 64-unit hidden layer.

## Run

Tested with Python 3.12.7. From this directory:

```sh
python -m pip install -r requirements.txt
python predict.py path/to/256x256.jpg
python train.py path/to/full_dataset.zip path/to/cnn_ela_model_weights.h5 --out reproduction
```

The model outputs a fake score. Both classifiers use a 0.5 threshold. The training command expects the original trusted full Keras H5 checkpoint and a ZIP containing full_dataset/{Real,Fake} JPEGs. The released v2 model can score an image without the original checkpoint or dataset.

## Evaluation design

Seed 2027 fixes a class-stratified 70/15/15 split of distinct decoded pixel hashes. Identical decoded images stay together. All 6,000 image files participate. Validation loss selects the v2 checkpoint. Both checkpoints score the exact same test filenames. The original notebook selected the first 1,500 files per class in filesystem order and its training manifest was unavailable. Some test images may therefore have been seen by the original model. Identity, source-video and near-duplicate overlap were not audited. JPEG compression differences between the two classes can affect performance on other datasets.

The run directory contains the trained checkpoint, split manifest, per-image predictions, baseline checkpoint hash and metrics. Real is 0 and Fake is 1.

## Measured results

Training used 4,200 images, validation used 900 and the comparison used 450 real and 450 fake images. Epoch 11 had the lowest validation loss.

| Checkpoint | Accuracy | Fake F1 | ROC AUC | Missed fakes | Real images flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original CNN | 57.56% | 0.499 | 0.599 | 260 | 122 |
| V2 | 61.22% | 0.605 | 0.669 | 183 | 166 |

At the shared threshold v2 detected 77 more fake images and flagged 44 more real images.
