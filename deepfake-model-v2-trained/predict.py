"""Score a single 256x256 JPEG with the released v2 checkpoint."""
import argparse
from pathlib import Path

from tensorflow import keras
from ela import prepare
from train import features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', default='run/v2.keras')
    args = parser.parse_args()
    vector = features(prepare(args.image.read_bytes()))
    model = keras.models.load_model(args.model, compile=False)
    score = float(model(vector[None], training=False).numpy().item())
    print(f"fake_score={score:.6f} prediction={'Fake' if score >= .5 else 'Real'}")
