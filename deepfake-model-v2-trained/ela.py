from io import BytesIO
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def prepare(data):
    with Image.open(BytesIO(data)) as source:
        if source.size != (256, 256):
            raise ValueError("Expected a 256 by 256 image")
        rgb = source.convert("RGB")
    encoded = BytesIO()
    rgb.save(encoded, format="JPEG", quality=95)
    encoded.seek(0)
    with Image.open(encoded) as jpeg:
        difference = ImageChops.difference(rgb, jpeg)
    peak = max(high for low, high in difference.getextrema())
    return np.asarray(ImageEnhance.Brightness(difference).enhance(255 / (peak or 1)))
