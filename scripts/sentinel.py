"""Warp a Sentinel-2 true-colour image (a COG URL) onto the map grid at GROUND_RES, saved in CHUNK tiles under
data/raw/aerial/ as if it were an aerial photo. At 10 m it only colours fields and gardens; it has no roof detail.

Contains modified Copernicus Sentinel data, processed by ESA (free and open, with attribution)."""

import sys

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from config import CHUNK, CRS, GROUND_RES, NX, NY, RAW, XMIN, YMAX, YMIN


GAIN = 2.1


def main(url):
    out = RAW / "aerial"
    out.mkdir(parents=True, exist_ok=True)
    w, h = int(NX * CHUNK / GROUND_RES), int(NY * CHUNK / GROUND_RES)
    dst = np.zeros((3, h, w), np.uint8)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(url) as src:
        for b in range(3):
            reproject(rasterio.band(src, b + 1), dst[b], dst_transform=from_origin(XMIN, YMAX, GROUND_RES, GROUND_RES),
                      dst_crs=CRS, resampling=Resampling.cubic)
    # The true-colour product is dark (reflectance 0-0.3 over 0-255); brighten it toward the map's palette.
    img = np.clip(np.moveaxis(dst, 0, -1).astype(np.float32) * GAIN, 0, 255).astype(np.uint8)
    px = int(CHUNK / GROUND_RES)
    for ix in range(NX):
        for iy in range(NY):
            row = (NY - 1 - iy) * px
            Image.fromarray(img[row : row + px, ix * px : ix * px + px]).save(out / f"{ix}_{iy}.png")
    print(f"sentinel-2 mosaic {w}x{h} at {GROUND_RES} m", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
