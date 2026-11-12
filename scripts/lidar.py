"""Environment Agency lidar mosaics (data/raw/lidar/, from scripts/fetch.py) at LIDAR_RES, row 0 at the north edge."""

import logging

import numpy as np
import tifffile
from scipy import ndimage as ndi

from config import CHUNK, LIDAR_RES, NX, NY, RAW

# The lidar marks no-data with -3.4e38, which tifffile cannot cast and warns about; the value is filtered below anyway.
logging.getLogger("tifffile").setLevel(logging.ERROR)


def mosaic(kind):
    px = int(CHUNK / LIDAR_RES)
    out = np.full((NY * px, NX * px), np.nan, dtype=np.float32)
    for ix in range(NX):
        for iy in range(NY):
            a = tifffile.imread(RAW / "lidar" / f"{kind}_{ix}_{iy}.tif").astype(np.float32)
            a[(a > 2000) | (a < -100)] = np.nan
            row = (NY - 1 - iy) * px
            out[row : row + px, ix * px : ix * px + px] = a[:px, :px]
    return out


def fill_nearest(a):
    bad = np.isnan(a)
    if not bad.any():
        return a
    idx = ndi.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def surface_and_terrain():
    """(dsm, dtm) with gaps in the terrain filled from the nearest measurement."""
    return mosaic("dsm"), fill_nearest(mosaic("dtm"))
