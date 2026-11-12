# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "tifffile", "imagecodecs", "pillow"]
# ///
"""Find individual trees in the canopy height model (Environment Agency lidar surface minus terrain).

Output (data/build/):
  trees.bin   float32 per tree: x, z, height, crown radius, crown base, kind, r, g, b, seed
  chm.npy     canopy height model at LIDAR_RES (row 0 north), used for hedges and shadows
"""

import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from aerial import load_aerial
from config import BUILD, CX, CY, GROUND_RES, LIDAR_RES, XMIN, YMAX
from lidar import surface_and_terrain

RES = LIDAR_RES


def young_woods(chm, rng, spacing=5.0):
    img = Image.new("L", (chm.shape[1], chm.shape[0]), 0)
    d = ImageDraw.Draw(img)
    for f in json.loads((BUILD / "surfaces.json").read_text()):
        if f["style"] == "forest":
            d.polygon([((x - XMIN) / RES, (YMAX - y) / RES) for x, y in f["rings"][0]], fill=1)
            for hole in f["rings"][1:]:
                d.polygon([((x - XMIN) / RES, (YMAX - y) / RES) for x, y in hole], fill=0)
    wood = np.array(img) > 0
    bare = wood & (ndi.uniform_filter(chm, int(8 / RES)) < 1.0)
    step = spacing / RES
    gy, gx = np.mgrid[0 : chm.shape[0] : step, 0 : chm.shape[1] : step]
    gx = gx.ravel() + rng.uniform(-0.4, 0.4, gx.size) * step
    gy = gy.ravel() + rng.uniform(-0.4, 0.4, gy.size) * step
    ok = (gx >= 0) & (gy >= 0) & (gx < chm.shape[1]) & (gy < chm.shape[0])
    gx, gy = gx[ok], gy[ok]
    keep = bare[gy.astype(int), gx.astype(int)]
    gx, gy = gx[keep], gy[keep]
    n = len(gx)
    h = rng.uniform(2.2, 4.5, n)
    x, y = XMIN + gx * RES, YMAX - gy * RES
    print(f"{n} saplings in young woods")
    return np.stack([x - CX, CY - y, h, 0.6 + 0.25 * h, np.full(n, 0.8), np.zeros(n), np.full(n, 0.42), np.full(n, 0.58),
                     np.full(n, 0.3), rng.random(n)], -1).astype(np.float32)


def footprint_mask(shape, grow_m):
    img = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    for ring in json.load(open(BUILD / "footprints.json")):
        d.polygon([((x + CX - XMIN) / RES, (YMAX - (CY - z)) / RES) for x, z in ring], fill=255)
    m = np.array(img) > 0
    return ndi.binary_dilation(m, iterations=max(1, int(grow_m / RES)))


def main():
    dsm, dtm = surface_and_terrain()
    chm = np.nan_to_num(dsm - dtm, nan=0.0)
    chm[chm < 0] = 0
    chm[footprint_mask(chm.shape, 1.0)] = 0
    np.save(BUILD / "chm.npy", chm.astype(np.float16))

    smooth = ndi.gaussian_filter(chm, 0.6 / RES)
    canopy = smooth > 2.5
    peaks = (smooth == ndi.maximum_filter(smooth, size=max(3, int(2.5 / RES) | 1))) & (smooth > 3.5)
    ys, xs = np.nonzero(peaks)
    hs = smooth[ys, xs]

    # Greedy suppression: taller trees claim a radius that grows with their height.
    order = np.argsort(-hs)
    taken = np.zeros_like(peaks)
    keep = []
    for i in order:
        y, x, h = ys[i], xs[i], hs[i]
        if taken[y, x]:
            continue
        keep.append(i)
        r = int(np.clip(0.12 * h + 1.0, 1.5, 4.5) / RES)
        taken[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1] = True
    ys, xs, hs = ys[keep], xs[keep], hs[keep]

    markers = np.zeros(chm.shape, dtype=np.int32)
    markers[ys, xs] = np.arange(1, len(ys) + 1)
    labels = watershed(-smooth, markers, mask=canopy)
    areas = ndi.sum_labels(np.ones_like(labels), labels, index=np.arange(1, len(ys) + 1)) * RES * RES
    radius = np.clip(np.sqrt(areas / np.pi), 1.0, 7.5)

    # Crown color from the Sentinel-2 image. At 10 m it cannot tell evergreens apart, so every tree is deciduous.
    aerial = load_aerial().astype(np.float32)
    scale = int(RES / GROUND_RES)
    small = aerial.reshape(aerial.shape[0] // scale, scale, aerial.shape[1] // scale, scale, 3).mean(axis=(1, 3))
    cols = np.stack([ndi.median(small[..., c], labels, index=np.arange(1, len(ys) + 1)) for c in range(3)], -1)
    cols = np.nan_to_num(cols, nan=90.0)
    r, g, b = cols[:, 0], cols[:, 1], cols[:, 2]
    evergreen = np.zeros(len(hs), bool)

    rng = np.random.default_rng(7)
    wx = XMIN + (xs + 0.5) * RES
    wy = YMAX - (ys + 0.5) * RES
    kind = evergreen.astype(np.float32)
    base = np.clip(hs * rng.uniform(0.25, 0.4, len(hs)), 1.2, 6)
    trees = np.stack(
        [wx - CX, CY - wy, hs, radius, base, kind, r / 255, g / 255, b / 255, rng.random(len(hs))], -1
    ).astype(np.float32)

    # Single trees from OpenStreetMap that the canopy model missed (young street trees).
    pts = np.array([p[:2] for p in json.loads((BUILD / "osm_trees.json").read_text())]).reshape(-1, 2)
    have = np.zeros(chm.shape, bool)
    have[ys, xs] = True
    have = ndi.binary_dilation(have, iterations=int(3 / RES))
    extra = []
    for x, y in pts:
        col, row = int((x - XMIN) / RES), int((YMAX - y) / RES)
        if not (0 <= col < chm.shape[1] and 0 <= row < chm.shape[0]) or have[row, col]:
            continue
        h = max(float(chm[row, col]), 4.0 + rng.random() * 3)
        extra.append([x - CX, CY - y, h, 1.2 + h * 0.15, 1.5, 0, 0.35, 0.42, 0.25, rng.random()])
    if extra:
        trees = np.concatenate([trees, np.array(extra, dtype=np.float32)])
    # Woods that OpenStreetMap maps but the 2022 lidar finds no canopy in are young plantings (much of the area lies in
    # the Forest of Marston Vale, planted since the 1990s): saplings on a loose 5 m grid.
    young = young_woods(chm, rng)
    if len(young):
        trees = np.concatenate([trees, young])
    # Masts show up in the canopy model as very tall, thin "trees".
    lm = BUILD / "landmarks.json"
    for m in json.loads(lm.read_text()).get("masts", []) if lm.exists() else []:
        trees = trees[np.hypot(trees[:, 0] - (m["x"] - CX), trees[:, 1] - (CY - m["y"])) > 6]
    trees.tofile(BUILD / "trees.bin")
    print(f"{len(trees)} trees ({len(extra)} from OpenStreetMap), {int(kind.sum())} evergreen, mean radius {radius.mean():.1f} m")


if __name__ == "__main__":
    main()
