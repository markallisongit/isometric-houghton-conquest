"""British National Grid <-> WGS84 (exact, with pyproj) and map projections."""

import math

import numpy as np
from pyproj import Transformer

from config import CRS, CX, CY, XMAX, XMIN, YMAX, YMIN

HX, HZ = (XMAX - XMIN) / 2, (YMAX - YMIN) / 2

_TO_GRID = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_TO_WGS = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def wgs_to_grid(lat, lon):
    x, y = _TO_GRID.transform(lon, lat)
    return float(x), float(y)


def grid_to_wgs(x, y):
    lon, lat = _TO_WGS.transform(x, y)
    return float(lat), float(lon)


def _terms(a, b):
    """Cubic polynomial terms in two normalized variables."""
    return np.stack([np.ones_like(a), a, b, a * a, a * b, b * b, a**3, a * a * b, a * b * b, b**3], -1)


def js_fit():
    """Cubic fits both ways over the area (plus a margin), for the viewer, which has no projection library.
    Grid meters are normalized as (x - cx) / 1000, lat/lon as (v - v0) * 100; errors stay well under 10 cm."""
    gx, gy = np.meshgrid(np.linspace(XMIN - 2000, XMAX + 2000, 25), np.linspace(YMIN - 2000, YMAX + 2000, 25))
    gx, gy = gx.ravel(), gy.ravel()
    lon, lat = _TO_WGS.transform(gx, gy)
    lat0, lon0 = grid_to_wgs(CX, CY)
    A = _terms((gx - CX) / 1000, (gy - CY) / 1000)
    B = _terms((lat - lat0) * 100, (lon - lon0) * 100)
    fit = lambda M, v: [float(c) for c in np.linalg.lstsq(M, v, rcond=None)[0]]
    return {"lat0": lat0, "lon0": lon0, "lat": fit(A, lat - lat0), "lon": fit(A, lon - lon0),
            "x": fit(B, gx - CX), "y": fit(B, gy - CY)}


def screen(x, z, h=0.0, look=45):
    """Local meters (x east, z south, h up) to the render's screen meters (u right, v up) for a camera looking `look`
    degrees from north, 30 degrees down (render/scene.js). Works on numbers and numpy arrays alike."""
    a = math.radians(look)
    return x * math.cos(a) + z * math.sin(a), (x * math.sin(a) - z * math.cos(a)) * 0.5 + h * math.cos(math.radians(30))


def grid_to_image(x, y, scale=4, height_m=0.0):
    """Pixel position in the full isometric render of a grid point at a height above ground (see render/scene.js)."""
    width = round(2 * (HX + HZ) / math.sqrt(2) * scale)
    lx, lz = x - CX, CY - y
    u = (lx + lz) / math.sqrt(2)
    v = (lx - lz) / (2 * math.sqrt(2)) + height_m * math.cos(math.radians(30))
    top = (HX + HZ) / math.sqrt(2) * 0.5 + 32
    return u * scale + width / 2, (top - v) * scale
