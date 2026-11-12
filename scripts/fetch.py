# /// script
# requires-python = ">=3.11"
# dependencies = ["pyproj", "numpy", "rasterio"]
# ///
"""Download the raw open data for the area: Environment Agency lidar, OpenStreetMap and a Sentinel-2 image."""

import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from config import CHUNK, NX, NY, RAW, XMAX, XMIN, YMAX, YMIN

UA = {"User-Agent": "isometric-village-map/0.1"}

# Environment Agency LIDAR Composite 2022 at 1 m (Open Government Licence): 5 km tiles from the Defra survey downloads.
EA_TILES = "https://environment.data.gov.uk/tiles/collections/survey/{product}/2022/1/{tile}"
EA_PRODUCTS = {"dtm": "lidar_composite_dtm", "dsm": "lidar_composite_first_return_dsm"}

# A cloud-free Sentinel-2 L2A scene (true colour, 10 m), from the open Sentinel-2 COGs on AWS (Element84 Earth Search).
SENTINEL_TCI = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/30/U/XC/2026/5/S2C_30UXC_20260525_0_L2A/TCI.tif"


def get(url, binary=False, tries=8):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
                return data if binary else json.loads(data)
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            print(f"retry {attempt + 1} {url[:100]}: {e}", file=sys.stderr)
            # The Environment Agency answers 429 when it is busy; give it room.
            time.sleep((20 if "429" in str(e) else 2) * (attempt + 1))


def os_tile(x, y):
    """The Ordnance Survey 5 km tile around a grid point, as the Defra downloads name it (TL0040 for TL04SW)."""
    letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # no I
    # 500 km squares: S at (0, 0), T east of it, N north of it, then O, H and J.
    big = {(0, 0): "S", (1, 0): "T", (0, 1): "N", (1, 1): "O", (0, 2): "H", (1, 2): "J"}[(int(x // 500000), int(y // 500000))]
    second = letters[(4 - int(y % 500000 // 100000)) * 5 + int(x % 500000 // 100000)]
    return f"{big}{second}{int(x % 100000 // 10000)}{int(x % 10000 // 5000) * 5}{int(y % 100000 // 10000)}{int(y % 10000 // 5000) * 5}"


def fetch_lidar():
    """Every 5 km lidar tile that touches the area, cut into CHUNK tiles under data/raw/lidar/."""
    d = RAW / "lidar"
    d.mkdir(exist_ok=True)
    want = [(k, ix, iy) for k in EA_PRODUCTS for ix in range(NX) for iy in range(NY) if not (d / f"{k}_{ix}_{iy}.tif").exists()]
    if not want:
        return
    import zipfile

    import rasterio
    from rasterio.windows import from_bounds

    tiles = {os_tile(x, y) for x in range(XMIN, XMAX, 500) for y in range(YMIN, YMAX, 500)}
    zips = RAW / "lidar_zip"
    zips.mkdir(exist_ok=True)
    for kind, product in EA_PRODUCTS.items():
        for tile in sorted(tiles):
            z = zips / f"{kind}_{tile}.zip"
            if not z.exists():
                print(f"lidar {kind} {tile}", file=sys.stderr)
                z.with_suffix(".tmp").write_bytes(get(EA_TILES.format(product=product, tile=tile), binary=True))
                z.with_suffix(".tmp").rename(z)
            with zipfile.ZipFile(z) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".tif"))
                with rasterio.MemoryFile(zf.read(name)) as mem, mem.open() as src:
                    b = src.bounds
                    for k, ix, iy in want:
                        x0, y0 = XMIN + ix * CHUNK, YMIN + iy * CHUNK
                        if k != kind or not (b.left <= x0 < b.right and b.bottom <= y0 < b.top):
                            continue
                        win = from_bounds(x0, y0, x0 + CHUNK, y0 + CHUNK, src.transform)
                        a = src.read(1, window=win, boundless=True, fill_value=src.nodata if src.nodata is not None else -9999)
                        prof = {"driver": "GTiff", "width": a.shape[1], "height": a.shape[0], "count": 1, "dtype": a.dtype,
                                "crs": src.crs, "transform": src.window_transform(win), "nodata": src.nodata}
                        with rasterio.open(d / f"{kind}_{ix}_{iy}.tif", "w", **prof) as dst:
                            dst.write(a, 1)


def fetch_sentinel():
    """The Sentinel-2 true-colour scene, warped onto the grid at GROUND_RES, in CHUNK tiles like an aerial photo."""
    d = RAW / "aerial"
    if d.exists() and len(list(d.glob("*.png"))) == NX * NY:
        return
    import subprocess

    subprocess.run(["uv", "run", "--quiet", "--with", "rasterio", "--with", "pillow", "python", "sentinel.py", SENTINEL_TCI], check=True)


def overpass(out, q, what):
    """Run an Overpass query into out (Overpass is often busy, so try a few servers)."""
    for host in ("overpass-api.de/api/interpreter", "overpass.private.coffee/api/interpreter", "maps.mail.ru/osm/tools/overpass/api/interpreter"):
        try:
            req = urllib.request.Request(f"https://{host}", data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
            with urllib.request.urlopen(req, timeout=180) as r:
                out.write_bytes(r.read())
            return
        except Exception as e:  # noqa: BLE001
            print(f"overpass {host}: {e}", file=sys.stderr)
    print(f"OpenStreetMap unavailable; {what} are skipped", file=sys.stderr)


def osm_bbox(margin=0):
    from geo import grid_to_wgs

    lats, lons = zip(*(grid_to_wgs(x, y) for x in (XMIN - margin, XMAX + margin) for y in (YMIN - margin, YMAX + margin)))
    return f"{min(lats)},{min(lons)},{max(lats)},{max(lons)}"


def fetch_osm_all():
    """Every OpenStreetMap feature in the area (with a margin): buildings, land use, water, roads, hedges, trees and
    addresses. scripts/osm.py turns it into the footprints and the ground surfaces."""
    out = RAW / "osm_all.json"
    if out.exists():
        return
    bbox = osm_bbox(100)
    q = f"""[out:json][timeout:300][maxsize:1073741824];
( way({bbox}); relation["type"="multipolygon"]({bbox}); node({bbox})(if:count_tags() > 0); );
out tags geom;"""
    overpass(out, q, "buildings and ground")
    if not out.exists():
        sys.exit("OpenStreetMap is needed for the buildings and the ground; try again later")


def fetch_osm():
    """Pitches, graveyards and points of interest from OpenStreetMap."""
    out = RAW / "osm_landmarks.json"
    if out.exists():
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:90];
( nwr["military"]({bbox}); way["leisure"="pitch"]({bbox}); nwr["landuse"="cemetery"]({bbox});
  nwr["amenity"="place_of_worship"]({bbox}); nwr["tourism"]({bbox}); nwr["historic"]({bbox}); );
out geom tags;"""
    overpass(out, q, "pitches and graveyards")


def fetch_routes():
    """Walking routes and the numbered walking-junction network (wandelknooppunten) from OpenStreetMap."""
    out = RAW / "osm_routes.json"
    if out.exists():
        return
    bbox = osm_bbox()
    q = f"""[out:json][timeout:120];
( relation["route"~"^(hiking|foot|walking)$"]({bbox}); node["rwn_ref"]({bbox}); );
out geom;"""
    overpass(out, q, "walking routes")


def fetch_roads():
    """The roads, cycle paths, footpaths and farm tracks that people use, from OpenStreetMap, with node ids to find
    junctions."""
    out = RAW / "osm_roads.json"
    if out.exists():
        return
    bbox = osm_bbox()
    kinds = "motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|service"
    q = f"""[out:json][timeout:120];
( way["highway"~"^({kinds})$"]({bbox}); way["highway"~"^(cycleway|footway|path|pedestrian|track|steps|bridleway)$"]({bbox}); );
out geom;"""
    overpass(out, q, "roads and paths for traffic")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [fetch_osm_all, fetch_osm, fetch_routes, fetch_roads, fetch_sentinel]
    # Overpass allows a couple of queries at a time per client.
    with ThreadPoolExecutor(2) as pool:
        for fut in [pool.submit(j) for j in [*jobs, fetch_lidar]]:
            fut.result()
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
