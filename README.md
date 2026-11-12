# Isometric Houghton Conquest

A zoomable isometric pixel-art map of Houghton Conquest in Bedfordshire, 3 × 4 km from the Midland Main Line to the ruins of Houghton House on the ridge to the south. It is built on [Isometric Helenaveen](https://github.com/jankeesvw/isometric-helenaveen) by [@jankeesvw](https://x.com/jankeesvw), who rendered a Dutch village procedurally from open data. This fork swaps the Dutch sources for British ones; the renderer and the viewer are theirs.

## Pipeline

Run `./build.sh` for everything. The steps:

1. `scripts/fetch.py` downloads the raw data into `data/raw/`:
   - the Environment Agency LIDAR Composite 2022 at 1 m (first-return surface model and terrain model), as 5 km tiles from the Defra survey downloads, cut into 500 m chunks;
   - everything OpenStreetMap has in the area (buildings, land use, roads, hedges, water, single trees, names), plus the pitches, graveyards, walking routes and road network the overlays use;
   - one cloud-free Sentinel-2 true-colour scene (25 May 2026, 10 m) from the open COGs on AWS. `scripts/sentinel.py` warps it onto the grid and brightens it.

   England has no open aerial photo at this resolution: the Environment Agency's vertical photography does not cover the area, and APGB is for the public sector only. The satellite image only colours fields and gardens, and has no roof detail.
2. `scripts/osm.py` turns OpenStreetMap into footprints and ground surfaces in British National Grid (EPSG:27700) metres. Roads, pavements, paths, streams, hedges and railways are buffered from lines into polygons by a width per kind.
3. `scripts/landmarks.py` names the places people look for: All Saints, Houghton House, the listed farmhouses, the shop, the pubs, King's Wood. It gives the church its tower, a block about 7 m square at the west end, flat at 20 m, measured in the lidar. Houghton House is a roofless shell, its walls 11 m high.
4. `scripts/build_buildings.py` fits a roof to every OpenStreetMap footprint, using the lidar surface minus the terrain inside it. The candidates are flat, a gable along either axis of the footprint's minimum rotated rectangle, and hipped. Outliers (trees over a roof, chimneys) are dropped and the model is refitted. A pitched roof has to beat the flat one clearly. The roof is the lower envelope of its planes over the footprint, and the walls run up to it, so gable ends come out as pentagons. Roof colours are seeded from a palette of clay tile, slate and concrete tile for houses, and sheet metal for farm buildings.
5. `scripts/build_trees.py` finds trees in the canopy height model (lidar surface minus terrain): local maxima for the tops, watershed segmentation for crown size. OpenStreetMap trees fill in what the lidar missed. Woods that OpenStreetMap maps but the 2022 lidar shows no canopy in get saplings: these are young plantings of the Forest of Marston Vale.
6. `scripts/build_ground.py` paints the surfaces in a fixed palette with patterns and bakes hard sun shadows into them. Land that OpenStreetMap leaves blank, which is most of the farmland, is cut into fields along the colours of the satellite image (k-means, a mode filter, then connected pieces between hedges and roads), each with its own crop colour and furrows.
7. `render/scene.js` and `render/run.mjs` render it in headless Chromium, tile by tile, from four directions. `vips dzsave` cuts Deep Zoom tiles for `viewer/index.html`, unchanged from the original apart from:
   - English only;
   - British National Grid ↔ GPS from cubic fits written into `map.json` by `scripts/geo.py`;
   - traffic on the left (`LEFT_HAND` in `scripts/build_traffic.py`);
   - the tour "Through time" (`scripts/build_history.py`, `#tour` opens it);
   - the walking routes: the John Bunyan Trail, the Greensand Ridge Walk and the Marston Vale Timberland Trail.

`./site.sh` cuts the rendered images into tiles and assembles `site/` without rendering anything. `./deploy.sh` publishes `site/` to the `gh-pages` branch of the `origin` remote. The tools: Python with [uv](https://docs.astral.sh/uv/), Node with npm or bun, and libvips.

Overlays (search, hover outlines, walking routes, the tour, traffic) are JSON files next to the tiles. Changing them needs no rendering, only the small build script and `./deploy.sh`.

## Iterating

`node render/run.mjs crop <x> <y> <w> <h> --scale 4 --out out/crop.png` renders one region of the full image. `node render/run.mjs crop 0 0 3000 1700 --scale 0.6` gives an overview of the whole map.

## Data

- Lidar: © Environment Agency copyright and/or database right 2022, Open Government Licence v3.0.
- Map data: © OpenStreetMap contributors, ODbL. The rendered map is a Produced Work; the JSON overlays are derived from OpenStreetMap and fall under the ODbL.
- Satellite image: contains modified Copernicus Sentinel data 2026.
- Listed buildings: National Heritage List for England, © Historic England, Open Government Licence.
- Place texts are written for this map from the sources above and from Wikipedia. Each tour stop names its source.
- Music: [My Street](https://opengameart.org/content/my-street) by congusbongus (CC0).
