"""The viewer's search list: named places first, then the streets, named buildings and addresses that
OpenStreetMap has inside the map (scripts/osm.py). Britain has no open address register with house numbers."""

import json

from config import BUILD, XMAX, XMIN, YMAX, YMIN


def main():
    lm = json.loads((BUILD / "landmarks.json").read_text())
    places = [{"name": p["name"], "note": p.get("note"), "x": round(p["x"], 1), "y": round(p["y"], 1), "h": p["h"]} for p in lm["places"]]
    seen = {p["name"] for p in places}
    for x, y, label in json.loads((BUILD / "osm_names.json").read_text()):
        if XMIN <= x <= XMAX and YMIN <= y <= YMAX and label not in seen:
            seen.add(label)
            places.append({"name": label, "x": round(x, 1), "y": round(y, 1), "h": 4})
    (BUILD / "places.json").write_text(json.dumps(places, ensure_ascii=False))
    print(f"{len(places)} places")


if __name__ == "__main__":
    main()
