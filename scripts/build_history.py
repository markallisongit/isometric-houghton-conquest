"""Houghton Conquest through time: a guided tour along the village's history, as an overlay for the viewer.

Texts are written for this map from Wikipedia and the National Heritage List for England; every stop names its
source. Photos (none yet) are Wikimedia Commons files with an open license, downloaded once and published with the
site, with credits.

Output: data/build/history.json and data/build/history/*.jpg
"""

import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request

from config import BUILD, CX, CY
from geo import wgs_to_grid

NHLE = "https://historicengland.org.uk/listing/the-list/list-entry/"
WIKI = "https://en.wikipedia.org/wiki/"
VILLAGE = ("Wikipedia: Houghton Conquest", WIKI + "Houghton_Conquest")
CHURCH = ("Wikipedia: Church of All Saints", WIKI + "Church_of_All_Saints,_Houghton_Conquest")
HOUSE = ("Wikipedia: Houghton House", WIKI + "Houghton_House")

STOPS = [
    {
        "year": "1086",
        "title": "Houghton in Domesday Book",
        "at": (504555, 241553),
        "zoom": 0.25,
        "text": "Domesday Book lists Houghton in the hundred of Redbornestoke. Adeliza de Grandmesnil held it from the king, and a man called Arnold held the land from her.",
        "source": VILLAGE,
    },
    {
        "year": "13th century",
        "title": "The Conquest family",
        "at": (505165, 240606),
        "text": "From the 13th to the 18th century the Conquest family held a manor here, and the village took their name: Houghton Conquest. Their house, Conquestbury, stood where Bury Farm is now, beside London Lane. When the family left, it was let fall into ruin.",
        "source": VILLAGE,
    },
    {
        "year": "14th century",
        "title": "The Church of All Saints",
        "at": (504360, 241429),
        "h": 20,
        "text": "The parish church is built in the 14th century, with a nave, aisles, a chancel and a battlemented tower with six bells. Over the north door a large St Christopher is painted in the 15th century; traces of other wall paintings can still be seen.",
        "source": CHURCH,
        "more": NHLE + "1113919",
    },
    {
        "year": "1615",
        "title": "Houghton House",
        "at": (503922, 239484),
        "h": 11,
        "text": "Mary Sidney, Countess of Pembroke, poet and patron, starts a great house on the ridge, possibly with Inigo Jones. James I visits in 1621, soon after it is finished; the Countess dies of smallpox that September.",
        "source": HOUSE,
        "more": NHLE + "1321465",
    },
    {
        "year": "1678",
        "title": "House Beautiful",
        "at": (503922, 239484),
        "h": 11,
        "text": "John Bunyan publishes The Pilgrim's Progress. Houghton House, on the ridge south of the village, is said to be his House Beautiful, where Christian is welcomed on his way.",
        "source": HOUSE,
    },
    {
        "year": "1794",
        "title": "The roof comes off",
        "at": (503922, 239484),
        "h": 11,
        "text": "The 5th Duke of Bedford has Houghton House stripped: the furnishings go and the roof is sold off as building material. The staircase ended up in the Swan Hotel in Bedford. The walls still stand.",
        "source": HOUSE,
        "more": "https://www.english-heritage.org.uk/visit/places/houghton-house/",
    },
    {
        "year": "1850s",
        "title": "The village shop",
        "at": (504571.9, 241600.8),
        "text": "A house is built in the village partly from an old manor house, most likely Conquestbury. Today it is the village shop and post office.",
        "source": VILLAGE,
    },
    {
        "year": "1880",
        "title": "The east window",
        "at": (504360, 241429),
        "h": 12,
        "text": "John William Burgon, Dean of Chichester, gives All Saints its stained-glass east window, in memory of three benefactors.",
        "source": CHURCH,
    },
    {
        "year": "1984",
        "title": "King's Wood and Glebe Meadows",
        "at": (504531, 239447),
        "text": "The ash and maple wood and the old meadows south of the village become a Site of Special Scientific Interest. Their rare plants had been found by a local teenager, Peter Sollars.",
        "source": ("Wikipedia: Kings Wood and Glebe Meadows", WIKI + "Kings_Wood_and_Glebe_Meadows,_Houghton_Conquest"),
    },
]


def commons_thumb(name, width=960):  # Wikimedia only serves a fixed set of thumbnail widths
    h = hashlib.md5(name.encode()).hexdigest()
    q = urllib.parse.quote(name)
    return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{h[0]}/{h[:2]}/{q}/{width}px-{q}"


def download(url, tries=6):
    """Wikimedia rate-limits bursts of thumbnail requests; wait politely and retry."""
    req = urllib.request.Request(url, headers={"User-Agent": "isometric-village-map/0.1"})
    for attempt in range(tries):
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            time.sleep(3)
            return data
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == tries - 1:
                raise
            time.sleep(15 * (attempt + 1))


def route_order(stops):
    """Visit the stops as a route over the map instead of by date: the first and last stay put, the rest is a
    nearest-neighbour path tidied with 2-opt, so dots that lie together come one after another. Stops on the same
    spot (the old and the new church) keep their chronological order."""
    first, last, middle = stops[0], stops[-1], stops[1:-1]
    dist = lambda a, b: ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
    path, todo = [first], middle[:]
    while todo:
        nxt = min(todo, key=lambda s: (dist(path[-1], s), stops.index(s)))
        path.append(nxt)
        todo.remove(nxt)
    path.append(last)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path) - 1):
                before = dist(path[i - 1], path[i]) + dist(path[j], path[j + 1])
                after = dist(path[i - 1], path[j]) + dist(path[i], path[j + 1])
                if after < before - 0.01:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    improved = True
    # Places within 40 m of each other in chronological order.
    i = 0
    while i < len(path):
        j = i
        while j + 1 < len(path) - 1 and i > 0 and dist(path[j + 1], path[i]) < 40:
            j += 1
        year = lambda st: int(m.group()) if (m := re.search(r"\d{4}", st["year"])) else 9999
        path[i : j + 1] = sorted(path[i : j + 1], key=lambda st: (year(st), stops.index(st)))
        i = j + 1
    return path


def main():
    out_dir = BUILD / "history"
    out_dir.mkdir(parents=True, exist_ok=True)
    stops = []
    for i, s in enumerate(STOPS, start=1):
        x, y = s["at"] if "at" in s else wgs_to_grid(*s["latlon"])
        stop = {"n": i, "year": s["year"], "title": s["title"], "text": s["text"], "x": round(x, 1), "y": round(y, 1),
                "h": s.get("h", 0), "zoom": s.get("zoom", 1.2),
                "source": {"label": s["source"][0], "url": s["source"][1]}, "more": s.get("more")}
        stop["photos"] = []
        for k, (name, credit) in enumerate(s.get("photos", [])):
            dst = out_dir / f"{i:02d}_{k}.jpg"
            if not dst.exists():
                dst.write_bytes(download(commons_thumb(name)))
                subprocess.run(["magick", str(dst), "-resize", "640x640>", "-quality", "82", str(dst)], check=True)
            stop["photos"].append({"src": f"history/{dst.name}", "credit": credit,
                                   "url": f"https://commons.wikimedia.org/wiki/File:{urllib.parse.quote(name)}"})
        stops.append(stop)
    stops = route_order(stops)
    for n, stop in enumerate(stops, start=1):
        stop["n"] = n
    # A version in every photo URL, so browsers never show a cached photo that has since been replaced.
    version = hashlib.md5(b"".join(f.read_bytes() for f in sorted(out_dir.glob("*.jpg")))).hexdigest()[:8]
    for stop in stops:
        for ph in stop["photos"]:
            ph["src"] += f"?v={version}"
    (BUILD / "history.json").write_text(json.dumps({"stops": stops}, ensure_ascii=False, indent=1))
    print(f"{len(stops)} stops, {sum(len(s['photos']) for s in stops)} photos on {sum(bool(s['photos']) for s in stops)} stops")


if __name__ == "__main__":
    main()
