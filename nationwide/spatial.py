"""Swisstopo STAC queries and tile-coordinate helpers.

Replaces the old per-coord HEAD scan (`_find_latest_url`) with a single
STAC query per run. STAC returns the coord → URL mapping for every tile
in a bbox in one shot, so neighbours come for free — no separate
neighbour-URL cache.
"""

import logging
import re
from pathlib import Path
from typing import Iterator

import requests

from nationwide.cache import load_stac_cache, save_stac_cache
from utils.constants import (
    COORD_RE,
    DSM_COLLECTION,
    SI_COLLECTION,
    STAC_BASE,
)

log = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})

_YEAR_RE = re.compile(r"_(\d{4})_")


def _stac_paginate(
    collection: str, bbox: str, limit: int = 100,
) -> Iterator[dict]:
    """Yield all STAC items for `collection` within `bbox` (WGS84)."""
    url: str | None = f"{STAC_BASE}/collections/{collection}/items"
    params: dict = {"bbox": bbox, "limit": limit}
    page = 0
    total_items = 0
    while url:
        page += 1
        resp = _SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        features = data.get("features", [])
        total_items += len(features)
        if page % 10 == 0 or page == 1:
            log.info("  STAC page %d: %d items so far ...", page, total_items)
        yield from features
        url = None
        params = {}
        for link in data.get("links", []):
            if link.get("rel") == "next":
                url = link["href"]
                break


def _extract_stac_tiles(
    items: Iterator[dict], target_year: int = 0,
) -> dict[str, str]:
    """Build coord → asset-URL dict, keeping newest year per coord unless `target_year` set."""
    result: dict[str, str] = {}
    result_year: dict[str, int] = {}
    for item in items:
        item_id = item.get("id", "")
        m = COORD_RE.search(item_id)
        if not m:
            continue
        coord = m.group(1)
        ym = _YEAR_RE.search(item_id)
        year = int(ym.group(1)) if ym else 0
        if target_year and year != target_year:
            continue
        if coord in result and year <= result_year.get(coord, 0):
            continue
        for asset in item.get("assets", {}).values():
            href = asset.get("href", "")
            if href.endswith(".tif"):
                result[coord] = href
                result_year[coord] = year
                break
    return result


def query_stac_bbox(bbox: str) -> list[tuple[str, str, str]]:
    """Return `(coord, rgb_url, dsm_url)` tuples for every matched pair in `bbox`.

    Results are cached in the STAC DuckDB table keyed by bbox string.
    """
    cached = load_stac_cache(bbox)
    if cached is not None:
        log.info(f"STAC cache hit: {len(cached)} tile pairs for bbox={bbox}")
        return cached

    log.info(f"Querying STAC for SwissIMAGE tiles in bbox={bbox} ...")
    rgb_tiles = _extract_stac_tiles(_stac_paginate(SI_COLLECTION, bbox))
    log.info(f"  Found {len(rgb_tiles)} SwissIMAGE tiles")

    log.info(f"Querying STAC for swissALTI3D tiles in bbox={bbox} ...")
    dsm_tiles = _extract_stac_tiles(_stac_paginate(DSM_COLLECTION, bbox))
    log.info(f"  Found {len(dsm_tiles)} swissALTI3D tiles")

    common = sorted(set(rgb_tiles) & set(dsm_tiles))
    log.info(f"  Matched pairs: {len(common)}")

    result = [(c, rgb_tiles[c], dsm_tiles[c]) for c in common]
    save_stac_cache(bbox, result)
    return result


def coords_to_wgs84_bbox(coords: list[str], pad_tiles: int = 1) -> str:
    """WGS84 bbox covering the given tile coords ± `pad_tiles` tile widths.

    Used to pick up neighbour tiles in one STAC query when the user
    passes `--coords` (rather than doing a HEAD scan per tile). We
    transform the four LV95 corners and take the min/max so projection
    distortion at the edges doesn't cut tiles off.
    """
    from pyproj import Transformer

    xs, ys = [], []
    for c in coords:
        x_km, y_km = c.split("-")
        xs.append(int(x_km))
        ys.append(int(y_km))

    min_e = (min(xs) - pad_tiles) * 1000
    max_e = (max(xs) + 1 + pad_tiles) * 1000
    min_n = (min(ys) - pad_tiles) * 1000
    max_n = (max(ys) + 1 + pad_tiles) * 1000

    tf = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for e, n in [(min_e, min_n), (min_e, max_n), (max_e, min_n), (max_e, max_n)]:
        lon, lat = tf.transform(e, n)
        lons.append(lon)
        lats.append(lat)

    return f"{min(lons):.4f},{min(lats):.4f},{max(lons):.4f},{max(lats):.4f}"


def load_url_csvs(
    rgb_csv: Path, dsm_csv: Path,
) -> list[tuple[str, str, str]]:
    """Inner-join two URL CSVs on coord, returning `(coord, rgb_url, dsm_url)`."""
    def _parse(path: Path) -> dict[str, str]:
        urls: dict[str, str] = {}
        for line in path.read_text().strip().splitlines():
            url = line.strip()
            if not url:
                continue
            m = COORD_RE.search(url)
            if m:
                urls[m.group(1)] = url
        return urls

    rgb_urls = _parse(rgb_csv)
    dsm_urls = _parse(dsm_csv)
    common = sorted(set(rgb_urls) & set(dsm_urls))
    log.info(
        "Loaded %d RGB + %d DSM URLs, %d matched pairs",
        len(rgb_urls), len(dsm_urls), len(common),
    )
    return [(c, rgb_urls[c], dsm_urls[c]) for c in common]
