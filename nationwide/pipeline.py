"""Large-scale rock detection on Swisstopo imagery.

One command: `large-rocks-mapping run --model <path> ( --all | --bbox | --coords | --urls )`.

Architecture:

    tile source (STAC or CSV)
        │
        ▼
    ProcessPoolExecutor[preprocess]  (download + hillshade + VRT + fuse)
        │
        ▼
    tile_queue  (bounded)
        │
        ▼
    main thread: batch → Detector.detect (GPU)
        │
        ▼
    write_queue  (bounded)
        │
        ▼
    writer thread: dedup + DuckDB + checkpoint

The GPU pipeline is synchronous — one batch at a time, no
double-buffering. Workers do all CPU I/O. Writer owns the DuckDB
connection. Everything downstream uses bounded queues for backpressure.
"""

import concurrent.futures
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from nationwide.cache import (
    get_cache_config,
    init_cache,
    set_stac_cache_dir,
)
from nationwide.db import (
    get_processed_tiles,
    init_db,
    mark_tile_done,
    write_detections,
)
from nationwide.detector import Detector
from nationwide.processing import (
    check_gdalbuildvrt,
    check_gdaldem,
    dedup_detections,
    ensure_cached,
    max_elevation,
    process_tile_from_cache,
    reinit_session,
)
from nationwide.spatial import (
    coords_to_wgs84_bbox,
    load_url_csvs,
    query_stac_bbox,
)
from utils.constants import SWITZERLAND_BBOX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile source resolution
# ---------------------------------------------------------------------------


def _resolve_tiles(
    all_switzerland: bool,
    bbox: str | None,
    coords: list[str] | None,
    rgb_urls: Path | None,
    dsm_urls: Path | None,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return (url_map, requested_coords).

    `url_map` covers every tile needed for processing, including
    neighbours. `requested_coords` is the subset the user actually asked
    to process — subsequent code looks up neighbours in the same dict.
    """
    n_modes = sum(bool(x) for x in (all_switzerland, bbox, coords, rgb_urls))
    if n_modes != 1:
        log.error("Pass exactly one of --all / --bbox / --coords / --urls.")
        raise typer.Exit(code=1)

    if all_switzerland:
        pairs = query_stac_bbox(SWITZERLAND_BBOX)
        url_map = {c: (r, d) for c, r, d in pairs}
        requested = sorted(url_map)
    elif bbox:
        pairs = query_stac_bbox(bbox)
        url_map = {c: (r, d) for c, r, d in pairs}
        requested = sorted(url_map)
    elif coords:
        pairs = query_stac_bbox(coords_to_wgs84_bbox(list(coords)))
        url_map = {c: (r, d) for c, r, d in pairs}
        requested = [c for c in coords if c in url_map]
    else:
        if dsm_urls is None:
            log.error("--rgb-urls requires --dsm-urls.")
            raise typer.Exit(code=1)
        pairs = load_url_csvs(rgb_urls, dsm_urls)
        url_map = {c: (r, d) for c, r, d in pairs}
        requested = sorted(url_map)
    return url_map, requested


def _attach_neighbours(
    requested: list[str],
    url_map: dict[str, tuple[str, str]],
) -> list[tuple]:
    """Build per-tile tuples `(coord, rgb, dsm, right, bottom, corner)`."""
    tiles = []
    for c in requested:
        rgb, dsm = url_map[c]
        x, y = c.split("-")
        x, y = int(x), int(y)
        nr = url_map.get(f"{x+1}-{y}")
        nb = url_map.get(f"{x}-{y-1}")
        nc = url_map.get(f"{x+1}-{y-1}")
        tiles.append((c, rgb, dsm, nr, nb, nc))
    return tiles


# ---------------------------------------------------------------------------
# Worker: preprocess one tile
# ---------------------------------------------------------------------------


def _process_one(
    entry: tuple,
    min_elevation: float,
) -> tuple[str, list | str | Exception]:
    """Download + preprocess one tile. Result is a list on success, a
    string (skip reason) if skipped, or an Exception if preprocessing
    failed.
    """
    coord, rgb_url, dsm_url, nb_right, nb_bottom, nb_corner = entry
    try:
        if min_elevation > 0:
            dsm_path = ensure_cached(dsm_url)
            if max_elevation(dsm_path) < min_elevation:
                return (coord, "low_elevation")
        patches = process_tile_from_cache(
            coord, rgb_url, dsm_url,
            neighbor_right=nb_right,
            neighbor_bottom=nb_bottom,
            neighbor_corner=nb_corner,
        )
        return (coord, patches)
    except Exception as exc:
        return (coord, exc)


# ---------------------------------------------------------------------------
# Producer / writer threads
# ---------------------------------------------------------------------------


def _producer(
    tiles: list[tuple],
    tile_queue: queue.Queue,
    workers: int,
    min_elevation: float,
    producer_exc: list[BaseException],
) -> None:
    """Run preprocessing across `workers` processes, streaming results."""
    cc = get_cache_config()
    init_args = (cc[0], cc[1]) if cc else (None, 0)
    pbar = tqdm(total=len(tiles), desc="Preprocess", unit="tile",
                position=1, leave=False)

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=reinit_session,
            initargs=init_args,
        ) as pool:
            pending: dict[concurrent.futures.Future, bool] = {}
            tile_iter = iter(tiles)
            for _ in range(workers * 2):
                entry = next(tile_iter, None)
                if entry is None:
                    break
                pending[pool.submit(_process_one, entry, min_elevation)] = True

            while pending:
                done, _ = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    del pending[future]
                    tile_queue.put(future.result())
                    pbar.update(1)
                    entry = next(tile_iter, None)
                    if entry is not None:
                        pending[pool.submit(_process_one, entry, min_elevation)] = True
    except BaseException as exc:
        producer_exc.append(exc)
        log.error("Producer failed: %s", exc)
    finally:
        pbar.close()
        tile_queue.put(None)


def _writer(
    con,
    write_queue: queue.Queue,
    stats: dict,
    pbar,
    no_dedup: bool,
    producer: threading.Thread,
    producer_exc: list[BaseException],
) -> None:
    """Drain write_queue, dedup, insert into DuckDB, checkpoint."""
    while True:
        try:
            item = write_queue.get(timeout=30)
        except queue.Empty:
            if not producer.is_alive() and producer_exc:
                log.error("Writer giving up: producer died")
                return
            continue

        if item is None:
            return
        tile_id, payload = item
        try:
            if isinstance(payload, str):
                mark_tile_done(con, tile_id, 0, skip_reason=payload)
                stats["tiles_skipped"] += 1
            else:
                dets = payload if no_dedup else dedup_detections(payload, distance_m=7.5)
                n = write_detections(con, dets)
                mark_tile_done(con, tile_id, n)
                stats["total_detections"] += n
                stats["tiles_processed"] += 1
        except Exception as exc:
            log.error("Writer failed for tile %s: %s", tile_id, exc)
            stats["tiles_failed"] += 1
        pbar.update(1)
        pbar.set_postfix(
            dets=stats["total_detections"],
            ok=stats["tiles_processed"],
            skip=stats["tiles_skipped"],
            fail=stats["tiles_failed"],
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _collect_batch(
    tile_queue: queue.Queue,
    max_tiles: int,
    write_queue: queue.Queue,
    producer: threading.Thread,
    producer_exc: list[BaseException],
) -> tuple[list[tuple[str, list]], list[tuple], bool]:
    """Pull up to `max_tiles` of ready tiles. Returns (batch, all_patches, sentinel)."""
    batch: list[tuple[str, list]] = []
    all_patches: list[tuple] = []
    sentinel = False

    while len(batch) < max_tiles:
        try:
            item = tile_queue.get(timeout=30)
        except queue.Empty:
            if not producer.is_alive() and producer_exc:
                return batch, all_patches, True
            continue

        if item is None:
            sentinel = True
            break

        coord, result = item
        tile_id = coord.replace("-", "_")

        if isinstance(result, Exception):
            log.warning("Tile %s failed: %s", coord, result)
            continue
        if isinstance(result, str):
            write_queue.put((tile_id, result))
            continue
        if not result:
            write_queue.put((tile_id, []))
            continue

        batch.append((tile_id, result))
        all_patches.extend(result)

    return batch, all_patches, sentinel


def _run(
    detector: Detector,
    output_db: Path,
    tiles: list[tuple],
    workers: int,
    max_batch_tiles: int,
    no_dedup: bool,
    min_elevation: float,
) -> None:
    """Streaming producer / GPU / writer loop."""
    if not tiles:
        log.info("No tiles to process.")
        return

    con = init_db(output_db)

    tile_queue: queue.Queue = queue.Queue(maxsize=max(workers * 3, 12))
    producer_exc: list[BaseException] = []
    producer = threading.Thread(
        target=_producer,
        args=(tiles, tile_queue, workers, min_elevation, producer_exc),
        daemon=True,
    )
    producer.start()

    stats = {
        "total_detections": 0, "tiles_processed": 0,
        "tiles_skipped": 0, "tiles_failed": 0,
    }
    pbar = tqdm(total=len(tiles), desc="Processing", unit="tile")
    write_queue: queue.Queue = queue.Queue(maxsize=64)
    writer = threading.Thread(
        target=_writer,
        args=(con, write_queue, stats, pbar, no_dedup, producer, producer_exc),
        daemon=True,
    )
    writer.start()

    t0 = time.time()
    while True:
        batch, patches, sentinel = _collect_batch(
            tile_queue, max_batch_tiles, write_queue, producer, producer_exc,
        )
        if patches:
            pbar.write(f"  >> {len(batch)} tiles, {len(patches)} patches")
            detections = detector.detect(patches)
            by_tile: dict[str, list] = defaultdict(list)
            for det in detections:
                by_tile[det.tile_id].append(det)
            for tile_id, _p in batch:
                write_queue.put((tile_id, by_tile.get(tile_id, [])))
        if sentinel:
            break

    write_queue.put(None)
    writer.join()
    pbar.close()
    producer.join()

    elapsed = time.time() - t0
    rate = stats["tiles_processed"] / elapsed if elapsed > 0 else 0

    db_total = con.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    gpkg_path = _export_gpkg(con, output_db)
    con.close()

    log.info("=" * 60)
    log.info("Pipeline complete")
    log.info(
        "  Tiles: %d ok, %d skipped, %d failed",
        stats["tiles_processed"], stats["tiles_skipped"], stats["tiles_failed"],
    )
    log.info("  Detections: %d (DB total: %d)", stats["total_detections"], db_total)
    log.info("  Time: %.0fs (%.2f tiles/s)", elapsed, rate)
    log.info("  Output: %s", output_db)
    if gpkg_path is not None:
        log.info("  GPKG:   %s", gpkg_path)
    log.info("=" * 60)


def _export_gpkg(con, output_db: Path) -> Path | None:
    """Dump detections as a GeoPackage. Returns the path, or None if no rows."""
    rows = con.execute(
        "SELECT tile_id, patch_id, easting, northing, confidence, "
        "bbox_w_m, bbox_h_m, class_id, rgb_source, dsm_source FROM detections",
    ).fetchall()
    if not rows:
        return None

    import geopandas as gpd
    from shapely.geometry import box

    records = []
    for tile_id, patch_id, e, n, conf_v, w_m, h_m, cls, rgb_src, dsm_src in rows:
        hw, hh = w_m / 2, h_m / 2
        records.append({
            "tile_id": tile_id,
            "patch_id": patch_id,
            "confidence": round(conf_v, 4),
            "bbox_w_m": round(w_m, 2),
            "bbox_h_m": round(h_m, 2),
            "class_id": cls,
            "rgb_source": rgb_src,
            "dsm_source": dsm_src,
            "geometry": box(e - hw, n - hh, e + hw, n + hh),
        })
    gdf = gpd.GeoDataFrame(records, crs="EPSG:2056")
    gpkg_path = output_db.with_suffix(".gpkg")
    gdf.to_file(gpkg_path, driver="GPKG", layer="rock_detections")
    return gpkg_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


app = typer.Typer(help="Large-scale rock detection on Swisstopo imagery.", no_args_is_help=True)


@app.command()
def run(
    model: Path = typer.Option(..., help="Path to the model .pt file"),
    output: Path = typer.Option(Path("detections.duckdb"), help="DuckDB output path"),
    all_switzerland: bool = typer.Option(False, "--all", help="Process all of Switzerland"),
    bbox: Optional[str] = typer.Option(None, help="WGS84 bounding box: west,south,east,north"),
    coords: Optional[list[str]] = typer.Option(None, help="Tile coordinates (e.g. 2587-1133)"),
    rgb_urls: Optional[Path] = typer.Option(None, "--rgb-urls", help="CSV of SwissIMAGE URLs"),
    dsm_urls: Optional[Path] = typer.Option(None, "--dsm-urls", help="CSV of swissALTI3D URLs"),
    min_elevation: float = typer.Option(1500.0, help="Skip tiles below this elevation in metres (0 disables)"),
    device: str = typer.Option("auto", help="Device: auto, cuda:0, cpu"),
    workers: int = typer.Option(0, help="Preprocess workers (0 = min(12, cpu_count))"),
    conf: float = typer.Option(0.10, help="Confidence threshold"),
    iou: float = typer.Option(0.70, help="IoU threshold"),
    cache_dir: Path = typer.Option(Path("data/tile_cache"), help="Disk cache directory"),
    cache_gb: float = typer.Option(500.0, help="Max download cache size in GB (0 to disable)"),
    max_batch_tiles: int = typer.Option(16, help="Max tiles per GPU batch (25 patches each)"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable spatial deduplication"),
) -> None:
    """Run the rock detection pipeline."""
    check_gdaldem()
    check_gdalbuildvrt()
    init_cache(cache_dir, cache_gb)
    set_stac_cache_dir(cache_dir)

    # --- Model up front: fail fast before network work ---
    detector = Detector(model, device=device, conf=conf, iou=iou)
    detector.warmup()

    # --- Resolve tile source (includes neighbours) ---
    url_map, requested = _resolve_tiles(all_switzerland, bbox, coords, rgb_urls, dsm_urls)
    log.info("Tile pairs: %d in URL map, %d requested", len(url_map), len(requested))

    # --- Filter already-processed tiles via checkpoint ---
    con = init_db(output)
    done_coords = {t.replace("_", "-") for t in get_processed_tiles(con)}
    con.close()
    remaining = [c for c in requested if c not in done_coords]
    if len(remaining) < len(requested):
        log.info("Skipping %d already-processed tiles", len(requested) - len(remaining))
    if not remaining:
        log.info("All tiles already processed.")
        return

    tiles_with_neighbours = _attach_neighbours(remaining, url_map)

    n_workers = workers if workers > 0 else min(12, os.cpu_count() or 8)
    log.info(
        "Workers: %d, max batch: %d tiles (%d patches)",
        n_workers, max_batch_tiles, max_batch_tiles * 25,
    )

    _run(
        detector, output, tiles_with_neighbours,
        workers=n_workers, max_batch_tiles=max_batch_tiles,
        no_dedup=no_dedup, min_elevation=min_elevation,
    )
