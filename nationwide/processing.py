"""Raster I/O and preprocessing for the nationwide pipeline.

All model-specific code (torch / ultralytics) lives in `detector.py`.
This module is pure GDAL / rasterio / numpy: download, cache, hillshade,
VRT mosaic, crop, resample, fuse. Given a tile's URLs and cached paths,
it returns a list of fused (3, 640, 640) uint8 patches ready to hand to
a detector.

The 25-patch invariant (5×5 per tile, independent of neighbour
availability) and RGB cubic-resampling parity are guarded by
`tests/test_patch_generation.py` — preserve the fixed-extent VRT and the
`_materialize_uncompressed` intermediate.
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import rasterio
import rasterio.io
import requests
from rasterio.enums import Resampling
from rasterio.windows import Window
from requests.adapters import HTTPAdapter

from nationwide.cache import cache_get, cache_path, cache_put, reinit_cache
from nationwide.db import Detection
from utils.constants import (
    DSM_RES,
    FUSION_CHANNEL,
    NEIGHBOR_STRIP_DSM,
    NEIGHBOR_STRIP_RGB,
    SRC_CROP_DSM,
    SRC_CROP_RGB,
    SRC_STRIDE_DSM,
    SRC_STRIDE_RGB,
    TILE_PX_DSM,
    TILE_PX_RGB,
    TILE_SIZE_PX,
)

log = logging.getLogger(__name__)

_POOL_SIZE = os.cpu_count() or 8
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
_SESSION.mount(
    "https://",
    HTTPAdapter(pool_maxsize=_POOL_SIZE, pool_connections=_POOL_SIZE),
)


# ---------------------------------------------------------------------------
# GDAL CLI sanity checks
# ---------------------------------------------------------------------------


def check_gdaldem() -> None:
    if shutil.which("gdaldem") is None:
        raise RuntimeError("gdaldem not found on PATH — install GDAL CLI tools")


def check_gdalbuildvrt() -> None:
    if shutil.which("gdalbuildvrt") is None:
        raise RuntimeError("gdalbuildvrt not found on PATH — install GDAL CLI tools")


def _run(cmd: list[str]) -> None:
    """subprocess.run with check=True and stderr surfaced on failure."""
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RuntimeError(f"{cmd[0]} failed: {stderr.strip() or exc}") from exc


# ---------------------------------------------------------------------------
# Download + cache glue
# ---------------------------------------------------------------------------


def reinit_session(cache_dir: str | None = None, cache_max_bytes: int = 0) -> None:
    """Re-create HTTP session and tile cache after a fork.

    `ProcessPoolExecutor` workers get a fresh module import, so the
    shared singletons need to be rebuilt in each worker.
    """
    global _SESSION
    _SESSION = requests.Session()
    _SESSION.headers.update({"User-Agent": "rock-detection-pipeline/0.1"})
    _SESSION.mount(
        "https://",
        HTTPAdapter(pool_maxsize=_POOL_SIZE, pool_connections=_POOL_SIZE),
    )
    reinit_cache(cache_dir, cache_max_bytes)


def download_to_memory(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    """Fetch a URL into bytes, using the on-disk cache when available."""
    cached = cache_get(url)
    if cached is not None:
        return cached

    for attempt in range(1, retries + 1):
        try:
            resp = _SESSION.get(url, timeout=timeout)
            resp.raise_for_status()
            if len(resp.content) == 0:
                raise IOError("Empty response")
            cache_put(url, resp.content)
            return resp.content
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise RuntimeError("Unreachable")


def ensure_cached(url: str, retries: int = 3, timeout: int = 120) -> Path:
    """Ensure URL is on disk in the tile cache and return its path.

    Under cache pressure eviction in another worker can remove a file we
    just wrote, so on the first miss we retry the download once before
    giving up.
    """
    p = cache_path(url)
    if p is not None:
        return p
    for attempt in (1, 2):
        data = download_to_memory(url, retries=retries, timeout=timeout)
        del data  # bytes are now on disk
        p = cache_path(url)
        if p is not None:
            return p
        log.warning(
            "Cache miss immediately after download (%s) attempt %d — retrying",
            url, attempt,
        )
    raise RuntimeError(f"Cache miss immediately after download: {url}")


# ---------------------------------------------------------------------------
# Hillshade + VRT + uncompressed materialisation
# ---------------------------------------------------------------------------


def generate_hillshade(dsm_path: str | Path) -> np.ndarray:
    """Compute combined hillshade via `gdaldem hillshade -combined -az 315 -alt 45 -compute_edges`.

    Matches the training data preprocessing; do not change the flags
    without retraining.
    """
    fd, out_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        _run([
            "gdaldem", "hillshade", str(dsm_path), out_path,
            "-combined", "-az", "315", "-alt", "45",
            "-compute_edges", "-of", "GTiff", "-q",
        ])
        with rasterio.open(out_path) as src:
            return src.read(1)
    finally:
        Path(out_path).unlink(missing_ok=True)


def _build_vrt(
    paths: list[str | Path],
    te: tuple[float, float, float, float] | None = None,
) -> str:
    """Build a GDAL VRT mosaicking `paths`, optionally padded to a fixed bbox.

    With `te=(xmin, ymin, xmax, ymax)` the VRT's extent is forced to that
    bbox in the source CRS; regions without source coverage are filled
    with the VRT's nodata (zero by default). This is how we keep the
    patch grid invariant under neighbour availability — missing neighbours
    become zero-padding rather than shrunken iteration dimensions.
    """
    fd, vrt_path = tempfile.mkstemp(suffix=".vrt")
    os.close(fd)
    cmd = ["gdalbuildvrt", "-q"]
    if te is not None:
        cmd += ["-te", *(f"{v:.6f}" for v in te)]
    cmd += [vrt_path] + [str(p) for p in paths]
    _run(cmd)
    return vrt_path


def _materialize_uncompressed(vrt_or_tif: str) -> str:
    """Write a VRT or TIFF out as an uncompressed GeoTIFF.

    GDAL's cubic resampling kernel produces slightly different output on
    JPEG-compressed sources vs uncompressed (max |Δ| ≈ 47 on uint8). For
    parity with the training data preprocessor we materialise an
    uncompressed intermediate before the cubic windowed read. This is
    load-bearing — see `tests/test_patch_generation.py::test_rgb_cubic_resampling_parity_with_uncompressed`.
    """
    fd, out_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    _run([
        "gdal_translate", "-q",
        "-of", "GTiff", "-co", "COMPRESS=NONE",
        str(vrt_or_tif), out_path,
    ])
    return out_path


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def crop_patches(
    data: np.ndarray,
    transform: rasterio.Affine,
    crop_px: int,
    stride_px: int,
    out_px: int,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int]]:
    """Extract overlapping patches from an in-memory raster array."""
    is_2d = data.ndim == 2
    h, w = data.shape[-2:]

    patches: list[tuple[np.ndarray, rasterio.Affine, int, int]] = []
    row_idx = 0
    y_off = 0
    while y_off + crop_px <= h:
        col_idx = 0
        x_off = 0
        while x_off + crop_px <= w:
            if is_2d:
                window = data[y_off:y_off + crop_px, x_off:x_off + crop_px]
            else:
                window = data[:, y_off:y_off + crop_px, x_off:x_off + crop_px]

            if crop_px != out_px:
                if is_2d:
                    patch = cv2.resize(
                        window, (out_px, out_px), interpolation=cv2.INTER_CUBIC,
                    )
                else:
                    bands = [
                        cv2.resize(window[b], (out_px, out_px), interpolation=cv2.INTER_CUBIC)
                        for b in range(window.shape[0])
                    ]
                    patch = np.stack(bands)
            else:
                patch = window.copy()

            x_map = transform.c + x_off * transform.a
            y_map = transform.f + y_off * transform.e
            patch_res_x = (crop_px * transform.a) / out_px
            patch_res_y = (crop_px * transform.e) / out_px
            patch_transform = rasterio.Affine(
                patch_res_x, 0, x_map, 0, patch_res_y, y_map,
            )

            patches.append((patch, patch_transform, row_idx, col_idx))
            col_idx += 1
            x_off += stride_px
        row_idx += 1
        y_off += stride_px

    return patches


def _crop_resample_rgb(
    rgb_path: str, width: int, height: int,
) -> list[tuple[np.ndarray, rasterio.Affine, int, int]]:
    """Read 3200×3200 windows from the source RGB and cubic-resample each to 640×640.

    Pixel-identical to `gdal_translate -srcwin -outsize -r cubic` when the
    source is uncompressed (validated across 22 224 patches).
    """
    patches: list[tuple[np.ndarray, rasterio.Affine, int, int]] = []
    with rasterio.open(rgb_path) as src:
        row_idx = 0
        y_off = 0
        while y_off + SRC_CROP_RGB <= height:
            col_idx = 0
            x_off = 0
            while x_off + SRC_CROP_RGB <= width:
                patch = src.read(
                    indexes=[1, 2, 3],
                    window=Window(x_off, y_off, SRC_CROP_RGB, SRC_CROP_RGB),
                    out_shape=(3, TILE_SIZE_PX, TILE_SIZE_PX),
                    resampling=Resampling.cubic,
                )
                patch_transform = rasterio.Affine(
                    src.transform.a * SRC_CROP_RGB / TILE_SIZE_PX, 0,
                    src.transform.c + x_off * src.transform.a,
                    0, src.transform.e * SRC_CROP_RGB / TILE_SIZE_PX,
                    src.transform.f + y_off * src.transform.e,
                )
                patches.append((patch, patch_transform, row_idx, col_idx))
                col_idx += 1
                x_off += SRC_STRIDE_RGB
            row_idx += 1
            y_off += SRC_STRIDE_RGB
    return patches


# ---------------------------------------------------------------------------
# Dedup and elevation
# ---------------------------------------------------------------------------


def dedup_detections(
    detections: list[Detection], distance_m: float = 7.5,
) -> list[Detection]:
    """NMS-like spatial dedup on EPSG:2056 centroids per tile."""
    if len(detections) <= 1:
        return detections

    tiles: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        tiles[det.tile_id].append(det)

    kept: list[Detection] = []
    for tile_dets in tiles.values():
        if len(tile_dets) <= 1:
            kept.extend(tile_dets)
            continue

        indexed = sorted(
            enumerate(tile_dets), key=lambda x: x[1].confidence, reverse=True,
        )
        suppressed: set[int] = set()
        for i, (idx_i, det_i) in enumerate(indexed):
            if idx_i in suppressed:
                continue
            for j in range(i + 1, len(indexed)):
                idx_j, det_j = indexed[j]
                if idx_j in suppressed:
                    continue
                dist = np.hypot(
                    det_i.easting - det_j.easting,
                    det_i.northing - det_j.northing,
                )
                if dist < distance_m:
                    suppressed.add(idx_j)

        for idx, det in enumerate(tile_dets):
            if idx not in suppressed:
                kept.append(det)

    return kept


def max_elevation(dsm_path: str | Path) -> float:
    """Return max elevation of an on-disk DSM tile."""
    with rasterio.open(str(dsm_path)) as src:
        return float(src.read(1).max())


# ---------------------------------------------------------------------------
# Tile processing
# ---------------------------------------------------------------------------


def process_tile_from_cache(
    coord: str,
    rgb_url: str,
    dsm_url: str,
    neighbor_right: tuple[str, str] | None = None,
    neighbor_bottom: tuple[str, str] | None = None,
    neighbor_corner: tuple[str, str] | None = None,
    cache_patches: bool = False,  # kept for test compatibility; ignored
) -> list[tuple[np.ndarray, rasterio.Affine, int, int, str, str, str]]:
    """Produce 25 fused (3,640,640) uint8 patches for one tile.

    Missing neighbours become zero-padding in the VRT — the patch grid
    is always 5×5 regardless of neighbour availability (see
    `tests/test_patch_generation.py`). `cache_patches` is accepted for
    signature compatibility and ignored; the fused-patch disk cache was
    removed in favour of recomputing from the still-cached downloads.
    """
    del cache_patches  # no-op; accepted to preserve the test signature
    tile_id = coord.replace("-", "_")

    # --- Resolve center tile paths ---
    rgb_center_path = ensure_cached(rgb_url)
    dsm_center_path = ensure_cached(dsm_url)

    nb_rgb_paths: dict[str, Path] = {}
    nb_dsm_paths: dict[str, Path] = {}

    for name, nb in [
        ("right", neighbor_right),
        ("bottom", neighbor_bottom),
        ("corner", neighbor_corner),
    ]:
        if nb is None:
            continue
        nb_rgb, nb_dsm = nb
        try:
            nb_rgb_paths[name] = ensure_cached(nb_rgb)
            nb_dsm_paths[name] = ensure_cached(nb_dsm)
        except Exception as exc:
            log.warning(
                "neighbor %s fetch failed for %s: %s — padding with zeros",
                name, coord, exc,
            )

    # --- Build VRTs with fixed extended extent ---
    with rasterio.open(dsm_center_path) as src:
        cx0, cy0, cx1, cy1 = src.bounds
    strip_m = NEIGHBOR_STRIP_DSM * DSM_RES
    te = (cx0, cy0 - strip_m, cx1 + strip_m, cy1)

    dsm_paths = [dsm_center_path] + [
        nb_dsm_paths[n] for n in ("right", "bottom", "corner") if n in nb_dsm_paths
    ]
    rgb_paths = [rgb_center_path] + [
        nb_rgb_paths[n] for n in ("right", "bottom", "corner") if n in nb_rgb_paths
    ]

    dsm_vrt = _build_vrt(dsm_paths, te=te)
    rgb_vrt = _build_vrt(rgb_paths, te=te)
    rgb_unc = _materialize_uncompressed(rgb_vrt)

    try:
        hillshade = generate_hillshade(dsm_vrt)
        with rasterio.open(dsm_vrt) as src:
            dsm_transform = src.transform

        full_dsm_extent = TILE_PX_DSM + NEIGHBOR_STRIP_DSM
        full_rgb_extent = TILE_PX_RGB + NEIGHBOR_STRIP_RGB
        hillshade = hillshade[:full_dsm_extent, :full_dsm_extent]

        hs_patches = crop_patches(
            hillshade, dsm_transform,
            crop_px=SRC_CROP_DSM, stride_px=SRC_STRIDE_DSM, out_px=TILE_SIZE_PX,
        )
        del hillshade

        rgb_patches = _crop_resample_rgb(rgb_unc, full_rgb_extent, full_rgb_extent)
    finally:
        Path(dsm_vrt).unlink(missing_ok=True)
        Path(rgb_vrt).unlink(missing_ok=True)
        Path(rgb_unc).unlink(missing_ok=True)

    # --- Fuse RGB + hillshade (green channel replacement) ---
    results: list[tuple[np.ndarray, rasterio.Affine, int, int, str, str, str]] = []
    for (hs_patch, hs_tf, row, col), (rgb_patch, _, _, _) in zip(hs_patches, rgb_patches):
        rgb_patch[FUSION_CHANNEL] = hs_patch
        results.append((rgb_patch, hs_tf, row, col, tile_id, rgb_url, dsm_url))

    return results
