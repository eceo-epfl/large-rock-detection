"""Disk caches for the nationwide pipeline.

Two caches live here:

* `TileCache` — an LRU-evicted store of downloaded Swisstopo GeoTIFFs
  keyed on the URL filename. Sized by a bytes budget; eviction is lazy
  (the O(N) directory scan only fires once ~1 GB has been written, not
  on every `put`).
* STAC response cache — a DuckDB table that maps a WGS84 bbox to the
  list of `(coord, rgb_url, dsm_url)` tuples returned by the STAC API.
  Spares repeat bbox runs from re-querying.

Neighbour URL resolution used to live here too; it now reads directly
from the STAC result dict, so there is no separate neighbour cache.
"""

import logging
import threading
from pathlib import Path
from urllib.parse import urlparse

import duckdb

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tile cache (LRU evicted)
# ---------------------------------------------------------------------------


class TileCache:
    """File-based LRU cache of downloaded tiles.

    Keys are the URL's filename (already unique for Swisstopo). LRU is
    tracked by file mtime — `path()` and `get()` touch the file on hit
    so recently-used files stay. `put()` tallies bytes written and only
    triggers a directory scan (for eviction) once the tally crosses a
    threshold, so per-put cost is constant.
    """

    _EVICT_THRESHOLD_BYTES = 1_000_000_000  # 1 GB

    def __init__(self, cache_dir: Path, max_bytes: int) -> None:
        self._dir = cache_dir
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        self._bytes_since_check = 0
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(url: str) -> str:
        return Path(urlparse(url).path).name

    def get(self, url: str) -> bytes | None:
        path = self._dir / self._key(url)
        try:
            data = path.read_bytes()
            if not data:
                path.unlink(missing_ok=True)
                return None
            path.touch()
            return data
        except (FileNotFoundError, OSError):
            return None

    def path(self, url: str) -> Path | None:
        p = self._dir / self._key(url)
        try:
            if p.exists() and p.stat().st_size > 0:
                p.touch()
                return p
        except OSError:
            pass
        return None

    def put(self, url: str, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            path = self._dir / self._key(url)
            path.write_bytes(data)
            self._bytes_since_check += len(data)
            threshold = max(self._max_bytes * 0.05, self._EVICT_THRESHOLD_BYTES)
            should_check = self._bytes_since_check > threshold
            if should_check:
                self._bytes_since_check = 0
        if should_check:
            self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        # Advisory: best-effort across worker processes. Files are
        # unlinked with missing_ok=True, so the worst case is a single
        # re-download if two processes race.
        with self._lock:
            try:
                files = [(f, f.stat()) for f in self._dir.iterdir() if f.is_file()]
            except OSError:
                return
            total = sum(s.st_size for _, s in files)
            if total <= self._max_bytes:
                return
            files.sort(key=lambda x: x[1].st_mtime)
            for f, s in files:
                if total <= self._max_bytes:
                    break
                try:
                    f.unlink()
                except (FileNotFoundError, OSError):
                    continue
                total -= s.st_size


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------


_tile_cache: TileCache | None = None
_cache_config: tuple[str, int] | None = None


def init_cache(cache_dir: Path, max_gb: float) -> None:
    """Set up the on-disk tile cache. Pass max_gb=0 to disable."""
    global _tile_cache, _cache_config
    if max_gb <= 0:
        _tile_cache = None
        _cache_config = None
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_gb * 1_000_000_000)
    _tile_cache = TileCache(cache_dir, max_bytes)
    _cache_config = (str(cache_dir), max_bytes)

    try:
        dl_bytes = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
    except OSError:
        dl_bytes = 0
    log.info(
        "Tile cache: %s (%.1f/%.1f GB)",
        cache_dir, dl_bytes / 1e9, max_gb,
    )


def get_cache_config() -> tuple[str, int] | None:
    return _cache_config


def reinit_cache(cache_dir: str | None, max_bytes: int) -> None:
    """Re-create the tile cache in a worker process after fork."""
    global _tile_cache
    if cache_dir is not None and max_bytes > 0:
        _tile_cache = TileCache(Path(cache_dir), max_bytes)
    else:
        _tile_cache = None


def cache_path(url: str) -> Path | None:
    if _tile_cache is None:
        return None
    return _tile_cache.path(url)


def cache_get(url: str) -> bytes | None:
    if _tile_cache is None:
        return None
    return _tile_cache.get(url)


def cache_put(url: str, data: bytes) -> None:
    if _tile_cache is not None:
        _tile_cache.put(url, data)


# ---------------------------------------------------------------------------
# STAC response cache (DuckDB)
# ---------------------------------------------------------------------------


_stac_cache_path: Path | None = None


def set_stac_cache_dir(cache_dir: Path) -> None:
    global _stac_cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    _stac_cache_path = cache_dir / "stac_cache.duckdb"


def load_stac_cache(bbox: str) -> list[tuple[str, str, str]] | None:
    if _stac_cache_path is None or not _stac_cache_path.exists():
        return None
    try:
        con = duckdb.connect(str(_stac_cache_path), read_only=True)
        try:
            row = con.execute(
                "SELECT n_tiles FROM stac_cache_meta WHERE bbox = ?", [bbox],
            ).fetchone()
            if row is None:
                return None
            tiles = con.execute(
                "SELECT coord, rgb_url, dsm_url FROM stac_cache WHERE bbox = ? ORDER BY coord",
                [bbox],
            ).fetchall()
        finally:
            con.close()
        if len(tiles) != row[0]:
            log.warning("STAC cache count mismatch, re-querying")
            return None
        return [(c, r, d) for c, r, d in tiles]
    except Exception as exc:
        log.warning(f"STAC cache read failed: {exc}")
        return None


def save_stac_cache(bbox: str, tiles: list[tuple[str, str, str]]) -> None:
    if _stac_cache_path is None:
        return
    try:
        con = duckdb.connect(str(_stac_cache_path))
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS stac_cache (
                    bbox     VARCHAR,
                    coord    VARCHAR,
                    rgb_url  VARCHAR NOT NULL,
                    dsm_url  VARCHAR NOT NULL,
                    PRIMARY KEY (bbox, coord)
                )
                """
            )
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS stac_cache_meta (
                    bbox       VARCHAR PRIMARY KEY,
                    n_tiles    INTEGER,
                    cached_at  TIMESTAMP DEFAULT current_timestamp
                )
                """
            )
            con.execute("DELETE FROM stac_cache WHERE bbox = ?", [bbox])
            con.execute("DELETE FROM stac_cache_meta WHERE bbox = ?", [bbox])
            con.executemany(
                "INSERT INTO stac_cache (bbox, coord, rgb_url, dsm_url) VALUES (?, ?, ?, ?)",
                [(bbox, c, r, d) for c, r, d in tiles],
            )
            con.execute(
                "INSERT INTO stac_cache_meta (bbox, n_tiles) VALUES (?, ?)",
                [bbox, len(tiles)],
            )
        finally:
            con.close()
        log.info(f"STAC cache saved: {len(tiles)} tile pairs for bbox={bbox}")
    except Exception as exc:
        log.warning(f"STAC cache write failed: {exc}")
