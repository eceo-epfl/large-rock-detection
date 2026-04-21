"""Shared constants for training and nationwide inference pipelines.

Training-critical values (tile size, overlap, resolutions, fusion channel)
are baked into the trained model and must not be changed without retraining.
"""

import re

# YOLO input size — model trained on 640×640 patches.
TILE_SIZE_PX = 640

# Patch overlap — ensures rocks near tile edges appear fully in ≥1 patch.
# Yields stride=430px, 4×4=16 patches per 1km tile.
OVERLAP_PX = 210

# Output resolution — training images are 50cm/pixel.
TARGET_RES = 0.5  # m/pixel

# SwissIMAGE DOP10 native resolution.
SWISSIMAGE_RES = 0.1  # m/pixel

# swissALTI3D raster native resolution.
DSM_RES = 0.5  # m/pixel

# Green channel replacement — best fusion strategy per baseline experiments.
# See src/baseline/baseline_dataset.py:58.
FUSION_CHANNEL = 1

# Derived patch geometry
TILE_GROUND_M = TILE_SIZE_PX * TARGET_RES               # 320m
STRIDE_PX = TILE_SIZE_PX - OVERLAP_PX                   # 430px
STRIDE_GROUND_M = STRIDE_PX * TARGET_RES                # 215m
SRC_CROP_DSM = int(TILE_GROUND_M / DSM_RES)             # 640px
SRC_STRIDE_DSM = int(STRIDE_GROUND_M / DSM_RES)         # 430px
SRC_CROP_RGB = int(TILE_GROUND_M / SWISSIMAGE_RES)      # 3200px at 10cm
SRC_STRIDE_RGB = int(STRIDE_GROUND_M / SWISSIMAGE_RES)  # 2150px at 10cm

# Neighbor strip sizes for cross-tile stitching (5th patch extends past tile edge)
TILE_PX_DSM = int(1000 / DSM_RES)                       # 2000px
TILE_PX_RGB = int(1000 / SWISSIMAGE_RES)                 # 10000px
NEIGHBOR_STRIP_DSM = 4 * SRC_STRIDE_DSM + SRC_CROP_DSM - TILE_PX_DSM  # 360px
NEIGHBOR_STRIP_RGB = 4 * SRC_STRIDE_RGB + SRC_CROP_RGB - TILE_PX_RGB  # 1800px

# Swisstopo CDN URL templates (year + coord are filled at runtime)
SI_TEMPLATE = (
    "https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/"
    "swissimage-dop10_{year}_{coord}/"
    "swissimage-dop10_{year}_{coord}_0.1_2056.tif"
)

DSM_TEMPLATE = (
    "https://data.geo.admin.ch/ch.swisstopo.swissalti3d/"
    "swissalti3d_{year}_{coord}/"
    "swissalti3d_{year}_{coord}_0.5_2056_5728.tif"
)

# STAC API
STAC_BASE = "https://data.geo.admin.ch/api/stac/v0.9"
SI_COLLECTION = "ch.swisstopo.swissimage-dop10"
DSM_COLLECTION = "ch.swisstopo.swissalti3d"

# Tile coordinate pattern (e.g. "2587-1133")
COORD_RE = re.compile(r"(\d{4}-\d{4})")

# Union of SwissIMAGE + swissALTI3D spatial extents (WGS84), from:
#   GET https://data.geo.admin.ch/api/stac/v0.9/collections/{collection}
#   → extent.spatial.bbox
# SwissIMAGE:   [5.9503666, 45.8151271, 10.4998461, 47.8091281]
# swissALTI3D:  [5.9503666, 45.7213375, 10.4998461, 47.8216742]
# We use the union so the STAC query returns all tiles from both layers;
# query_stac_bbox() then intersects the results to keep only matched pairs.
SWITZERLAND_BBOX = "5.95,45.72,10.50,47.83"
