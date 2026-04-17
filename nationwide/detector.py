"""YOLO detector for the nationwide pipeline.

Everything torch / ultralytics lives in this file. To apply a different
model family, replace this file with one exposing the same `Detector`
surface: `__init__(model_path, device, conf, iou, imgsz)`, `warmup()`,
and `detect(patches) -> list[Detection]`.
"""

import logging
from pathlib import Path

import numpy as np

from nationwide.db import Detection
from utils.constants import TILE_SIZE_PX

log = logging.getLogger(__name__)


def _yolo_to_map_coords(
    cx: float, cy: float, w: float, h: float,
    img_size: int, transform,
) -> tuple[float, float, float, float]:
    """Convert YOLO normalized box to EPSG:2056 centroid + bbox metres."""
    px, py = cx * img_size, cy * img_size
    pw, ph = w * img_size, h * img_size
    easting, northing = transform * (px, py)
    width_m = abs(pw * transform.a)
    height_m = abs(ph * transform.e)
    return easting, northing, width_m, height_m


def _extract(results, meta) -> list[Detection]:
    detections: list[Detection] = []
    for result, (transform, row, col, tile_id, rgb_url, dsm_url) in zip(results, meta):
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        patch_id = f"{tile_id}_{row}_{col}"
        for i in range(len(boxes)):
            cx, cy, w, h = boxes.xywhn[i].cpu().numpy()
            conf_val = float(boxes.conf[i].cpu())
            cls = int(boxes.cls[i].cpu())
            easting, northing, w_m, h_m = _yolo_to_map_coords(
                cx, cy, w, h, TILE_SIZE_PX, transform,
            )
            detections.append(Detection(
                tile_id=tile_id, patch_id=patch_id,
                easting=float(easting), northing=float(northing),
                confidence=conf_val,
                bbox_w_m=float(w_m), bbox_h_m=float(h_m),
                class_id=cls, rgb_source=rgb_url, dsm_source=dsm_url,
            ))
    return detections


class Detector:
    """Thin wrapper around ultralytics YOLO. Owns the model and the device."""

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        conf: float = 0.10,
        iou: float = 0.70,
        imgsz: int = TILE_SIZE_PX,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(f"Device {device} requested but no GPU available.")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._torch = torch
        self._model = YOLO(model_path)
        if device != "cpu":
            self._model.to(device)
        self._device = device
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        log.info(f"Detector: {model_path} on {device}")

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        if self._device == "cpu":
            return
        dummy = self._torch.zeros(1, 3, self._imgsz, self._imgsz, device=self._device)
        self._model.predict(
            source=dummy, conf=self._conf, iou=self._iou, imgsz=self._imgsz,
            save=False, verbose=False,
        )
        del dummy
        self._torch.cuda.empty_cache()

    def detect(self, patches: list[tuple]) -> list[Detection]:
        """Run the model on a list of `(patch_array, transform, row, col, tile_id, rgb_url, dsm_url)`.

        Patch arrays are uint8 `(3, H, W)`. Conversion to float is done on
        the device to avoid a host-side float32 buffer. On GPU OOM the
        chunk size is halved in a loop (not recursion) and retried.
        """
        if not patches:
            return []

        arrays = [p[0] for p in patches]
        meta = [(p[1], p[2], p[3], p[4], p[5], p[6]) for p in patches]
        stacked = np.stack(arrays)  # uint8, host

        detections: list[Detection] = []
        chunk_size = len(arrays)
        start = 0
        while start < len(arrays):
            end = min(start + chunk_size, len(arrays))
            tensor = None
            try:
                tensor = self._torch.from_numpy(
                    np.ascontiguousarray(stacked[start:end])
                ).to(self._device, non_blocking=(self._device != "cpu"))
                tensor = tensor.float().div_(255.0)
                results = self._model.predict(
                    source=tensor, conf=self._conf, iou=self._iou,
                    imgsz=self._imgsz, save=False, verbose=False,
                )
            except (self._torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                del tensor
                self._torch.cuda.empty_cache()
                new_chunk = chunk_size // 2
                log.warning(
                    "OOM on chunk of %d patches; halving to %d",
                    end - start, new_chunk,
                )
                if new_chunk == 0:
                    raise RuntimeError(
                        "CUDA OOM on a single-patch chunk — cannot proceed"
                    ) from exc
                chunk_size = new_chunk
                continue  # retry this slice with smaller chunk
            detections.extend(_extract(results, meta[start:end]))
            start = end

        return detections
