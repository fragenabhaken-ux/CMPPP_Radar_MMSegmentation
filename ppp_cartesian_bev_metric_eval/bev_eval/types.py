from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BEVGeometry:
    """Geometry of a Cartesian BEV raster.

    Array convention:
        rows    -> x_forward in [x_min, x_max)
        columns -> y_lateral in [y_min, y_max)
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: int
    width: int

    def __post_init__(self) -> None:
        if not np.isfinite([self.x_min, self.x_max, self.y_min, self.y_max]).all():
            raise ValueError("BEV bounds must be finite.")
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min must be < x_max, got {self.x_min} >= {self.x_max}.")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min must be < y_max, got {self.y_min} >= {self.y_max}.")
        if self.height <= 0 or self.width <= 0:
            raise ValueError(f"Invalid BEV shape {(self.height, self.width)}.")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.height

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.width

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def cell_centres(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float64)
        cols = np.asarray(cols, dtype=np.float64)
        x = self.x_min + (rows + 0.5) * self.dx
        y = self.y_min + (cols + 0.5) * self.dy
        return np.stack([x, y], axis=-1)

    def point_indices(
        self,
        xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return row indices, column indices, and an in-bounds mask."""
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected xy with shape (N, 2), got {xy.shape}.")

        x = xy[:, 0]
        y = xy[:, 1]
        finite = np.isfinite(x) & np.isfinite(y)
        inside = (
            finite
            & (x >= self.x_min)
            & (x < self.x_max)
            & (y >= self.y_min)
            & (y < self.y_max)
        )

        rows = np.full(x.shape, -1, dtype=np.int64)
        cols = np.full(y.shape, -1, dtype=np.int64)
        rows[inside] = np.floor((x[inside] - self.x_min) / self.dx).astype(np.int64)
        cols[inside] = np.floor((y[inside] - self.y_min) / self.dy).astype(np.int64)

        # Guard against upper-bound floating point round-off.
        rows[inside] = np.clip(rows[inside], 0, self.height - 1)
        cols[inside] = np.clip(cols[inside], 0, self.width - 1)
        return rows, cols, inside

    @classmethod
    def from_attrs(
        cls,
        attrs: dict[str, Any],
        shape: tuple[int, int],
    ) -> "BEVGeometry":
        h, w = int(shape[0]), int(shape[1])
        required = ["bev_x_min", "bev_x_max", "bev_y_min", "bev_y_max"]
        missing = [key for key in required if key not in attrs]
        if missing:
            raise KeyError(
                "HDF5 export is missing Cartesian BEV geometry attributes: "
                f"{missing}."
            )

        attr_h = int(attrs.get("bev_height", h))
        attr_w = int(attrs.get("bev_width", w))
        if (attr_h, attr_w) != (h, w):
            raise ValueError(
                "HDF5 BEV shape attributes do not match pred_lambda: "
                f"attrs={(attr_h, attr_w)}, array={(h, w)}."
            )

        return cls(
            x_min=float(attrs["bev_x_min"]),
            x_max=float(attrs["bev_x_max"]),
            y_min=float(attrs["bev_y_min"]),
            y_max=float(attrs["bev_y_max"]),
            height=h,
            width=w,
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "bev_x_min": self.x_min,
            "bev_x_max": self.x_max,
            "bev_y_min": self.y_min,
            "bev_y_max": self.y_max,
            "bev_height": self.height,
            "bev_width": self.width,
            "bev_dx": self.dx,
            "bev_dy": self.dy,
            "bev_cell_area": self.cell_area,
        }
