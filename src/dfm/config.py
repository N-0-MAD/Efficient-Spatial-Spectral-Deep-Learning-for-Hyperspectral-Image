"""Shared project constants."""

from __future__ import annotations

SENTINEL2_L2A_REFLECTANCE_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

SENTINEL2_RGB_ORDER: tuple[str, str, str] = ("B04", "B03", "B02")

REDWOOD_NATIONAL_PARK = {
    "latitude": 41.3,
    "longitude": -124.0,
    "buffer_degrees": 0.05,
}


def bbox_from_center(latitude: float, longitude: float, buffer_degrees: float) -> list[float]:
    """Return a STAC-style [min_lon, min_lat, max_lon, max_lat] bbox."""

    return [
        longitude - buffer_degrees,
        latitude - buffer_degrees,
        longitude + buffer_degrees,
        latitude + buffer_degrees,
    ]

