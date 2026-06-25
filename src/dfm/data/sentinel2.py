"""Sentinel-2 STAC loading helpers.

These functions import STAC dependencies lazily so local tests can run without
remote sensing packages installed.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from dfm.config import SENTINEL2_L2A_REFLECTANCE_BANDS, bbox_from_center


def redwood_bbox(latitude: float = 41.3, longitude: float = -124.0, buffer: float = 0.05) -> list[float]:
    """Return the default Redwood National Park bounding box."""

    return bbox_from_center(latitude=latitude, longitude=longitude, buffer_degrees=buffer)


def search_sentinel2_l2a(
    bbox: list[float],
    date_range: str = "2023-06-01/2023-08-31",
    max_cloud_cover: float = 5,
):
    """Search Microsoft Planetary Computer for Sentinel-2 L2A items."""

    try:
        import planetary_computer
        import pystac_client
    except ImportError as exc:
        raise ImportError(
            "Install pystac-client and planetary-computer to search Sentinel-2 STAC."
        ) from exc

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    return list(search.items())


def load_sentinel2_l2a(
    items: Iterable,
    bbox: list[float],
    bands: Iterable[str] = SENTINEL2_L2A_REFLECTANCE_BANDS,
    resolution: int = 20,
):
    """Load Sentinel-2 L2A bands as an xarray dataset."""

    try:
        import odc.stac
    except ImportError as exc:
        raise ImportError("Install odc-stac to load Sentinel-2 STAC items.") from exc

    return odc.stac.load(
        list(items),
        bbox=bbox,
        bands=list(bands),
        resolution=resolution,
    )


def xarray_to_chw(data, bands: Iterable[str] = SENTINEL2_L2A_REFLECTANCE_BANDS) -> np.ndarray:
    """Convert an odc-stac xarray dataset to a `(C, H, W)` NumPy array."""

    arrays = []
    for band in bands:
        if band not in data:
            raise KeyError(f"Band {band} was not found in the xarray dataset.")
        band_array = data[band]
        if "time" in band_array.dims:
            band_array = band_array.isel(time=0)
        arrays.append(np.asarray(band_array.values))
    return np.stack(arrays, axis=0)

