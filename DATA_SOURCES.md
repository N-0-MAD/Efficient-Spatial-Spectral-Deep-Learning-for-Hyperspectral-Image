# Data Sources

This project should use a staged data strategy: start with an easy multispectral source, then move to true hyperspectral data once the model and evaluation code are stable.

## Recommended Order

1. **Sentinel-2 L2A**
   - Role: engineering dataset and pipeline validation.
   - Bands: 12 commonly usable L2A reflectance bands in this project.
   - Strength: open, stable, easy to access through STAC.
   - Limitation: not a `>50` band stress test.
   - Access: Microsoft Planetary Computer STAC.

2. **EnMAP**
   - Role: best next candidate for a real spaceborne hyperspectral scalability test.
   - Bands: EnMAP documentation describes 246 contiguous bands from roughly 420 nm to 2450 nm.
   - Strength: modern hyperspectral Earth observation mission with official data access.
   - Access: https://www.enmap.org/data_access

3. **AVIRIS**
   - Role: high-quality airborne hyperspectral data, useful if coverage matches the study area or similar forest regions.
   - Strength: strong hyperspectral research history and public data portal.
   - Access: https://aviris.jpl.nasa.gov/data/get_aviris_data.html

4. **EMIT**
   - Role: accessible NASA hyperspectral data source for `>50` band scaling experiments.
   - Strength: public NASA products and tutorials; useful for model stress testing.
   - Limitation: mission focus is mineral dust source regions, so it may be less ideal for dense forest species work.
   - Access: https://earth.jpl.nasa.gov/emit/emit/data/data-products

5. **PRISMA**
   - Role: alternative spaceborne hyperspectral source.
   - Strength: 400-2500 nm imaging spectroscopy with 30 m pixels according to NASA/JPL SBG notes.
   - Access: https://www.asi.it/en/earth-science/prisma/

6. **Public hyperspectral benchmark cubes**
   - Role: fast local experiments before downloading large mission products.
   - Good for: Indian Pines, Pavia University, Salinas, Houston, Chikusei.
   - Limitation: many are older or small, so they should support model debugging rather than final forest-monitoring claims.

## Practical Choice

For this project:

```text
Sentinel-2 L2A -> Salinas -> Kennedy Space Center or NEON -> EnMAP or AVIRIS
```

That gives a smooth path from a working STAC pipeline to a true high-band scalability experiment.

## First Labeled Benchmark: Salinas

Salinas is the first training dataset in this repo.

- Sensor: NASA/JPL AVIRIS.
- Scene: Salinas Valley, California.
- Size: 512 lines by 217 samples.
- Original sensor bands: 224.
- Common corrected benchmark bands: 204 after removing water absorption bands.
- Labels: 16 land-cover/agricultural classes.
- Official source: https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes
- DOI mirror used by the downloader: https://zenodo.org/records/15771735
- Files used:
  - https://zenodo.org/records/15771735/files/Salinas_corrected.mat?download=1
  - https://zenodo.org/records/15771735/files/Salinas_gt.mat?download=1
