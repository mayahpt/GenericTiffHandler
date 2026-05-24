# GenericTiffHandler

A lazy-loading wrapper around TIFF-family whole-slide images (WSI), supporting
`.tif`, `.tiff`, `.svs`, `.ndpi`, and `.scn` formats via `tifffile` + `zarr` + `dask`.

Designed as a lightweight alternative to OpenSlide for research and
histopathology pipelines.

## Features

- **Lazy loading** — images are loaded as Dask arrays, keeping memory usage low
- **Multi-format** — supports SVS, NDPI, SCN, and plain TIFF files
- **Tile extraction** — grid-based and coordinate-based tile extraction
- **Tissue-aware sampling** — three modes (`naive`, `faster`, `newer`) for
  selecting tiles containing tissue
- **Magnification management** — read metadata from SVS, NDPI, SCN; down-sample
  to target magnification
- **OME-TIFF export** — save as tiled, pyramidal OME-TIFF with JPEG compression
  (requires `pyvips`)

## Installation

### Quick install (pip)

```bash
pip install -r requirements.txt
```

### Optional: pyvips (for saving OME-TIFF)

```bash
pip install pyvips
```

## Quick Start

```python
from GenericTiffHandler import GenericTiffHandler

# Load a whole-slide image
wsi = GenericTiffHandler("path/to/slide.svs")

# Get image dimensions
print(wsi.get_image_dimensions())

# Extract a thumbnail
thumb = wsi.get_thumbnail(thumbnail_size=20)
thumb.save("thumbnail.jpg")

# Extract a tile at grid position (5, 4)
tile = wsi.get_tile(tile_height=512, tile_width=512, overlap=0, y=5, x=4)
tile.save("tile_5_4.jpg")
```

## Requirements

- numpy, Pillow, tifffile, zarr, dask, matplotlib
- scikit-image, joblib, tqdm, tqdm-joblib
- pyvips *(optional — needed only for `save_to_tiff_with_metadata()`)*

## Author

**Maya Silva** — mayahptsilva@gmail.com

## License

MIT
