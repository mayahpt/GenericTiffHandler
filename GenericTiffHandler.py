import numpy as np
import os
from PIL import Image
import PIL.Image
import tifffile
import zarr
import dask.array as da
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm_joblib import ParallelPbar
from joblib import delayed, Parallel
import glob
import pathlib
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import dask

try:
    import pyvips
    _HAS_PYVIPS = True
except (ImportError, OSError):
    _HAS_PYVIPS = False
    pyvips = None

_HAS_HISTOMICSTK = False

PIL.Image.MAX_IMAGE_PIXELS = None

# ── Constants ──────────────────────────────────────────────────────────────────

SIMPLER_FILETYPES = {'.tif', '.tiff', '.svs', '.ndpi'}
COMPLEX_FILETYPES = {'.scn'}

THUMBNAIL_DOWNSAMPLE   = 20
TILE_RESIZE_SIZE       = 512
TISSUE_NORMALIZE_MAX   = 255
BBOX_PADDING           = 100
DEFAULT_CHUNK_2D       = (2048, 2048)
DEFAULT_CHUNK_3D       = (2048, 2048, 3)


# ── Helpers ────────────────────────────────────────────────────────────────────

def standardize_image_for_display(image):
    """
    Select an appropriate 2-D (or 2-D+channels) slice from an array of any
    supported dimensionality so that it can be displayed or passed to PIL.

    Rules
    -----
    - 2-D: returned unchanged.
    - 3-D (z, H, W): middle z-slice selected when first axis > 3 and last axis
      is not a channel count (1–3); channels-first (3, H, W) is transposed to
      (H, W, 3); otherwise returned unchanged.
    - 4-D: middle slice for (z, H, W, C) layouts; first slice for
      channels-first (C, …) layouts.
    - 5-D: batch/singleton axes are squeezed, then channels-first is transposed.
    """
    shape = image.shape
    ndim  = len(shape)

    if ndim == 2:
        return image

    if ndim == 3:
        if shape[0] > 3 and shape[-1] not in (1, 2, 3):
            return image[shape[0] // 2]
        if shape[0] == 3:
            return np.transpose(image, (1, 2, 0))
        return image

    if ndim == 4:
        if shape[0] > 3 and shape[-1] in (1, 2, 3):
            mid = image[shape[0] // 2]
            if mid.ndim == 3 and mid.shape[0] == 3:
                return np.transpose(mid, (1, 2, 0))
            return mid
        if shape[0] in (1, 2, 3):
            return image[0]
        return image[shape[0] // 2]

    if ndim == 5:
        squeezed = np.squeeze(image, axis=(0, 1))
        if squeezed.ndim == 3 and squeezed.shape[0] == 3:
            return np.transpose(squeezed, (1, 2, 0))
        return squeezed

    raise ValueError(f"Unexpected image shape: {shape}")


def standardize_shape(shape):
    """
    Extract the spatial (H, W) or (z, H, W) dimensions from a raw shape tuple.

    Rules
    -----
    - 2-D: returned as-is.
    - 3-D: strips the channel axis when it is the first (1–3) or last (1–3)
      element; otherwise assumed volumetric and returned as-is.
    - 4-D: strips the channel axis (first or last).
    - 5-D: returns the last two dimensions.
    """
    n = len(shape)

    if n == 2:
        return shape

    if n in (3, 4):
        if shape[0] in (1, 2, 3):
            return shape[1:]
        if shape[-1] in (1, 2, 3):
            return shape[:-1]
        return shape

    if n == 5:
        return shape[-2:]

    raise ValueError(f"Unexpected shape: {shape}")


def _get_file_extension(path):
    """Return the lower-cased extension (including dot) of *path*."""
    return os.path.splitext(path)[1].lower()


def _compute_if_dask(array):
    """Call ``.compute()`` on *array* if it is a Dask array; otherwise return it unchanged."""
    return array.compute() if isinstance(array, da.Array) else array


# ── Main class ─────────────────────────────────────────────────────────────────

class GenericTiffHandler:
    """
    Lazy-loading wrapper around TIFF-family whole-slide images.

    Supports ``.tif``, ``.tiff``, ``.svs``, ``.ndpi`` (simple layout) and
    ``.scn`` (complex/multi-series layout) via :mod:`tifffile` + :mod:`zarr` +
    :mod:`dask`.  Can also be constructed directly from a NumPy / Dask array
    via :meth:`from_array`.

    Parameters
    ----------
    path : str, optional
        Path to a TIFF-family file.
    image_array : array-like, optional
        Pre-loaded array.  Exactly one of *path* or *image_array* must be given.
    channel : int, optional
        If given, index the loaded array along axis 0 to select a single channel.
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, path=None, image_array=None, channel=None):
        if path is None and image_array is None:
            raise ValueError("Either 'path' or 'image_array' must be provided.")

        self.path        = None
        self.image_array = None
        self._tiff_image = None  # kept only while the file is open during init

        if path is not None:
            self._load_from_path(path, channel)
        else:
            self.image_array = image_array

        # Magnification / pixel-size metadata
        self.og_mag     = None
        self.og_mpp     = None
        self.current_mag = None
        self.current_mpp = None

        if self.path is not None:
            ext = _get_file_extension(self.path)
            self.og_mag      = self._read_original_magnification(ext)
            self.og_mpp      = self._read_original_pixel_size(ext)
            self.current_mag = self.og_mag
            self.current_mpp = self.og_mpp

        self.tissue_mask_path = None
        self.is_dask = isinstance(self.image_array, da.Array)

    def _load_from_path(self, path, channel):
        """Open *path* and populate ``self.image_array``."""
        self.path = path
        ext = _get_file_extension(path)

        if ext in SIMPLER_FILETYPES:
            with tifffile.TiffFile(path) as tif:
                store = tif.aszarr(level=0)
            z = zarr.open(store, mode='r')
            chunks = DEFAULT_CHUNK_2D if z.ndim == 2 else DEFAULT_CHUNK_3D
            self.image_array = da.from_zarr(z, chunks=chunks)
            if channel is not None:
                self.image_array = self.image_array[channel]

        elif ext in COMPLEX_FILETYPES:
            with tifffile.TiffFile(path) as tif:
                store = tif.series[1].pages[0].aszarr(level=0)
            z = zarr.open(store, mode='r')
            chunks = DEFAULT_CHUNK_2D if z.ndim == 2 else DEFAULT_CHUNK_3D
            self.image_array = da.from_zarr(z, chunks=chunks)

        else:
            raise ValueError(f"Unsupported file type: '{ext}'.")

    @classmethod
    def from_array(cls, image_array):
        """
        Construct a :class:`GenericTiffHandler` directly from an array.

        Parameters
        ----------
        image_array : numpy.ndarray or dask.array.Array

        Returns
        -------
        GenericTiffHandler
        """
        return cls(image_array=image_array)

    # ── Metadata helpers ──────────────────────────────────────────────────────

    def _get_tiff_file(self):
        """Re-open the source file and return a :class:`tifffile.TiffFile`."""
        if self.path is None:
            raise RuntimeError("No file path associated with this handler.")
        return tifffile.TiffFile(self.path)

    def _read_original_magnification(self, ext, verbose=False):
        """Parse the objective magnification from file metadata."""
        if ext == '.scn':
            with self._get_tiff_file() as tif:
                root = ET.fromstring(tif.scn_metadata)
            ns = "http://www.leica-microsystems.com/scn/2010/10/01"
            objectives = [
                np.float32(e.text)
                for e in root.findall(f".//{{{ns}}}objective")
            ]
            return int(np.max(objectives))

        if ext == '.svs':
            with self._get_tiff_file() as tif:
                desc = tif.pages[0].tags[270].value
            parts = [p.split('=')[1] for p in desc.split('|') if 'Mag' in p]
            return int(parts[0])

        if ext == '.ndpi':
            with self._get_tiff_file() as tif:
                return int(tif.pages[0].tags[65421].value)

        if verbose:
            print(f"No magnification metadata available for '{ext}'.")
        return None

    def _read_original_pixel_size(self, ext, verbose=False):
        """Parse the pixel size (µm/px, MPP) from file metadata."""
        if ext == '.scn':
            with self._get_tiff_file() as tif:
                x_res = tif.series[1].pages[0].tags['XResolution'].value[0]
                y_res = tif.series[1].pages[0].tags['YResolution'].value[0]
            return np.unique((10000 / x_res, 10000 / y_res))

        if ext == '.svs':
            with self._get_tiff_file() as tif:
                desc = tif.pages[0].tags[270].value
            parts = [p.split('=')[1] for p in desc.split('|') if 'MPP' in p]
            return float(parts[0])

        if ext == '.ndpi':
            with self._get_tiff_file() as tif:
                x_res = tif.pages[0].tags['XResolution'].value[0]
                y_res = tif.pages[0].tags['YResolution'].value[0]
            return np.unique((10000 / x_res, 10000 / y_res))

        if verbose:
            print(f"No pixel-size metadata available for '{ext}'.")
        return None

    # ── Public metadata API ───────────────────────────────────────────────────

    def get_original_magnification(self, ext=None, verbose=False):
        """Return the original (scan-time) magnification, reading from metadata if needed."""
        if self.og_mag is not None:
            return self.og_mag
        ext = ext or _get_file_extension(self.path)
        return self._read_original_magnification(ext, verbose=verbose)

    def get_original_pixel_size(self, ext=None, verbose=False):
        """Return the original pixel size (MPP), reading from metadata if needed."""
        if self.og_mpp is not None:
            return self.og_mpp
        ext = ext or _get_file_extension(self.path)
        return self._read_original_pixel_size(ext, verbose=verbose)

    def get_current_magnification(self):
        """Return the magnification at the current downsampling level."""
        return self.current_mag

    def get_current_pixel_size(self):
        """Return the pixel size (MPP) at the current downsampling level."""
        return self.current_mpp

    def set_magnification_settings(self, mag, mpp):
        """Manually set both the *original* and *current* magnification/MPP."""
        self.og_mag      = mag
        self.og_mpp      = mpp
        self.current_mag = mag
        self.current_mpp = mpp

    def get_current_path(self):
        """Return the filesystem path associated with this handler (may be ``None``)."""
        return self.path

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def get_image_dimensions(self):
        """Return the standardized spatial dimensions of the full image."""
        return standardize_shape(self.image_array.shape)

    def get_tile_dimensions(self, tile_height, tile_width, overlap):
        """
        Return ``(tiles_y, tiles_x)`` — the number of tiles along each axis.
        """
        dims = self.get_image_dimensions()
        img_height, img_width = dims[-2], dims[-1]
        tiles_y = int(np.ceil(img_height / tile_height))
        tiles_x = int(np.ceil(img_width  / tile_width))
        return tiles_y, tiles_x

    def get_coordinates_for_tile(self, col, row, tile_height, tile_width, overlap):
        """
        Return ``(coord_y, coord_x, eff_height, eff_width)`` for a tile at
        grid position ``(col, row)``.

        Parameters
        ----------
        col, row : int
            Zero-based tile indices (col = x, row = y) — OpenSlide convention.
        tile_height, tile_width : int
            Nominal tile size in pixels.
        overlap : int
            Overlap (in pixels) added on each shared edge.

        Raises
        ------
        ValueError
            When ``col`` or ``row`` fall outside the valid range.
        """
        dims = self.get_image_dimensions()
        img_height, img_width = dims[-2], dims[-1]
        tiles_y, tiles_x = self.get_tile_dimensions(tile_height, tile_width, overlap)

        if not (0 <= col < tiles_x):
            raise ValueError(f"Invalid col position: {col} (range 0–{tiles_x - 1})")
        if not (0 <= row < tiles_y):
            raise ValueError(f"Invalid row position: {row} (range 0–{tiles_y - 1})")

        coord_y = max(0, row * tile_height - overlap)
        coord_x = max(0, col * tile_width  - overlap)

        if row == tiles_y - 1:
            eff_height = img_height - coord_y
        else:
            eff_height = tile_height + (overlap if row == 0 else 2 * overlap)

        if col == tiles_x - 1:
            eff_width = img_width - coord_x
        else:
            eff_width = tile_width + (overlap if col == 0 else 2 * overlap)

        return coord_y, coord_x, eff_height, eff_width

    # ── Tile extraction ───────────────────────────────────────────────────────

    def get_tile(self, tile_height, tile_width, overlap, col, row, as_image=True):
        """
        Extract a tile from the image at grid position ``(col, row)``.
        Matches OpenSlide convention (col = x, row = y).

        Parameters
        ----------
        as_image : bool
            When ``True`` (default) return a :class:`PIL.Image.Image`;
            otherwise return the raw array slice.
        """
        slide = standardize_image_for_display(self.image_array)
        cy, cx, eff_h, eff_w = self.get_coordinates_for_tile(col, row, tile_height, tile_width, overlap)

        tile = (
            slide[cy:cy + eff_h, cx:cx + eff_w]
            if slide.ndim == 2
            else slide[cy:cy + eff_h, cx:cx + eff_w, :]
        )

        if not as_image:
            return tile
        return Image.fromarray(_compute_if_dask(tile))

    def get_tile_from_coordinates(self, x1, y1, x2, y2, as_image=True):
        """
        Extract a tile using absolute pixel coordinates.

        Parameters
        ----------
        x1, y1, x2, y2 : int
            Bounding-box corners in pixel space.
        as_image : bool
            When ``True`` return a :class:`PIL.Image.Image`.
        """
        slide = standardize_image_for_display(self.image_array)
        tile  = slide[y1:y2, x1:x2] if slide.ndim == 2 else slide[y1:y2, x1:x2, :]

        if not as_image:
            return tile
        return Image.fromarray(_compute_if_dask(tile))

    def get_thumbnail(self, thumbnail_size):
        """
        Return a thumbnail :class:`PIL.Image.Image` by uniform stride-based
        downsampling with factor *thumbnail_size*.
        """
        thumb = standardize_image_for_display(self.image_array)[::thumbnail_size, ::thumbnail_size]
        return Image.fromarray(_compute_if_dask(thumb))

    # ── Normalisation ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize_mask(mask):
        """
        Normalize *mask* to the ``[0, 1]`` range.

        An all-constant mask is returned as an all-zero array.
        """
        mask    = np.asarray(mask)
        lo, hi  = np.min(mask), np.max(mask)
        if hi == lo:
            return np.zeros_like(mask, dtype=np.uint8)
        return (mask - lo) / (hi - lo)

    # ── Magnification conversion ──────────────────────────────────────────────

    def convert_between_magnification(self, target_magnification=20, method='1'):
        """
        Downsample the image to *target_magnification*.

        Parameters
        ----------
        target_magnification : int
            Desired magnification.  Must be positive and ≤ current magnification.
        method : {'1', '2'}
            ``'1'`` uses stride-based slicing; ``'2'`` uses explicit index arrays.

        Raises
        ------
        ValueError
            For invalid *target_magnification* values or an unrecognised *method*.
        """
        if target_magnification == self.current_mag or target_magnification == 0:
            return

        if target_magnification < 0:
            raise ValueError("target_magnification must be positive.")
        if target_magnification > self.current_mag:
            raise ValueError("target_magnification must be ≤ current magnification.")
        if target_magnification > self.og_mag:
            raise ValueError("target_magnification must be ≤ original magnification.")

        factor = self.current_mag / target_magnification

        if method == '1':
            step = int(factor)
            self.image_array  = self.image_array[::step, ::step]
            self.current_mag  = target_magnification
            self.current_mpp  = self.current_mpp * factor

        elif method == '2':
            new_h = int(self.image_array.shape[0] / factor)
            new_w = int(self.image_array.shape[1] / factor)
            if self.is_dask:
                row_idx = (da.arange(new_h) * factor).astype(int)
                col_idx = (da.arange(new_w) * factor).astype(int)
            else:
                row_idx = (np.arange(new_h) * factor).astype(int)
                col_idx = (np.arange(new_w) * factor).astype(int)
            self.image_array  = self.image_array[row_idx][:, col_idx]
            self.current_mag  = target_magnification
            self.current_mpp  = self.current_mpp * factor

        else:
            raise ValueError("method must be '1' or '2'.")

    # ── Tissue-aware tile selection ───────────────────────────────────────────

    def calculate_useful_tiles(
        self,
        tile_height,
        tile_width,
        overlap,
        tissue_mask_path=None,
        tissue_percentage_threshold=25,
        cpu_workers=12,
        mode='newer',
        grid_step=4,
    ):
        """
        Return a list of ``(col, row)`` tile positions that contain sufficient
        tissue.

        Parameters
        ----------
        tile_height, tile_width : int
            Nominal tile size.
        overlap : int
            Tile overlap in pixels.
        tissue_mask_path : str, optional
            Path to a pre-computed tissue mask.  When ``None``, the method
            searches ``<dataset_root>/Tissue Masks/`` for a matching file; if
            nothing is found, it falls back to :mod:`histomicstk` per-tile
            segmentation.
        tissue_percentage_threshold : float
            Minimum tissue percentage (0–100) for a tile to be considered useful.
        cpu_workers : int
            Parallel workers for joblib.
        mode : {'naive', 'faster', 'newer'}
            ``'naive'``  – evaluates every tile in parallel.
            ``'faster'`` – coarse grid scan followed by local refinement.
            ``'newer'``  – bounding-box pre-filter on connected tissue regions.
        grid_step : int
            Step size for the coarse grid in ``'faster'`` mode.

        Returns
        -------
        list of (int, int)
            ``(col, row)`` pairs of useful tiles.

        Raises
        ------
        ValueError
            For unrecognised *mode*.
        RuntimeError
            When no file path is associated with this handler.
        """
        if self.path is None:
            raise RuntimeError("calculate_useful_tiles requires a file path.")
        if mode not in ('naive', 'faster', 'newer'):
            raise ValueError(f"Unknown mode '{mode}'. Choose 'naive', 'faster', or 'newer'.")

        # ── Locate or fall back to tissue mask ────────────────────────────────
        if tissue_mask_path is not None:
            self.tissue_mask_path = tissue_mask_path
        elif self.tissue_mask_path is None:
            dataset_path = os.path.dirname(os.path.dirname(self.path))
            stem         = os.path.splitext(os.path.basename(self.path))[0]
            candidates   = glob.glob(
                os.path.join(dataset_path, 'Tissue Masks', stem + '*.tiff')
            )
            self.tissue_mask_path = candidates[0] if candidates else None

        tiles_y, tiles_x = self.get_tile_dimensions(tile_height, tile_width, overlap)

        # ── Per-tile tissue masks (in-memory fallback) ────────────────────────
        tissue_masks = {}
        if self.tissue_mask_path is None:
            for row in range(tiles_y):
                for col in range(tiles_x):
                    tile = np.asarray(
                        self.get_tile(tile_height, tile_width, overlap, col, row)
                            .resize((TILE_RESIZE_SIZE, TILE_RESIZE_SIZE))
                    )
                    tissue_masks[(col, row)] = _compute_tissue_mask(tile)

        # ── Shared processing helpers ─────────────────────────────────────────
        proc_args = dict(
            tissue_mask_path=self.tissue_mask_path,
            tissue_masks=tissue_masks,
            tile_height=tile_height,
            tile_width=tile_width,
            overlap=overlap,
            threshold=tissue_percentage_threshold,
            og_mag=self.og_mag,
            og_mpp=self.og_mpp,
            current_mag=self.current_mag,
            current_mpp=self.current_mpp,
        )

        def _run_parallel(positions, label):
            results = ParallelPbar(label)(n_jobs=cpu_workers, backend='loky')(
                delayed(_evaluate_tile)(c, r, **proc_args)
                for c, r in positions
            )
            return [t for t in results if t is not None]

        # ── Mode: naive ───────────────────────────────────────────────────────
        if mode == 'naive':
            all_positions = [
                (c, r)
                for r in range(tiles_y)
                for c in range(tiles_x)
            ]
            return _run_parallel(all_positions, "Calculating useful tiles...")

        # ── Mode: faster ──────────────────────────────────────────────────────
        if mode == 'faster':
            coarse_positions = [
                (c, r)
                for r in range(0, tiles_y, grid_step)
                for c in range(0, tiles_x, grid_step)
            ]
            coarse_hits = _run_parallel(coarse_positions, "Selecting coarse results...")

            candidate_set = {
                (c + dr, r + dc)
                for c, r in coarse_hits
                for dc in range(-(grid_step - 1), grid_step)
                for dr in range(-(grid_step - 1), grid_step)
                if 0 <= r + dc < tiles_y and 0 <= c + dr < tiles_x
            }
            return _run_parallel(list(candidate_set), "Creating refined selection...")

        # ── Mode: newer ───────────────────────────────────────────────────────
        mask_obj = GenericTiffHandler(self.tissue_mask_path)
        mask_ext = _get_file_extension(self.tissue_mask_path)

        if mask_obj.get_original_pixel_size(mask_ext) is None:
            mask_obj.set_magnification_settings(
                self.get_original_magnification(mask_ext),
                self.get_original_pixel_size(mask_ext),
            )
        mask_obj.convert_between_magnification(self.current_mag)

        mask_thumb      = np.asarray(mask_obj.get_thumbnail(THUMBNAIL_DOWNSAMPLE))
        regions         = regionprops(label(mask_thumb))

        candidate_tiles = list({
            tile
            for region in regions
            for tile in _tiles_in_bbox(
                *_expand_bbox(region.bbox, BBOX_PADDING, THUMBNAIL_DOWNSAMPLE),
                tile_height, tile_width, overlap, mask_obj,
            )
        })
        return _run_parallel(candidate_tiles, "Creating refined selection...")

    # ── Saving ────────────────────────────────────────────────────────────────

    def save_to_tiff_with_metadata(self, saving_path):
        """
        Save the image as a tiled, pyramidal OME-TIFF with JPEG compression.

        Only 2-D (single-band) arrays are supported when saving from an in-memory
        array.  When a file path is available the image is read directly by
        :mod:`pyvips` and the 2-D restriction does not apply.

        Parameters
        ----------
        saving_path : str
            Destination path for the output file.

        Raises
        ------
        ValueError
            When no *saving_path* is provided or the in-memory array is not 2-D.
        """
        if not _HAS_PYVIPS:
            raise ImportError(
                "pyvips is required for saving TIFF files. "
                "Install with: pip install pyvips"
            )

        if saving_path is None:
            raise ValueError("saving_path must be provided.")

        if self.path is not None:
            vips_image = pyvips.Image.new_from_file(self.path, access="sequential")
        else:
            array = _compute_if_dask(self.image_array)
            if array.ndim != 2:
                raise ValueError("Only 2-D arrays are supported for saving.")
            h, w = array.shape
            vips_image = pyvips.Image.new_from_memory(
                array.tobytes(), w, h, bands=1, format='uchar'
            )

        img   = vips_image.copy()
        xml   = _build_ome_xml(img.width, img.height, img.bands)

        img.set_type(pyvips.GValue.gint_type, "page-height", img.height)
        img.set_type(pyvips.GValue.gstr_type, "image-description", xml)

        img.tiffsave(
            saving_path,
            compression="jpeg",
            tile=True,
            tile_width=512,
            tile_height=512,
            Q=100,
            pyramid=True,
            subifd=True,
        )
        return True


# ── Module-level helpers (extracted from methods) ─────────────────────────────

def _compute_tissue_mask(tile: np.ndarray) -> np.ndarray:
    """
    Produce a binary tissue mask for a single tile using Otsu thresholding.
    Returns a uint8 array of 0/1 values matching tile's
    spatial dimensions.
    """
    try:
        gray = rgb2gray(tile) if tile.ndim == 3 else tile.astype(float)
        threshold = threshold_otsu(gray)
        # Tissue is darker than background in H&E slides
        mask = (gray < threshold).astype(np.uint8)
        return mask
    except Exception:
        return np.zeros(tile.shape[:2], dtype=np.uint8)


def _evaluate_tile(c, r, *, tissue_mask_path, tissue_masks,
                   tile_height, tile_width, overlap, threshold,
                   og_mag, og_mpp, current_mag, current_mpp):
    """
    Return ``(c, r)`` when the tile contains more than *threshold* %
    tissue, otherwise return ``None``.

    This function is designed to be called in parallel via joblib.
    """
    if tissue_mask_path is None:
        tile_tissue = tissue_masks[(c, r)]
    else:
        mask_obj = GenericTiffHandler(tissue_mask_path)
        mask_obj.set_magnification_settings(og_mag, og_mpp)
        mask_obj.convert_between_magnification(current_mag, method='1')
        tile_tissue = np.asarray(
            mask_obj.get_tile(tile_height, tile_width, overlap, c, r)
        )
        if tile_tissue.size == 0:
            return None
        if tile_tissue.max() == TISSUE_NORMALIZE_MAX:
            tile_tissue = tile_tissue / TISSUE_NORMALIZE_MAX

    pct = tile_tissue.sum() / tile_tissue.size * 100
    return (c, r) if pct > threshold else None


def _expand_bbox(bbox, padding, downsample):
    """
    Scale a bounding box from thumbnail space back to full-resolution space
    and add *padding* pixels on every side.

    Parameters
    ----------
    bbox : (row_min, col_min, row_max, col_max)
        Bounding box in thumbnail coordinates.
    padding : int
        Extra pixels to add on each side in full-resolution space.
    downsample : int
        The thumbnail downsampling factor.

    Returns
    -------
    (x_min, y_min, x_max, y_max)
        Expanded bounding box in full-resolution *tile-grid* coordinate order
        (x = column direction, y = row direction).
    """
    row_min, col_min, row_max, col_max = bbox
    y_min = row_min * downsample - padding
    x_min = col_min * downsample - padding
    y_max = row_max * downsample + padding
    x_max = col_max * downsample + padding
    return x_min, y_min, x_max, y_max


def _tiles_in_bbox(x_min, y_min, x_max, y_max,
                   tile_height, tile_width, overlap, handler):
    """
    Yield ``(col, row)`` indices for all tiles that intersect the given
    bounding box.

    Parameters
    ----------
    x_min, y_min, x_max, y_max : int
        Bounding box in full-resolution pixel coordinates.
    handler : GenericTiffHandler
        Used to resolve tile coordinates.
    """
    tiles_y, tiles_x = handler.get_tile_dimensions(tile_height, tile_width, overlap)
    for row in range(tiles_y):
        for col in range(tiles_x):
            ty, tx, th, tw = handler.get_coordinates_for_tile(
                col, row, tile_height, tile_width, overlap
            )
            # Standard rectangle-overlap test
            if not (ty + th <= y_min or ty >= y_max or
                    tx + tw <= x_min or tx >= x_max):
                yield (col, row)


def _build_ome_xml(width, height, bands):
    """Return a minimal OME-XML metadata string for a single uint8 image."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        ' xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06'
        ' http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">'
        '<Image ID="Image:0">'
        '<Pixels DimensionOrder="XYCZT" ID="Pixels:0"'
        f' SizeC="{bands}" SizeT="1" SizeX="{width}" SizeY="{height}"'
        ' SizeZ="1" Type="uint8"/>'
        '</Image>'
        '</OME>'
    )