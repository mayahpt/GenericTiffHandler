import json
import numpy as np
import os
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import glob
import shutil
import pyvips
import tqdm
import PIL.Image
import tifffile
import zarr
import dask.array as da
import matplotlib.pyplot as plt
import openslide
from openslide.deepzoom import DeepZoomGenerator
import xml.etree.ElementTree as ET  # Needed for XML metadata processing

#-------------------------------------------------------------------
# Setup OpenSlide and VIPS paths
OPENSLIDE_PATH = os.path.abspath((os.path.join(os.path.dirname(os.path.realpath(__file__)),r".\Utils\OpenSlide\openslide-bin-4.0.0.3-windows-x64\bin")))
VIPS_BIN_PATH = os.path.abspath((os.path.join(os.path.dirname(os.path.realpath(__file__)),r".\Utils\VIPS\vips-dev-8.15\bin")))

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
    os.add_dll_directory(VIPS_BIN_PATH)
else:
    os.environ['PATH'] = os.pathsep.join((VIPS_BIN_PATH, os.environ['PATH']))

# Increase maximum number of pixels that PIL can process
PIL.Image.MAX_IMAGE_PIXELS = None 

#-------------------------------------------------------------------
def standardize_image_for_display(image):
    """
    Prepares an image for display by selecting an appropriate slice and reordering dimensions.
    
    - 2D images are returned unchanged.
    - For 3D arrays:
         • If the first dimension is larger than 3 and the last is not 1-3, assume a z-stack 
           and select the middle slice.
         • If the image is channels-first (3, height, width), it is transposed to (height, width, channels).
         • Otherwise, the image is returned as is.
    - For 4D arrays (e.g. (z, height, width, channels) or (channels, z, height, width)):
         • If a z-stack is detected, the middle slice is used.
         • Otherwise, channels-first images are handled by selecting the first slice.
    - For 5D arrays, extra dimensions are squeezed before processing.
    """
    shape = image.shape
    ndim = len(shape)
    
    if ndim == 2:
        return image
    elif ndim == 3:
        # Check if it's a z-stack: first dimension > 3 and last dimension not typical for channels.
        if shape[0] > 3 and shape[-1] not in (1, 2, 3):
            mid_index = shape[0] // 2
            return image[mid_index]
        # Channels-first image: (3, height, width)
        if shape[0] == 3:
            return np.transpose(image, (1, 2, 0))
        return image
    elif ndim == 4:
        # Likely (z, height, width, channels) or (channels, z, height, width)
        if shape[0] > 3 and shape[-1] in (1, 2, 3):
            mid_index = shape[0] // 2
            slice_img = image[mid_index]
            if slice_img.ndim == 3 and slice_img.shape[0] == 3:
                return np.transpose(slice_img, (1, 2, 0))
            return slice_img
        elif shape[0] in (1, 2, 3):
            return image[0]
        else:
            mid_index = shape[0] // 2
            return image[mid_index]
    elif ndim == 5:
        # Squeeze extra dimensions (e.g. batch dimensions) and process as above.
        squeezed = np.squeeze(image, axis=(0, 1))
        if squeezed.ndim == 3 and squeezed.shape[0] == 3:
            return np.transpose(squeezed, (1, 2, 0))
        return squeezed
    else:
        raise ValueError(f"Unexpected image shape: {shape}")

#-------------------------------------------------------------------
def standardize_shape(shape):
    """
    Standardizes the shape of an image array to extract its spatial dimensions.
    
    - For 2D images, returns (height, width).
    - For 3D images, distinguishes between:
         • Channels-first (e.g. (channels, height, width)) if the first element is 1-3.
         • Channels-last (e.g. (height, width, channels)) if the last element is 1-3.
         • A volumetric (z-stack) image if neither axis represents typical channels.
    - For 4D images, assumes either (z, height, width, channels) or (channels, z, height, width) and removes the channel axis.
    - For 5D images, squeezes extra dimensions and returns the last two dimensions.
    """
    if len(shape) == 2:
        return shape
    elif len(shape) == 3:
        if shape[0] in (1, 2, 3):
            return shape[1:]
        elif shape[-1] in (1, 2, 3):
            return shape[:-1]
        else:
            # Assume volumetric image (z, height, width)
            return shape
    elif len(shape) == 4:
        if shape[0] in (1, 2, 3):
            return shape[1:]
        elif shape[-1] in (1, 2, 3):
            return shape[:-1]
        else:
            return shape
    elif len(shape) == 5:
        return shape[-2:]
    else:
        raise ValueError(f"Unexpected shape: {shape}")

#-------------------------------------------------------------------
class GenericTiffHandler:
    """
    Handler for TIFF files supporting lazy loading as Dask arrays.
    Capable of processing both simple and complex TIFF formats.
    """
    def __init__(self, path=None, tiff_image_dask_array=None, channel=None):
        if path:
            self.path = path
            SIMPLER_FILETYPES = {'.tif', '.tiff', '.svs', '.ndpi'}
            COMPLEX_FILETYPES = {'.scn'}
            ext = os.path.splitext(self.path)[1].lower()

            if ext in SIMPLER_FILETYPES:
                with tifffile.TiffFile(self.path) as tiff_image:
                    tiff_image_store = tiff_image.aszarr(level=0)
                tiff_image = zarr.open(tiff_image_store, mode='r')
                self.tiff_image_dask_array = da.from_zarr(tiff_image)
                if channel is not None:
                    self.tiff_image_dask_array = self.tiff_image_dask_array[channel]
            elif ext in COMPLEX_FILETYPES:
                with tifffile.TiffFile(self.path) as tiff_image:
                    tiff_image_store = tiff_image.pages[3].aszarr(level=0)
                tiff_image = zarr.open(tiff_image_store, mode='r')
                self.tiff_image_dask_array = da.from_zarr(tiff_image)
            else:
                raise ValueError("Unsupported file type.")
        elif tiff_image_dask_array is not None:
            self.tiff_image_dask_array = tiff_image_dask_array
        else:
            raise ValueError("Either 'path' or 'tiff_image_dask_array' must be provided.")

        # Initialize metadata attributes (if available)
        self.ogMag = None
        self.ogMpp = None
        self.currentMag = None
        self.currentMpp = None

    @classmethod
    def from_dask_array(cls, tiff_image_dask_array):
        return cls(tiff_image_dask_array=tiff_image_dask_array)
    
    def get_tile_dimensions(self, tile_height, tile_width, overlap):
        """
        Calculates the number of tiles along y and x dimensions.
        """
        slide_dims = standardize_shape(np.shape(self.tiff_image_dask_array))
        if len(slide_dims) == 3:  # e.g. (z, height, width)
            _, width, height = slide_dims
        else:
            width, height = slide_dims
        
        tiles_y = int(np.ceil(height / tile_height))
        tiles_x = int(np.ceil(width / tile_width))
        return tiles_y, tiles_x
    
    def get_image_dimensions(self):
        """
        Returns standardized spatial dimensions of the image.
        """
        return standardize_shape(np.shape(self.tiff_image_dask_array))
    
    def get_thumbnail(self, thumbnail_size):
        """
        Generates a thumbnail by downsampling the image.
        """
        thumbnail = standardize_image_for_display(self.tiff_image_dask_array)[::thumbnail_size, ::thumbnail_size]
        thumbnail = thumbnail.compute()
        return Image.fromarray(thumbnail)
    
    def getCoordinatesForTile(self, pos_y, pos_x, tile_height, tile_width, overlap):        
        """
        Calculates the top-left coordinate and effective tile size for a given tile position.
        """
        image_dims = self.get_image_dimensions()
        if len(image_dims) == 3:
            # Use first slice if image is volumetric
            _, image_height, image_width = image_dims
        else:
            image_height, image_width = image_dims
        
        tiles_y, tiles_x = self.get_tile_dimensions(tile_height, tile_width, overlap)

        if pos_x < 0 or pos_x >= tiles_x:
            raise ValueError(f"Invalid x position: {pos_x}")
        if pos_y < 0 or pos_y >= tiles_y:
            raise ValueError(f"Invalid y position: {pos_y}")
        
        coord_y = (pos_y * tile_height) - overlap if pos_y > 0 else 0
        coord_x = (pos_x * tile_width) - overlap if pos_x > 0 else 0
        
        if pos_y == tiles_y - 1:
            effective_tile_height = image_height - coord_y
        else:
            effective_tile_height = tile_height + (overlap if pos_y == 0 else 2 * overlap)
        
        if pos_x == tiles_x - 1:
            effective_tile_width = image_width - coord_x
        else:
            effective_tile_width = tile_width + (overlap if pos_x == 0 else 2 * overlap)
        
        return coord_y, coord_x, effective_tile_height, effective_tile_width
    
    def get_tile(self, tile_height, tile_width, overlap, y, x, asImage=True):
        """
        Extracts a tile from the image given tile dimensions, overlap, and position.
        """
        slide = standardize_image_for_display(self.tiff_image_dask_array)
        coord_y, coord_x, eff_tile_height, eff_tile_width = self.getCoordinatesForTile(y, x, tile_height, tile_width, overlap)
        
        if len(slide.shape) == 2:
            tile = slide[coord_x:(coord_x + eff_tile_width), coord_y:(coord_y + eff_tile_height)]
        elif len(slide.shape) == 3:
            tile = slide[coord_x:(coord_x + eff_tile_width), coord_y:(coord_y + eff_tile_height), :]
        
        if asImage:
            tile = tile.compute(scheduler='threads')
            return Image.fromarray(tile)
        else:
            return tile
    
    def getNormalizedMask(self, mask):
        """
        Normalizes a mask array to the range [0, 1].
        """
        mask = np.asarray(mask)
        min_val = np.min(mask)
        max_val = np.max(mask)
        if max_val == min_val:
            return np.zeros_like(mask, dtype=np.uint8)
        return (mask - min_val) / (max_val - min_val)

    def get_original_magnification(self, filetype):
        """
        Retrieves the original magnification from image metadata based on file type.
        """
        if self.ogMag is not None:
            return self.ogMag
        else:
            if filetype == '.scn':
                metadata_str = self.tiff_image.scn_metadata
                root = ET.fromstring(metadata_str)
                objective_elements = root.findall(".//{http://www.leica-microsystems.com/scn/2010/10/01}objective")
                objectives = [np.float32(elem.text) for elem in objective_elements]
                return int(np.max(objectives))
            elif filetype == '.svs':
                metadata_str = self.tiff_image.pages[0].tags[270].value
                objective_elements = [element.split('=')[1] for element in metadata_str.split('|') if "Mag" in element]
                return int(objective_elements[0])
            elif filetype == '.ndpi':
                return int(self.tiff_image.pages[0].tags[65421].value)
            else:
                print("No metadata available for this file type.")
            return None
    
    def get_original_pixel_size(self, filetype):
        """
        Retrieves the original pixel size (MPP) from image metadata based on file type.
        """
        if self.ogMpp is not None:
            return self.ogMpp
        else:
            if filetype == '.scn':
                x_res = self.tiff_image.pages[3].tags['XResolution'].value[0]
                y_res = self.tiff_image.pages[3].tags['YResolution'].value[0]
                pixel_size = (10000 / x_res, 10000 / y_res)
                return np.unique(pixel_size)
            elif filetype == '.svs':
                metadata_str = self.tiff_image.pages[0].tags[270].value
                mpp_elements = [element.split('=')[1] for element in metadata_str.split('|') if "MPP" in element]
                return float(mpp_elements[0])
            elif filetype == '.ndpi':
                x_res = self.tiff_image.pages[0].tags['XResolution'].value[0]
                y_res = self.tiff_image.pages[0].tags['YResolution'].value[0]
                pixel_size = (10000 / x_res, 10000 / y_res)
                return np.unique(pixel_size)
            else:
                print("Could not find metadata information for this file type.")
                return None
    
    def get_current_magnification(self):
        """
        Returns the current magnification of the image.
        """
        return self.currentMag
    
    def get_current_pixel_size(self):
        """
        Returns the current pixel size (MPP) of the image.
        """
        return self.currentMpp

    def convert_between_magnification(self, target_magnification=20, method='1'):
        """
        Converts the image to a target magnification using one of two methods:
        
        - Method '1': Downsamples via slicing using a conversion factor.
        - Method '2': Uses advanced indexing to resize the image.
        """
        if target_magnification == self.currentMag or target_magnification == 0:
            return None
        if target_magnification < 0:
            raise ValueError("The target magnification must be a positive number.")
        if target_magnification > self.currentMag:
            raise ValueError("The target magnification must be less than the current magnification.")
        if target_magnification > self.ogMag:
            raise ValueError("The target magnification must be less than the original magnification.")
        
        conversion_factor = self.currentMag / target_magnification
        if method == '1':
            self.tiff_image_dask_array = self.tiff_image_dask_array[::int(conversion_factor), ::int(conversion_factor)]
            self.currentMag = target_magnification
            self.currentMpp = self.currentMpp * conversion_factor
        elif method == '2':
            new_height = int(self.tiff_image_dask_array.shape[0] / conversion_factor)
            new_width = int(self.tiff_image_dask_array.shape[1] / conversion_factor)
            row_indices = (da.arange(new_height) * conversion_factor).astype(int)
            col_indices = (da.arange(new_width) * conversion_factor).astype(int)
            resized_image = self.tiff_image_dask_array[row_indices][:, col_indices]
            self.currentMag = target_magnification
            self.currentMpp = self.currentMpp * conversion_factor
            self.tiff_image_dask_array = resized_image
        else:
            raise ValueError("Invalid method. Choose '1' or '2'.")
