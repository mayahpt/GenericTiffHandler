import json
import numpy as np
#%matplotlib inline
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# The path can also be read from a config file, etc.
import os
OPENSLIDE_PATH = os.path.abspath(r".\Utils\OpenSlide\openslide-bin-4.0.0.3-windows-x64\bin")


if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
        from openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import glob
import shutil
VIPS_BIN_PATH = os.path.abspath(r".\Utils\VIPS\vips-dev-8.15\bin")
add_dll_dir = getattr(os, 'add_dll_directory', None)
if callable(add_dll_dir):
    add_dll_dir(VIPS_BIN_PATH)
else:
    os.environ['PATH'] = os.pathsep.join((VIPS_BIN_PATH, os.environ['PATH']))

import pyvips
import tqdm
import PIL.Image
import tifffile
import zarr
import dask.array as da
import matplotlib.pyplot as plt

# Increase the maximum number of pixels that can be processed by PIL
PIL.Image.MAX_IMAGE_PIXELS = None 

def standardize_image_for_display(image):
    shape = image.shape
    ndim = len(shape)

    if ndim == 2:
        return image
    elif ndim == 3:
        if shape[0] == 3:
            return np.transpose(image, (1, 2, 0))
        return image
    elif ndim == 5:
        return np.transpose(np.squeeze(image, axis=(0, 1)), (1, 2, 0))
    else:
        raise ValueError(f"Unexpected shape: {shape}")

def standardize_shape(shape):
    # Check if the shape is 2D (grayscale image)
    if len(shape) == 2:
        return shape
    
    # Check if the shape is 3D (channels first or last)
    if len(shape) == 3:
        if shape[0] <= 3:  # channels first
            return shape[1:]
        else:  # channels last
            return shape[:-1]
        
    # Check if the shape is 5D (extra dimensions for some reason)
    elif len(shape) == 5:
        return shape[-2:]
    
    # Add more conditions if necessary for other potential shapes
    else:
        raise ValueError(f"Unexpected shape: {shape}")


class GenericTiffHandler:
    def __init__(self, path=None, tiff_image_dask_array=None, channel=None):
        
        if path:
            self.path = path

            # What kind of file is it?
            SIMPLER_FILETYPES = {'.tif', '.tiff', '.svs','.ndpi'}
            COMPLEX_FILETYPES = {'.scn'}

            if os.path.splitext(self.path)[1].lower() in SIMPLER_FILETYPES:
                # Load Zarr array lazily as a Dask array
                with tifffile.TiffFile(self.path) as tiff_image:
                    tiff_image_store = tiff_image.aszarr(level=0)
                tiff_image = zarr.open(tiff_image_store, mode='r')
                self.tiff_image_dask_array = da.from_zarr(tiff_image)
                # Check if the image has multiple channels
                if channel is not None:
                    self.tiff_image_dask_array = self.tiff_image_dask_array[channel]
                else:
                    self.tiff_image_dask_array = self.tiff_image_dask_array
            
            elif os.path.splitext(self.path)[1].lower() in COMPLEX_FILETYPES:
                with tifffile.TiffFile(self.path) as tiff_image:
                    tiff_image_store = tiff_image.pages[3].aszarr(level=0)
                tiff_image = zarr.open(tiff_image_store, mode='r')
                self.tiff_image_dask_array = da.from_zarr(tiff_image)
        
        elif tiff_image_dask_array is not None:
            self.tiff_image_dask_array = tiff_image_dask_array
        else:
            raise ValueError("Either 'path' or 'tiff_image_dask_array' must be provided.")

    @classmethod
    def from_dask_array(cls, tiff_image_dask_array):
        return cls(tiff_image_dask_array=tiff_image_dask_array)
    
    def get_tile_dimensions(self, tile_height, tile_width, overlap):
        slide_dimensions = standardize_shape(np.shape(self.tiff_image_dask_array))
        width, height = slide_dimensions
        
        tiles_y = int(np.ceil(height / tile_height))
        tiles_x = int(np.ceil(width / tile_width))
        
        return tiles_y, tiles_x
    
    def get_image_dimensions(self):
        return standardize_shape(np.shape(self.tiff_image_dask_array))
    
    def get_thumbnail(self, thumbnail_size):
        thumbnail = standardize_image_for_display(self.tiff_image_dask_array)[::thumbnail_size, ::thumbnail_size]
        thumbnail = thumbnail.compute()
        # Convert the NumPy array to a PIL Image
        thumbnail_Image = Image.fromarray(thumbnail)
        return thumbnail_Image
    
    def getCoordinatesForTile(self, pos_y, pos_x, tile_height, tile_width, overlap):        
        image_height, image_width = self.get_image_dimensions()        
        tiles_y, tiles_x = self.get_tile_dimensions(tile_height, tile_width, overlap)

        # Check if the positions are within the image bounds
        # or pos_x * (tile_width - overlap) >= image_width
        # or pos_y * (tile_height - overlap) >= image_height
        if pos_x < 0  or pos_x >= tiles_x:
            raise ValueError(f"Invalid x position: {pos_x}")
        if pos_y < 0  or pos_y >= tiles_y:
            raise ValueError(f"Invalid y position: {pos_y}")
        
        # Calculate the coordinates of the top-left corner of the tile
        coord_y = (pos_y * tile_height) - overlap if pos_y > 0 else 0
        coord_x = (pos_x * tile_width) - overlap if pos_x > 0 else 0
        
        # Calculate the effective tile size considering the borders and image boundaries
        if pos_y == tiles_y - 1:
            effective_tile_height = image_width - coord_y
        else:
            effective_tile_height = tile_height + (overlap if pos_y == 0 else 2 * overlap)
        
        if pos_x == tiles_x - 1:
            effective_tile_width = image_height - coord_x
        else:
            effective_tile_width = tile_width + (overlap if pos_x == 0 else 2 * overlap)
        
        return coord_y, coord_x, effective_tile_height, effective_tile_width
    
    def get_tile(self,tile_height, tile_width,overlap,y, x,asImage=True):
        slide = standardize_image_for_display(self.tiff_image_dask_array)
        coord_y, coord_x, effective_tile_height, effective_tile_width = self.getCoordinatesForTile(y, x, tile_height, tile_width, overlap)
        # Check the shape of the slide array
        if len(slide.shape) == 2:
            # Grayscale image
            tile = slide[coord_x:(coord_x+effective_tile_width),coord_y:(coord_y+effective_tile_height)]
        elif len(slide.shape) == 3:
            # RGB image
            tile = slide[coord_x:(coord_x+effective_tile_width),coord_y:(coord_y+effective_tile_height),:]
        
        if asImage:
            tile = tile.compute(scheduler='threads')
            tile_Image = Image.fromarray(tile)
            return tile_Image
        else:
            return tile
    
    def getNormalizedMask(self,mask):
        mask = np.asarray(mask)
        min_val = np.min(mask)
        max_val = np.max(mask)
    
        # Check if the range is zero to avoid division by zero
        if max_val == min_val:
            mask = np.zeros_like(mask, dtype=np.uint8)
        else:
            mask = (mask - min_val) / (max_val - min_val)
        return mask

    
