import numpy as np
import os
from PIL import Image
import PIL.Image
import tifffile
import zarr
import dask.array as da
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET  # Needed for XML metadata processing
from tqdm_joblib import ParallelPbar
from joblib import delayed,Parallel
import glob
import histomicstk as htk
import pathlib
from skimage.measure import label, regionprops
import dask

import pyvips

# HistomicsTK Imports
from histomicstk.preprocessing.color_conversion import (
    rgb_to_lab, lab_mean_std, rgb_to_hsi
)
from histomicstk.preprocessing.color_deconvolution import (
    stain_color_map, rgb_separate_stains_macenko_pca, color_deconvolution, find_stain_index
)
from histomicstk.saliency.tissue_detection import (
    get_slide_thumbnail, get_tissue_mask, threshold_multichannel
)
from histomicstk.utils import simple_mask

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
    def __init__(self, path=None, image_array=None, channel=None):
        
        self.path = None
        self.image_array = None

        if path:
            self.path = path
            SIMPLER_FILETYPES = {'.tif', '.tiff', '.svs', '.ndpi'}
            COMPLEX_FILETYPES = {'.scn'}
            ext = os.path.splitext(self.path)[1].lower()

            if ext in SIMPLER_FILETYPES:
                with tifffile.TiffFile(self.path) as tiff_image:
                    self.tiff_image = tiff_image
                    tiff_image_store = tiff_image.aszarr(level=0)
                tiff_image_zarr = zarr.open(tiff_image_store, mode='r')
                self.image_array = da.from_zarr(tiff_image_zarr,chunks=(2048,2048) if len(tiff_image_zarr.shape) == 2 else (2048,2048,3))
                if channel is not None:
                    self.image_array = self.image_array[channel]
            elif ext in COMPLEX_FILETYPES:
                with tifffile.TiffFile(self.path) as tiff_image:
                    self.tiff_image = tiff_image
                    #tiff_image_store = tiff_image.pages[3].aszarr(level=0)
                    tiff_image_store = tiff_image.series[1].pages[0].aszarr(level=0)
                tiff_image_zarr = zarr.open(tiff_image_store, mode='r')
                self.image_array = da.from_zarr(tiff_image_zarr,chunks=(2048,2048) if len(tiff_image_zarr.shape) == 2 else (2048,2048,3))
            else:
                raise ValueError("Unsupported file type.")
        elif image_array is not None:
            self.image_array = image_array
        else:
            raise ValueError("Either 'path' or 'image_array' must be provided.")

        # Initialize metadata attributes (if available)
        self.ogMag = None
        self.ogMpp = None
        self.currentMag = None
        self.currentMpp = None
        
        if self.path:
            self.ogMag = self.get_original_magnification(os.path.splitext(os.path.basename(self.path))[-1],verbose=False)
            self.ogMpp = self.get_original_pixel_size(os.path.splitext(os.path.basename(self.path))[-1],verbose=False)
            self.currentMag = self.ogMag
            self.currentMpp = self.ogMpp
        
        self.tissue_mask_path = None
        if hasattr(self.image_array,'chunks'):
            self.isDaskArray = True
        else:
            self.isDaskArray = False

    @classmethod
    def from_array(cls, image_array):
        """
        Create an instance of the class from a Dask array or a NumPy array.

        Parameters:
        -----------
        image_array : dask.array.core.Array or numpy.ndarray
            An array representing the TIFF image.

        Returns:
        --------
        GenericTiffHandler
            An instance of the GenericTiffHandler class initialized with the given array.
        
        Practical Example
        --------
        GenericTiffHandler.from_array(image_array)
        
        """
        return cls(image_array=image_array)

    def get_tile_dimensions(self, tile_height, tile_width, overlap):
        """
        Calculates the number of tiles along y and x dimensions.
        """
        slide_dims = standardize_shape(np.shape(self.image_array))
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
        return standardize_shape(np.shape(self.image_array))
    
    def get_thumbnail(self, thumbnail_size):
        """
        Generates a thumbnail by downsampling the image.
        """
        thumbnail = standardize_image_for_display(self.image_array)[::thumbnail_size, ::thumbnail_size]
        if self.isDaskArray:
            thumbnail = thumbnail.compute()
        else:
            thumbnail = thumbnail

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
            effective_tile_height = image_width - coord_y
        else:
            effective_tile_height = tile_height + (overlap if pos_y == 0 else 2 * overlap)
        
        if pos_x == tiles_x - 1:
            effective_tile_width = image_height - coord_x
        else:
            effective_tile_width = tile_width + (overlap if pos_x == 0 else 2 * overlap)
        
        return coord_y, coord_x, effective_tile_height, effective_tile_width
    
    def get_tile(self, tile_height, tile_width, overlap, y, x, asImage=True):
        """
        Extracts a tile from the image given tile dimensions, overlap, and position.
        """
        slide = standardize_image_for_display(self.image_array)
        coord_y, coord_x, eff_tile_height, eff_tile_width = self.getCoordinatesForTile(y, x, tile_height, tile_width, overlap)
        
        if len(slide.shape) == 2:
                tile = slide[coord_x:(coord_x + eff_tile_width), coord_y:(coord_y + eff_tile_height)]
        elif len(slide.shape) == 3:
                tile = slide[coord_x:(coord_x + eff_tile_width), coord_y:(coord_y + eff_tile_height), :]

        if asImage:
            if self.isDaskArray:
                    tile = tile.compute() #scheduler='threading'
            else:
                tile = tile
            return Image.fromarray(tile)
        else:
            return tile
        
    def get_tile_from_coordinates(self,x1,y1,x2,y2, asImage=True):
        slide = standardize_image_for_display(self.image_array)

        if len(slide.shape) == 2:
            tile = slide[y1:y2, x1:x2]
        elif len(slide.shape) == 3:
            tile = slide[y1:y2, x1:x2, :]
        
        if asImage:
            if self.isDaskArray:
                tile = tile.compute() #scheduler='threads'
            else:
                tile = tile
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

    def get_original_magnification(self, filetype,verbose=False):
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
                if verbose:
                    print("No metadata available for this file type.")
            return None
    
    def get_original_pixel_size(self, filetype,verbose=False):
        """
        Retrieves the original pixel size (MPP) from image metadata based on file type.
        """
        if self.ogMpp is not None:
            return self.ogMpp
        else:
            if filetype == '.scn':
                x_res = self.tiff_image.series[1].pages[0].tags['XResolution'].value[0]
                y_res = self.tiff_image.series[1].pages[0].tags['YResolution'].value[0]
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
                if verbose:
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
            self.image_array = self.image_array[::int(conversion_factor), ::int(conversion_factor)]
            self.currentMag = target_magnification
            self.currentMpp = self.currentMpp * conversion_factor
        elif method == '2':
            new_height = int(self.image_array.shape[0] / conversion_factor)
            new_width = int(self.image_array.shape[1] / conversion_factor)
            if self.is_dask_array():
                row_indices = (da.arange(new_height) * conversion_factor).astype(int)
                col_indices = (da.arange(new_width) * conversion_factor).astype(int)
            else:
                row_indices = (np.arange(new_height) * conversion_factor).astype(int)
                col_indices = (np.arange(new_width) * conversion_factor).astype(int)
            resized_image = self.image_array[row_indices][:, col_indices]
            self.currentMag = target_magnification
            self.currentMpp = self.currentMpp * conversion_factor
            self.image_array = resized_image
        else:
            raise ValueError("Invalid method. Choose '1' or '2'.")
    
    def calculate_useful_tiles(self,tile_height,tile_width,overlap,
                               tissue_mask_path=None, 
                               tissue_percentage_threshold=25, cpu_workers=12, mode='newer',grid_step=4):
        
        assert mode in ['naive','faster','newer']
        
        def get_tiles_in_bbox(
            bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax,
            tile_height, tile_width, overlap
        ):
            """
            Returns a list of (pos_y, pos_x) tile indices that intersect with the given bounding box.
            
            The bounding box is given in the same coordinate space as your entire image, i.e.
            (bbox_ymin, bbox_xmin, bbox_ymax, bbox_xmax).
            """
            # 1) Determine how many tiles exist in Y and X
            tiles_y, tiles_x = MaskObject.get_tile_dimensions(tile_height, tile_width, overlap)
            
            # 2) Prepare a container for all tile positions that intersect with the bbox
            intersecting_tiles = []
            
            # 3) Iterate over all possible tile positions
            for pos_y in range(tiles_y):
                for pos_x in range(tiles_x):
                    # Use your existing function to get the tile's bounding region
                    tile_y, tile_x, eff_tile_h, eff_tile_w = MaskObject.getCoordinatesForTile(
                        pos_y, pos_x, tile_height, tile_width, overlap
                    )
                    # Compute tile's bounding box
                    tile_ymax = tile_y + eff_tile_h
                    tile_xmax = tile_x + eff_tile_w
                    
                    # 4) Check if the tile bounding box overlaps the given bbox.
                    # One common overlap check is:
                    #   overlap exists if (not) ( tile is completely left OR above OR right OR below the bbox )
                    # i.e. overlap if tile_y < bbox_ymax and tile_ymax > bbox_ymin etc.
                    
                    if not (
                        tile_ymax <= bbox_ymin  # tile is completely above
                        or tile_y >= bbox_ymax  # tile is completely below
                        or tile_xmax <= bbox_xmin  # tile is completely left
                        or tile_x >= bbox_xmax     # tile is completely right
                    ):
                        # We have some overlap
                        intersecting_tiles.append((pos_y, pos_x))
            
            return intersecting_tiles
        
        def process_tile(tile_params):
            col, row, tissue_mask_path, tissue_masks, tile_height, tile_width, overlap, tissue_percentage_threshold,ogMag,ogMpp,currentMag,currentMpp = tile_params
            if tissue_mask_path is None:
                tile_tissue = tissue_masks[(col, row)]
            else:
                tissue_mask_obj = GenericTiffHandler(tissue_mask_path)
                tissue_mask_obj.set_magnification_settings(mag=ogMag,mpp=ogMpp)
                tissue_mask_obj.convert_between_magnification(currentMag, method='1')
                tile_tissue = np.asarray(tissue_mask_obj.get_tile(tile_height, tile_width, overlap, col, row))
                # Debug: check if the tile is empty and log the details
                if tile_tissue.size == 0:
                    print(f"DEBUG: Empty tile from tissue mask file at col {col}, row {row}.")
                    print(f"       Requested tile size: ({tile_height}, {tile_width}), overlap: {overlap}")
                    print(f"       Tissue mask path: {tissue_mask_path}")
                if np.max(tile_tissue) == 255:
                    tile_tissue = tile_tissue / 255

            tissue_percentage = (np.sum(tile_tissue) / (tile_tissue.shape[0] * tile_tissue.shape[1])) * 100
            return (col, row) if tissue_percentage > tissue_percentage_threshold else None
        
        def __get_tissue_mask__(tile):
            """ Return a binary mask of the tissue in the tile."""
            tile = np.asarray(tile)
            
            background_mask, _ = threshold_multichannel(rgb_to_hsi(tile), {
                'hue': {'min': 0, 'max': 1.0},
                'saturation': {'min': 0, 'max': 0.2},
                'intensity': {'min': 220, 'max': 255},
            }, just_threshold=True)
            
            background_mask = np.asarray(background_mask, dtype=np.uint8)
            # Invert the mask to get the tissue area
            tissue_mask = (1-background_mask).astype(np.uint8)
            # Ensure the mask is binary
            return tissue_mask

        assert self.path is not None, "The path to the image must be provided for this operation."

        dataset_path = os.path.dirname(os.path.dirname(self.path))

        if tissue_mask_path is None:
            self.tissue_mask_path = glob.glob(os.path.join(dataset_path, 'Tissue Masks', 
                                                         os.path.splitext(os.path.basename(self.path))[0] + '*.tiff'))
            if len(self.tissue_mask_path) == 0:
                self.tissue_mask_path = None
            else:
                self.tissue_mask_path = self.tissue_mask_path[0]
        else:
            self.tissue_mask_path = tissue_mask_path

        Tiles_y, Tiles_x = self.get_tile_dimensions(tile_height, tile_width, overlap)
          
        tissue_masks = {}
        if self.tissue_mask_path is None:
            mode = 'naive'  # Default to naive mode if no tissue mask is provided
            # Process each tile individually to avoid caching all tiles in memory
            if mode == 'naive':
                # Generate parameters for each tile
                tile_params = [
                    (col, row, tile_height, tile_width, overlap, tissue_percentage_threshold, self.path)
                    for col in range(Tiles_y)
                    for row in range(Tiles_x)
                ]
                
                # Define a new process_tile function to create mask on demand
                def process_tile_with_mask(tile_params):
                    col, row, tile_height, tile_width, overlap, tissue_percentage_threshold, path = tile_params
                    # Get tile directly and process it
                    tile = GenericTiffHandler(path).get_tile(tile_height, tile_width, overlap, col, row, asImage=True)
                    # Convert tile to numpy array and get tissue mask
                    tile_tissue = __get_tissue_mask__(tile)
                    tissue_percentage = (np.sum(tile_tissue) / (tile_tissue.shape[0] * tile_tissue.shape[1])) * 100
                    return (col, row) if tissue_percentage > tissue_percentage_threshold else None
                
                results = ParallelPbar("Calculating useful tiles...")(n_jobs=cpu_workers, backend='loky')(
                    delayed(process_tile_with_mask)(params) for params in tile_params
                )
                return [tile for tile in results if tile is not None ]

        elif self.tissue_mask_path is not None:
            if mode=='naive':
                tile_params = [
                    (col, row, self.tissue_mask_path, tissue_masks, tile_height, tile_width, overlap, tissue_percentage_threshold,self.ogMag,self.ogMpp,self.currentMag,self.currentMpp)
                    for col in range(Tiles_y)
                    for row in range(Tiles_x)
                ]
                
                results = ParallelPbar("Calculating useful tiles...")(n_jobs=cpu_workers, backend='loky')(
                    delayed(process_tile)(params) for params in tile_params
                )
                return [tile for tile in results if tile is not None]
            elif mode == 'faster':
                ### 🔹 Step 2: Coarse Grid Search (Initial Fast Scan)
                initial_tiles = [
                    (col, row, self.tissue_mask_path, tissue_masks, tile_height, tile_width, overlap, tissue_percentage_threshold,self.ogMag,self.ogMpp,self.currentMag,self.currentMpp)
                    for col in range(0, Tiles_y, grid_step)
                    for row in range(0, Tiles_x, grid_step)
                ]
                
                coarse_results = ParallelPbar("Selecting coarse results...")(n_jobs=cpu_workers, backend='loky')(
                    delayed(process_tile)(params) for params in initial_tiles
                )
                
                # Remove None values and keep only valid tiles
                refined_candidates = [tile for tile in coarse_results if tile is not None]
                
                ### 🔹 Step 3: Expand Search Around Found Tiles
                candidate_tiles = set()
                for col, row in refined_candidates:
                    for dx in range(-grid_step + 1, grid_step):
                        for dy in range(-grid_step + 1, grid_step):
                            new_col, new_row = col + dx, row + dy
                            if 0 <= new_col < Tiles_y and 0 <= new_row < Tiles_x:
                                candidate_tiles.add((new_col, new_row))
                
                ### 🔹 Step 4: Run Parallel Processing for Refinement
                final_tiles = ParallelPbar("Creating refined selection...")(n_jobs=cpu_workers, backend='loky')(
                    delayed(process_tile)((col, row, self.tissue_mask_path,tissue_masks, tile_height, tile_width, overlap, tissue_percentage_threshold,self.ogMag,self.ogMpp,self.currentMag,self.currentMpp))
                    for col, row in candidate_tiles
                )

                # Remove None values and return the final tile selection
                return [tile for tile in final_tiles if tile is not None]
            elif mode == 'newer':
                if self.path is not None:

                    MaskObject = GenericTiffHandler(self.tissue_mask_path)
                    if MaskObject.get_original_pixel_size(os.path.splitext(os.path.basename(tissue_mask_path))[-1]) is None:
                        MaskObject.set_magnification_settings(self.get_original_magnification(
                            os.path.splitext(os.path.basename(tissue_mask_path))[-1]
                        ), self.get_original_pixel_size(
                            os.path.splitext(os.path.basename(tissue_mask_path))[-1]
                        ))
                    MaskObject.convert_between_magnification(self.currentMag)
                    downsampling_factor = 20
                    mask_downsampled = MaskObject.get_thumbnail(downsampling_factor)

                    mask_label = label(np.asarray(mask_downsampled))
                    regions = regionprops(mask_label)

                    bboxes =  [region.bbox for region in regions]
                    bboxes_original = [(bbox[0]*downsampling_factor-100, bbox[1]*downsampling_factor-100, bbox[2]*downsampling_factor+100, bbox[3]*downsampling_factor+100) for bbox in bboxes]
                    bboxes_origal_reshaped = [(bbox[1], bbox[0], bbox[3], bbox[2]) for bbox in bboxes_original]

                    # Get the tiles that are included in the bounding boxes
                    
                    candidate_tiles = []
                    for bbox in bboxes_origal_reshaped:
                        candidate_tiles.append(get_tiles_in_bbox(*bbox, tile_height, tile_width, overlap))
                    candidate_tiles = [item for sublist in candidate_tiles for item in sublist]
                    candidate_tiles = list(set(candidate_tiles))

                    # Run parallel processing for refinement
                    final_tiles = ParallelPbar("Creating refined selection...")(n_jobs=cpu_workers, backend='loky')(
                        delayed(process_tile)((col, row, self.tissue_mask_path,tissue_masks, tile_height, tile_width, overlap, tissue_percentage_threshold,self.ogMag,self.ogMpp,self.currentMag,self.currentMpp))
                        for col, row in candidate_tiles
                    )

                    return [tile for tile in final_tiles if tile is not None]

                else:
                    self.calculate_useful_tiles(tile_height, tile_width, overlap, tissue_mask_path, tissue_percentage_threshold, cpu_workers, mode='naive')
            
            else:
                raise ValueError("Tissue mask path not found/provided.")

    def get_current_path(self):
        return self.path
    
    def is_dask_array(self):
        return self.isDaskArray
    
    def set_magnification_settings(self, mag, mpp):
        self.ogMag = mag
        self.ogMpp = mpp
        self.currentMag = mag
        self.currentMpp = mpp
    
    def save_to_tiff_with_metadata(self, saving_path=None):
        import pyvips  # Local import to avoid global dependency issues
            
        if saving_path is None:
            raise ValueError("A saving_path must be provided.")
            
        # Create a pyvips image based on the available data:
        if self.path is not None:
            # Read via pyvips from the file path
            pyvipsImage = pyvips.Image.new_from_file(self.path, access="sequential")
        else:
            # Use the image_array. Only 2D images (binary) are supported.
            if self.isDaskArray:
                if len(self.image_array.shape) != 2:
                    raise ValueError("Only 2D dask array images are supported for saving.")
                array = self.image_array.compute()
            else:
                if len(self.image_array.shape) != 2:
                    raise ValueError("Only 2D images are supported for saving.")
                array = self.image_array
            
        # Convert the numpy array to a pyvips image.
        height, width = array.shape
        bands = 1  # For binary/2D image, only one band is present
        fmt = 'uchar'
        array_bytes = array.tobytes()
        pyvipsImage = pyvips.Image.new_from_memory(array_bytes, width, height, bands, fmt)
        
        # Copy image to add metadata
        pyvipsImageTemp = pyvipsImage.copy()
        image_height = pyvipsImageTemp.height
        image_width = pyvipsImageTemp.width
        image_bands = pyvipsImageTemp.bands
            
        # Construct XML metadata
        xml_str = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2016-06 http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd">
    <Image ID="Image:0">
        <Pixels DimensionOrder="XYCZT"
                ID="Pixels:0"
                SizeC="{image_bands}"
                SizeT="1"
                SizeX="{image_width}"
                SizeY="{image_height}"
                SizeZ="1"
                Type="uint8">
        </Pixels>
    </Image>
</OME>"""
            
        # Set the metadata parameters
        pyvipsImageTemp.set_type(pyvips.GValue.gint_type, "page-height", image_height)
        pyvipsImageTemp.set_type(pyvips.GValue.gstr_type, "image-description", xml_str)
        
        # Save the image as a TIFF with metadata, tiling, and pyramid structure.
        pyvipsImageTemp.tiffsave(saving_path, compression="jpeg", tile=True,
                                tile_width=512, tile_height=512, Q=100,
                                pyramid=True, subifd=True)
        
        return True