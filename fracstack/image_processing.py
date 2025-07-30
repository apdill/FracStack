import numpy as np
import skimage.io as io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import os


def invert_array(arr):
    """
    Invert binary or grayscale arrays with automatic format detection.
    
    This function automatically detects the array format and applies appropriate
    inversion. For binary arrays (0,1), it flips values. For uint8 arrays (0-255),
    it subtracts from 255 to create a negative image effect.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array to invert. Supported formats:
        - Binary arrays with values in [0, 1]
        - Grayscale arrays with values in [0, 255]
        
    Returns
    -------
    np.ndarray
        Inverted array of the same dtype and shape as input:
        - Binary: 0 becomes 1, 1 becomes 0
        - Grayscale: pixel_value becomes (255 - pixel_value)
        
    Raises
    ------
    ValueError
        If array values exceed the expected range for binary or uint8 formats
        
    Examples
    --------
    >>> import numpy as np
    >>> # Binary array inversion
    >>> binary_arr = np.array([[0, 1], [1, 0]])
    >>> inverted = invert_array(binary_arr)
    >>> print(inverted)
    [[1 0]
     [0 1]]
    
    >>> # Grayscale array inversion
    >>> gray_arr = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    >>> inverted = invert_array(gray_arr)
    >>> print(inverted)
    [[255 127]
     [  0 191]]
    
    Notes
    -----
    The function uses the maximum value in the array to determine the format:
    - max_value ≤ 1: Treated as binary
    - max_value ≤ 255: Treated as uint8 grayscale
    - max_value > 255: Raises ValueError
    
    This automatic detection works well for typical image processing workflows
    but may not be suitable for arrays with unusual value ranges.
    """
    if arr.max() <= 1:
        # Handle binary arrays (0,1) by flipping 0 to 1 and 1 to 0
        return np.where(arr == 0, 1, 0)
    elif arr.max() <= 255:
        # Handle uint8 arrays (0 to 255) by subtracting from 255
        return 255 - arr
    else:
        raise ValueError("Array values exceed expected range for binary or uint8 format.")

def process_image_to_array(file_path, threshold=None, invert = False):
    """
    Load an image file and convert it to a binary array with thresholding.
    
    This function handles the complete workflow of loading an image, converting
    to grayscale if needed, applying thresholding to create a binary image,
    and optionally inverting the result. It supports automatic threshold
    selection and various threshold strategies.
    
    Parameters
    ----------
    file_path : str
        Path to the image file to load. Supports formats compatible with
        skimage.io.imread (PNG, TIFF, JPEG, etc.)
    threshold : int, float, str, or None, default None
        Threshold value or strategy for binarization:
        - None: Use image mean as threshold (automatic)
        - int/float: Specific threshold value
        - 'min': Use minimum pixel value as threshold
        - 'max': Use maximum pixel value as threshold
    invert : bool, default False
        Whether to invert the binary image after thresholding
        
    Returns
    -------
    np.ndarray
        Binary image array with dtype int and values in {0, 1}
        
    Raises
    ------
    ValueError
        If the specified threshold is outside the range of image values
        
    Examples
    --------
    >>> # Basic usage with automatic threshold
    >>> binary_img = process_image_to_array('image.png')
    >>> 
    >>> # Custom threshold
    >>> binary_img = process_image_to_array('image.png', threshold=128)
    >>> 
    >>> # Inverted binary image
    >>> binary_img = process_image_to_array('image.png', threshold=100, invert=True)
    >>> 
    >>> # Use minimum value as threshold
    >>> binary_img = process_image_to_array('image.png', threshold='min')
    
    Notes
    -----
    Processing Steps:
    1. Load image using skimage.io.imread
    2. Convert to grayscale by averaging RGB channels if needed
    3. Apply threshold to create binary image (>=threshold becomes 1)
    4. Optionally invert the binary result
    
    The function automatically handles:
    - Color to grayscale conversion
    - Automatic threshold selection
    - Threshold validation
    - Binary conversion with proper dtype
    
    Threshold strategies:
    - Automatic (None): Uses image mean, good for balanced images
    - Minimum: Creates mostly white images (most pixels become 1)
    - Maximum: Creates mostly black images (most pixels become 0)
    - Custom value: Allows precise control over binarization
    """
    image_array = io.imread(file_path)
    
    # Convert to grayscale by averaging channels if it's not already
    if len(image_array.shape) == 3:
        image_array = image_array.mean(axis=2)
    
    # Binarize the image with the given threshold
    if threshold is None:
        threshold = image_array.mean()
    elif threshold == 'min':
        threshold = image_array.min()
    elif threshold == 'max':
        threshold = image_array.max()
    else:  
        if threshold < image_array.min() or threshold > image_array.max():
            raise ValueError(f"Threshold {threshold} is out of the range of image values ({image_array.min()} to {image_array.max()}).")

    binary_image_array = (image_array >= threshold).astype(int)

    # Optionally invert the binary image
    if invert:
        binary_image_array = invert_array(binary_image_array)

    return binary_image_array

def pad_image_for_boxcounting(input_array, max_size, pad_factor = 1, manual_pad = 0, invert = False):
    """
    Pad binary image symmetrically to mitigate edge effects during box counting analysis.
    
    This function pads the input image to ensure that box counting grids can be properly
    aligned and that edge effects are minimized. The padding is calculated based on the
    maximum box size and applied symmetrically around the image.
    
    Parameters
    ----------
    input_array : np.ndarray
        2D binary input array to be padded
    max_size : int
        Maximum box size that will be used for box counting. Used to calculate
        appropriate padding dimensions.
    pad_factor : float, default 1
        Padding factor that multiplies the calculated padding size. Values > 1
        provide more padding for better edge effect mitigation.
    manual_pad : int, default 0
        Additional manual padding in pixels to add beyond the calculated padding
    invert : bool, default False
        Whether to invert the input array before padding
        
    Returns
    -------
    np.ndarray
        Padded binary array with dtype bool. The padding uses constant values of 0
        (background) to avoid introducing artificial structure.
        
    Notes
    -----
    Padding Calculation:
    The function calculates new dimensions as:
    - new_height = ceil(height / max_size) * max_size * pad_factor + manual_pad
    - new_width = ceil(width / max_size) * max_size * pad_factor + manual_pad
    
    This ensures that the padded dimensions are multiples of max_size scaled by
    pad_factor, which helps with grid alignment during box counting.
    
    Edge Effect Mitigation:
    Proper padding is crucial for accurate fractal dimension estimation because:
    - Reduces bias from incomplete boxes at image boundaries
    - Ensures consistent sampling across different grid alignments
    - Prevents artificial scaling artifacts near edges
    
    The padding uses constant values of 0 (background) to maintain the fractal
    structure without introducing artificial patterns.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Basic padding for box counting
    >>> binary_img = np.random.choice([0, 1], size=(100, 100), p=[0.8, 0.2])
    >>> padded = pad_image_for_boxcounting(binary_img, max_size=32)
    >>> print(f"Original: {binary_img.shape}, Padded: {padded.shape}")
    >>> 
    >>> # Extra padding for better edge effect mitigation
    >>> padded = pad_image_for_boxcounting(binary_img, max_size=32, pad_factor=1.5)
    >>> 
    >>> # Manual additional padding
    >>> padded = pad_image_for_boxcounting(binary_img, max_size=32, manual_pad=10)
    
    See Also
    --------
    boxcount : Box counting function that benefits from proper padding
    measure_dimension : High-level function that uses this for edge effect mitigation
    """

    if invert == True:
        input_array = invert_array(input_array)

    height, width = input_array.shape

    # Calculate new dimensions
    new_height = int(np.ceil(height / max_size) *  max_size * pad_factor + manual_pad)
    new_width = int(np.ceil(width / max_size) *  max_size * pad_factor + manual_pad)

    # Calculate padding on each side
    pad_top = (new_height - height) // 2
    pad_bottom = new_height - height - pad_top
    pad_left = (new_width - width) // 2
    pad_right = new_width - width - pad_left

    # Apply padding
    padded_array = np.pad(input_array, 
                          pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
                          mode='constant', 
                          constant_values=0)

    return padded_array.astype(bool)

def bounding_box_diameter(region):
    """
    Calculate the diameter of a region based on its bounding box.
    
    This function computes the diameter as the maximum dimension (height or width)
    of the region's bounding box. This provides a fast approximation of object
    size that's useful for quick size characterization.
    
    Parameters
    ----------
    region : skimage.measure.RegionProperties
        Region properties object from skimage.measure.regionprops containing
        the bounding box information in the .bbox attribute
        
    Returns
    -------
    float
        Diameter of the region as the maximum of bounding box height and width
        
    Notes
    -----
    This function provides a computationally efficient approximation of object
    diameter by using the bounding box dimensions rather than computing the
    actual maximum pairwise distance between pixels.
    
    The bounding box diameter is:
    - Fast to compute (O(1) operation)
    - Always >= actual maximum diameter for convex objects
    - May overestimate diameter for non-convex objects
    - Suitable for quick size screening and filtering
    
    For more accurate diameter calculations, use the calculate_real_diameter
    function in find_largest_smallest_objects, which computes true pairwise
    distances between pixels.
    
    Examples
    --------
    >>> from skimage import measure
    >>> import numpy as np
    >>> # Create a simple binary image
    >>> img = np.zeros((50, 50))
    >>> img[20:30, 20:40] = 1  # Rectangle
    >>> regions = measure.regionprops(measure.label(img))
    >>> diameter = bounding_box_diameter(regions[0])
    >>> print(f"Bounding box diameter: {diameter}")  # Should be 20 (max of 10, 20)
    
    See Also
    --------
    find_largest_smallest_objects : Function that computes true pairwise distances
    skimage.measure.regionprops : Source of region properties objects
    """
    # Calculate the height and width of the bounding box
    min_row, min_col, max_row, max_col = region.bbox
    height = max_row - min_row
    width = max_col - min_col
    # Return the larger of the two as the diameter
    return max(height, width)



def find_largest_smallest_objects(binary_image, invert=False):
    """
    Identify and analyze the largest and smallest objects in a binary image.
    
    This function performs connected component analysis to identify objects in a binary
    image, then finds the largest and smallest objects based on area. It computes
    accurate diameters using pairwise distance calculations with optimizations for
    computational efficiency.
    
    Parameters
    ----------
    binary_image : np.ndarray
        2D binary image array where objects are represented as 1s and background as 0s
    invert : bool, default False
        Whether to invert the binary image before analysis
        
    Returns
    -------
    tuple
        (largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image)
        
        - largest_object : skimage.measure.RegionProperties or None
            Region properties of the largest object by area
        - smallest_object : skimage.measure.RegionProperties or None  
            Region properties of the smallest object by area
        - largest_diameter : float or None
            True diameter of largest object (maximum pairwise distance)
        - smallest_diameter : float or None
            True diameter of smallest object (maximum pairwise distance)
        - labeled_image : np.ndarray
            Labeled image from connected component analysis
            
    Notes
    -----
    Object Detection and Filtering:
    - Uses 8-connectivity for connected component analysis
    - Filters out objects smaller than 2x2 pixels (bounding box)
    - Objects are sorted by area to find largest and smallest
    
    Diameter Calculation Algorithm:
    The function uses an optimized approach for diameter calculation:
    
    1. **Small Objects** (< 2 pixels): Diameter = 0
    2. **Medium Objects** (2-1000 pixels): Direct pairwise distance calculation
    3. **Large Objects** (> 1000 pixels): Convex hull optimization
    4. **Very Large Objects** (> 5000 pixels): Random sampling for efficiency
    
    Performance Optimizations:
    - Convex hull reduces computation for large objects
    - Random sampling caps computation for very large objects
    - Validation against image diagonal prevents unrealistic results
    
    The true diameter is computed as the maximum Euclidean distance between
    any two pixels in the object, providing accurate size measurements for
    irregular shapes.
    
    Examples
    --------
    >>> import numpy as np
    >>> from skimage import measure
    >>> 
    >>> # Create a binary image with objects
    >>> img = np.zeros((100, 100))
    >>> img[10:20, 10:30] = 1  # Rectangle
    >>> img[50:55, 50:55] = 1  # Small square
    >>> 
    >>> largest, smallest, large_d, small_d, labeled = find_largest_smallest_objects(img)
    >>> print(f"Largest object area: {largest.area}")
    >>> print(f"Largest diameter: {large_d}")
    >>> print(f"Smallest object area: {smallest.area}")
    >>> print(f"Smallest diameter: {small_d}")
    
    See Also
    --------
    bounding_box_diameter : Fast diameter approximation using bounding box
    skimage.measure.label : Connected component labeling
    skimage.measure.regionprops : Region property analysis
    """
    if invert:
        binary_image = invert_array(binary_image)

    # Label connected regions in the binary image
    labeled_image, num_features = measure.label(binary_image, connectivity=2, return_num=True)

    if num_features == 0:
        print("No objects found.")
        return None, None, None, None, labeled_image

    # Get region properties for each labeled object
    regions = measure.regionprops(labeled_image)

    def calculate_real_diameter(region):
        coords = region.coords  # Pixel coordinates of the object (already in global space)

        if len(coords) < 2:  # If the object has fewer than 2 pixels
            return 0  # Diameter is zero

        # Use convex hull for large objects
        if len(coords) > 1000:
            hull = ConvexHull(coords)
            coords = coords[hull.vertices]

        # Cap the number of points for pairwise calculation
        max_points = 5000
        if len(coords) > max_points:
            coords = coords[np.random.choice(len(coords), max_points, replace=False)]

        # Compute all pairwise distances and return the maximum
        diameter = pdist(coords).max()

        # Validate the diameter against the image size
        image_height, image_width = binary_image.shape
        max_possible_diameter = np.sqrt(image_height**2 + image_width**2)
        if diameter > max_possible_diameter:
            print(f"Warning: Calculated diameter ({diameter}) exceeds image diagonal ({max_possible_diameter}).")
            diameter = max_possible_diameter

        return diameter

    # Filter out regions with bounding box smaller than 2x2
    valid_regions = [r for r in regions if (r.bbox[2] - r.bbox[0] >= 2 and r.bbox[3] - r.bbox[1] >= 2)]

    if not valid_regions:
        print("No valid objects found with a minimum size of 2x2.")
        return None, None, None, None, labeled_image

    # Find the largest object
    largest_object = max(valid_regions, key=lambda r: r.area)
    largest_diameter = np.round(calculate_real_diameter(largest_object))

    # Find the smallest object
    smallest_object = None
    smallest_diameter = None

    for region in sorted(valid_regions, key=lambda r: r.area):
        if region.area > 0:  # Ensure the object has pixels
            smallest_object = region
            smallest_diameter = np.round(calculate_real_diameter(region))
            break  # Stop once the first valid smallest object is found

    return largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image

def find_largest_empty_spaces(binary_array, n, plot=False, save=False, save_path=None):
    """
    Find the n largest empty square regions in a binary image using dynamic programming.
    
    This function identifies the largest empty (background) square regions in a binary
    image using an efficient dynamic programming algorithm. It can find multiple
    non-overlapping squares and provides visualization of the results.
    
    Parameters
    ----------
    binary_array : np.ndarray
        2D binary image array where 0 represents empty space and 1 represents objects
    n : int
        Number of largest empty squares to find
    plot : bool, default False
        Whether to create visualizations showing the found squares
    save : bool, default False
        Whether to save the visualization plot to file
    save_path : str, optional
        Directory path for saving the plot. Required if save=True.
        
    Returns
    -------
    list
        List of square widths (in pixels) for the n largest empty squares,
        sorted from largest to smallest
        
    Notes
    -----
    Algorithm Details:
    
    The function uses a dynamic programming approach to efficiently find the largest
    empty squares:
    
    1. **Dynamic Programming**: Uses a 1D array to track the maximum square size
       ending at each position, updated row by row
    2. **Iterative Search**: Finds the largest square, masks it, then repeats
       for the next largest
    3. **Non-overlapping**: Ensures found squares don't overlap by masking
       previously found regions
    
    Time Complexity: O(n * rows * cols) where n is the number of squares to find
    Space Complexity: O(cols) for the dynamic programming array
    
    The algorithm is particularly useful for:
    - Analyzing void spaces in porous materials
    - Finding placement locations for objects
    - Characterizing spatial distribution of empty regions
    - Quality control in manufacturing processes
    
    Visualization Features:
    - Overlays green rectangles on the original image
    - Numbers each square by size ranking
    - Shows size distribution plot
    - Saves results if requested
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a binary image with some empty spaces
    >>> img = np.random.choice([0, 1], size=(100, 100), p=[0.7, 0.3])
    >>> 
    >>> # Find 5 largest empty squares
    >>> square_sizes = find_largest_empty_spaces(img, n=5)
    >>> print(f"Largest empty square size: {square_sizes[0]}")
    >>> 
    >>> # Find and visualize empty spaces
    >>> square_sizes = find_largest_empty_spaces(img, n=10, plot=True)
    >>> 
    >>> # Save visualization
    >>> square_sizes = find_largest_empty_spaces(img, n=5, plot=True, 
    ...                                         save=True, save_path='./results')
    
    See Also
    --------
    create_bounded_pattern : Function for removing unnecessary padding
    """
    
    binary_array_copy = binary_array.copy()
    def find_largest_square(binary_array):
        rows, cols = binary_array.shape
        dp_row = np.zeros(cols, dtype=np.uint16)
        max_size = 0
        max_pos = (0, 0)
        for i in tqdm(range(rows), desc = 'finding largest square...', leave=False):
            prev = 0
            for j in range(cols):
                temp = dp_row[j]
                if binary_array[i, j] == 0:
                    dp_row[j] = min(dp_row[j], dp_row[j-1] if j > 0 else 0, prev) + 1
                    if dp_row[j] > max_size:
                        max_size = dp_row[j]
                        max_pos = (i, j)
                else:
                    dp_row[j] = 0
                prev = temp
        return max_size, max_pos

    def mask_square(binary_array, top_left, size):
        i, j = top_left
        binary_array[i:i+size, j:j+size] = 1

    def find_n_largest_squares(binary_array, n):
        largest_squares = []
        for _ in tqdm(range(n), desc="finding largest empty spaces", leave = False):
            max_size, max_pos = find_largest_square(binary_array)
            if max_size == 0:
                break
            # Since max_size is a scalar, don't index into it
            max_size_int = int(max_size)
            top_left = (int(max_pos[0] - max_size_int + 1), int(max_pos[1] - max_size_int + 1))
            largest_squares.append((max_size_int, top_left))
            mask_square(binary_array, top_left, max_size_int)
        return largest_squares

    # If modifying binary_array in-place is acceptable
    largest_squares = find_n_largest_squares(binary_array_copy, n)
    square_widths = [size for size, _ in largest_squares]

    # Plot only the final results for efficiency
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 9))
        visualized_array = np.stack((binary_array,) * 3, axis=-1) * 255
        thickness = 2

        for idx, (size, top_left) in enumerate(largest_squares):
            bottom_right = (top_left[0] + size - 1, top_left[1] + size - 1)
            visualized_array[top_left[0]:top_left[0]+thickness, top_left[1]:bottom_right[1]+1] = [0, 255, 0]
            visualized_array[bottom_right[0]-thickness+1:bottom_right[0]+1, top_left[1]:bottom_right[1]+1] = [0, 255, 0]
            visualized_array[top_left[0]:bottom_right[0]+1, top_left[1]:top_left[1]+thickness] = [0, 255, 0]
            visualized_array[top_left[0]:bottom_right[0]+1, bottom_right[1]-thickness+1:bottom_right[1]+1] = [0, 255, 0]
            center_x = (top_left[1] + bottom_right[1]) // 2
            center_y = (top_left[0] + bottom_right[0]) // 2
            axes[0].text(center_x, center_y, str(idx+1), color='red', fontsize=12, ha='center', va='center', fontweight='bold')

        axes[0].imshow(visualized_array)
        axes[0].set_title(f'Top {n} Largest Squares with Ranking')
        axes[0].axis('off')
        axes[1].plot(range(1, len(square_widths)+1), square_widths, marker='o', linestyle='-')
        axes[1].set_title('Square Widths vs. Ranking')
        axes[1].set_xlabel('Ranking')
        axes[1].set_ylabel('Square Width')
        plt.tight_layout()
        plt.show()

        if save and save_path:
            save_file = os.path.join(save_path, f"largest_empty_spaces.png")
            fig.savefig(save_file)

    return square_widths


def create_mask_from_largest_object(binary_image):
    """
    Create a binary mask isolating the largest object in a binary image.
    
    This function identifies the largest connected component in a binary image
    and creates a mask that isolates only that object. The mask is inverted
    so that the largest object appears as background (0) and everything else
    as foreground (255).
    
    Parameters
    ----------
    binary_image : np.ndarray
        2D binary image array where objects are represented as 1s and 
        background as 0s
        
    Returns
    -------
    np.ndarray
        Binary mask with dtype uint8 and values in {0, 255}:
        - 0: Largest object (inverted)
        - 255: Everything else (inverted)
        
        Returns array of zeros if no objects are found.
        
    Notes
    -----
    Processing Steps:
    1. Uses find_largest_smallest_objects to identify the largest object
    2. Creates a mask where only the largest object is marked as 255
    3. Inverts the mask so the largest object becomes background (0)
    
    The inversion is useful for certain image processing workflows where
    the largest object should be treated as background rather than foreground.
    
    Edge Cases:
    - If no objects are found, returns an array of zeros
    - If multiple objects have the same maximum area, selects one arbitrarily
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create binary image with multiple objects
    >>> img = np.zeros((100, 100))
    >>> img[10:30, 10:30] = 1  # Large square
    >>> img[50:55, 50:55] = 1  # Small square
    >>> 
    >>> # Create mask of largest object
    >>> mask = create_mask_from_largest_object(img)
    >>> print(f"Mask shape: {mask.shape}")
    >>> print(f"Unique values: {np.unique(mask)}")
    >>> 
    >>> # The largest object area will be 0 (background) in the mask
    >>> # Everything else will be 255 (foreground)
    
    See Also
    --------
    find_largest_smallest_objects : Function used to identify the largest object
    invert_array : Function used for mask inversion
    """
    # Use find_largest_smallest_objects to get the largest object and labeled image
    largest_object, _, _, _, labeled_image = find_largest_smallest_objects(binary_image)
    
    # Check if any objects were found
    if largest_object is None:
        print("No largest object found. Returning an empty mask.")
        return np.zeros_like(binary_image, dtype=np.uint8)

    # Create the mask for the largest object
    largest_object_label = largest_object.label  # Get the label of the largest object
    mask = (labeled_image == largest_object_label).astype(np.uint8) * 255

    mask = invert_array(mask)

    return mask

def create_bounded_pattern(pattern_array, margin=10):
    """
    Create a minimal bounding box around any fractal pattern, removing unnecessary padding.
    
    This function analyzes a 2D binary pattern array to find the minimal bounding box
    that contains all non-zero pixels, then crops the array to this region with an
    optional margin. This is particularly useful for removing excessive padding from
    generated fractal patterns or processed images.
    
    Parameters
    ----------
    pattern_array : np.ndarray
        2D binary array containing the fractal pattern or image data
    margin : int, default 10
        Additional pixels to add around the tight bounding box for safety margin.
        The margin is applied symmetrically on all sides but clipped to array bounds.
        
    Returns
    -------
    dict
        Dictionary containing the bounded pattern and metadata:
        
        - 'bounded_pattern' : np.ndarray
            Cropped pattern containing only the fractal data with margin
        - 'bounding_box' : tuple
            (top, left, bottom, right) coordinates of the crop region
        - 'offset' : tuple
            (x_offset, y_offset) to place bounded pattern in original coordinate system
        - 'original_shape' : tuple
            Shape of the original input pattern
        - 'bounded_shape' : tuple
            Shape of the bounded (cropped) pattern
        - 'memory_reduction' : float
            Percentage of memory saved by the cropping operation
            
    Notes
    -----
    Algorithm Details:
    
    1. **Non-zero Detection**: Finds all pixels with non-zero values
    2. **Bounding Box Calculation**: Computes tight bounds around non-zero pixels
    3. **Margin Application**: Adds specified margin while respecting array bounds
    4. **Cropping**: Extracts the bounded region as a copy
    5. **Metadata Calculation**: Computes memory savings and coordinate information
    
    Memory Efficiency:
    This function is particularly valuable for:
    - Reducing memory usage of sparse patterns
    - Removing excessive padding from generated fractals
    - Optimizing storage for pattern databases
    - Preparing patterns for further processing
    
    Coordinate System:
    - Bounding box coordinates use (top, left, bottom, right) convention
    - Offset uses (x, y) convention for placing bounded pattern back
    - All coordinates are suitable for numpy array slicing
    
    Edge Cases:
    - Empty patterns return a minimal 1x1 zero array
    - Patterns with margin extending beyond bounds are clipped
    - Boolean and numeric arrays are handled automatically
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a sparse pattern with lots of padding
    >>> pattern = np.zeros((200, 200))
    >>> pattern[80:120, 80:120] = 1  # Small central square
    >>> 
    >>> # Create bounded version
    >>> result = create_bounded_pattern(pattern, margin=5)
    >>> print(f"Original shape: {result['original_shape']}")
    >>> print(f"Bounded shape: {result['bounded_shape']}")
    >>> print(f"Memory reduction: {result['memory_reduction']:.1f}%")
    >>> 
    >>> # Use the bounded pattern
    >>> bounded = result['bounded_pattern']
    >>> offset = result['offset']
    >>> print(f"Pattern can be placed at offset {offset}")
    >>> 
    >>> # Reconstruct original coordinates if needed
    >>> top, left, bottom, right = result['bounding_box']
    >>> reconstructed = np.zeros_like(pattern)
    >>> reconstructed[top:bottom, left:right] = bounded
    
    See Also
    --------
    find_largest_empty_spaces : Function for analyzing empty regions
    pad_image_for_boxcounting : Function for adding padding for box counting
    """
    # Ensure we have a numpy array
    if not isinstance(pattern_array, np.ndarray):
        pattern_array = np.array(pattern_array)
    
    # Find non-zero pixels
    if pattern_array.dtype == bool:
        nonzero_pixels = np.where(pattern_array)
    else:
        nonzero_pixels = np.where(pattern_array > 0)
    
    if len(nonzero_pixels[0]) == 0:
        # Empty pattern - return minimal 1x1 pattern
        return {
            'bounded_pattern': np.zeros((1, 1), dtype=pattern_array.dtype),
            'bounding_box': (0, 0, 1, 1),
            'offset': (0, 0),
            'original_shape': pattern_array.shape,
            'bounded_shape': (1, 1),
            'memory_reduction': 99.99
        }
    
    # Calculate bounding box
    min_row, max_row = nonzero_pixels[0].min(), nonzero_pixels[0].max()
    min_col, max_col = nonzero_pixels[1].min(), nonzero_pixels[1].max()
    
    # Add margin but keep within bounds
    height, width = pattern_array.shape
    top = max(0, min_row - margin)
    left = max(0, min_col - margin)
    bottom = min(height, max_row + margin + 1)
    right = min(width, max_col + margin + 1)
    
    # Extract bounded region
    bounded_pattern = pattern_array[top:bottom, left:right].copy()
    
    # Calculate memory savings
    original_size = pattern_array.size
    bounded_size = bounded_pattern.size
    memory_reduction = ((original_size - bounded_size) / original_size) * 100
    
    result = {
        'bounded_pattern': bounded_pattern,
        'bounding_box': (top, left, bottom, right),
        'offset': (left, top),  # (x, y) offset for placing in original coordinates
        'original_shape': pattern_array.shape,
        'bounded_shape': bounded_pattern.shape,
        'memory_reduction': memory_reduction
    }
    
    return result