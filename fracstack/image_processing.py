import numpy as np
import skimage.io as io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
import os


def invert_array(arr):
    if arr.max() <= 1:
        # Handle binary arrays (0,1) by flipping 0 to 1 and 1 to 0
        return np.where(arr == 0, 1, 0)
    elif arr.max() <= 255:
        # Handle uint8 arrays (0 to 255) by subtracting from 255
        return 255 - arr
    else:
        raise ValueError("Array values exceed expected range for binary or uint8 format.")

def process_image_to_array(file_path, threshold=None, invert = False):
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

def pad_image_for_boxcounting(input_array, largest_box_size, pad_factor = 1, manual_pad = 0, invert = False):
    """
    Pads the input image symmetrically to ensure the grid fully covers it during box counting.
    
    Args:
        input_array (np.ndarray): Input binary array (2D).
        largest_box_size (int): The largest box size used for box counting.
    
    Returns:
        np.ndarray: Padded array.
    """

    if invert == True:
        input_array = invert_array(input_array)

    height, width = input_array.shape

    # Calculate new dimensions
    new_height = int(np.ceil(height / largest_box_size) *  largest_box_size * pad_factor + manual_pad)
    new_width = int(np.ceil(width / largest_box_size) *  largest_box_size * pad_factor + manual_pad)

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

def calculate_diameter(region):
    return region.equivalent_diameter

def bounding_box_diameter(region):
    # Calculate the height and width of the bounding box
    min_row, min_col, max_row, max_col = region.bbox
    height = max_row - min_row
    width = max_col - min_col
    # Return the larger of the two as the diameter
    return max(height, width)



def find_largest_smallest_objects(binary_image, invert=False):
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
    Create a binary mask for the largest object in a binary image.
    
    Args:
        binary_image (np.ndarray): Input binary image with objects labeled as 1s and background as 0s.
    
    Returns:
        np.ndarray: Binary mask of the largest object with 1s for the largest object and 0s elsewhere.
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
