import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score
from .image_processing import invert_array


def get_sizes(num_sizes, minsize, maxsize):
    sizes = list(np.around(np.geomspace(minsize, maxsize, num_sizes)).astype(int))
    for index in range(1, len(sizes)):
        size = sizes[index]
        prev_size = sizes[index - 1]
        if size <= prev_size:
            sizes[index] = prev_size + 1
            if prev_size == maxsize:
                return sizes[:index]
    return sizes

def get_mincount(array, size, mask=None, num_offsets=1):
    """
    Vectorized count of the minimum number of boxes covering non-zero regions of the array,
    constrained by an optional mask, considering multiple random grid offsets.

    Args:
        array (np.ndarray): 2D binary numpy array to analyze.
        size (int): Size of the boxes (size x size).
        mask (np.ndarray, optional): Optional 2D binary mask. Defaults to None.
        num_offsets (int, optional): Number of random grid offsets to test (includes centered grid).
                                     If num_offsets <= 1, only the centered grid is tested.
                                     Defaults to 1.

    Returns:
        int: Minimum number of boxes covering non-zero regions across all offsets.
    """
    shape = array.shape
    counts = []

    # Step 1: Calculate the centered grid offset
    centered_x_offset = (shape[0] % size) // 2
    centered_y_offset = (shape[1] % size) // 2

    # Step 2: Determine total possible unique offsets
    total_possible_offsets = size ** 2
    num_offsets = min(num_offsets, total_possible_offsets)  # Cap the number of offsets

    # Step 3: Generate random offsets if num_offsets > 1
    if num_offsets > 1 and total_possible_offsets > 1:
        # Exclude the centered grid from random offsets
        random_indices = np.random.choice(range(1, total_possible_offsets), size=num_offsets - 1, replace=False)
        offsets_x = random_indices % size
        offsets_y = random_indices // size
    else:
        offsets_x = np.array([], dtype=int)
        offsets_y = np.array([], dtype=int)

    # Include the centered grid as the first offset
    offsets_x = np.concatenate(([centered_x_offset], offsets_x)) if num_offsets > 0 else np.array([centered_x_offset])
    offsets_y = np.concatenate(([centered_y_offset], offsets_y)) if num_offsets > 0 else np.array([centered_y_offset])

    for x_offset, y_offset in zip(offsets_x, offsets_y):
        # Calculate the number of complete boxes along each dimension
        num_boxes_x = (shape[0] - x_offset) // size
        num_boxes_y = (shape[1] - y_offset) // size

        if num_boxes_x == 0 or num_boxes_y == 0:
            counts.append(0)
            continue

        # Calculate the end indices to ensure complete boxes
        end_x = x_offset + num_boxes_x * size
        end_y = y_offset + num_boxes_y * size

        # Slice the array to extract complete boxes
        sliced_array = array[x_offset:end_x, y_offset:end_y]

        # Reshape to (num_boxes_x, num_boxes_y, size, size)
        reshaped = sliced_array.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)

        if mask is not None:
            # Slice and reshape the mask similarly
            sliced_mask = mask[x_offset:end_x, y_offset:end_y]
            reshaped_mask = sliced_mask.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)

            # Determine which boxes have at least one mask element and one non-zero element
            box_has_mask = reshaped_mask.any(axis=(2, 3))
            box_has_content = reshaped.any(axis=(2, 3))
            valid_boxes = box_has_mask & box_has_content
            count = np.sum(valid_boxes)
        else:
            # Determine which boxes have at least one non-zero element
            box_has_content = reshaped.any(axis=(2, 3))
            count = np.sum(box_has_content)

        counts.append(count)

    # Return the minimum count across all offsets
    return min(counts) if counts else 0

def boxcount(array, num_sizes=10, min_size=None, max_size=None, num_pos=1, invert=False, mask=None):
    """
    Perform box-count analysis, optionally constrained to a specific mask.

    Args:
        array (np.ndarray): 2D binary numpy array (counts elements that aren't 0, or elements that aren't 1 if inverted).
        num_sizes (int): Number of box sizes.
        min_size (int, optional): Smallest box size in pixels (defaults to 1).
        max_size (int, optional): Largest box size in pixels (defaults to 1/5 smaller dimension of array).
        invert (bool, optional): If True, invert the binary array. Defaults to False.
        mask (np.ndarray, optional): Optional binary mask (1 for valid regions, 0 for excluded regions).

    Returns:
        tuple: (sizes, counts) where sizes is a list of box sizes and counts is a list of box counts.
    """
    if invert:
        array = invert_array(array)

    # Apply mask if provided
    if mask is not None:
        array = array * mask  # Exclude regions outside the mask

    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape) // 5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    counts = []

    for size in tqdm(sizes, desc='Calculating box counts', leave=False):
        counts.append(get_mincount(array, size, mask, num_offsets=num_pos))
    return sizes, counts

def compute_fractal_dimension(sizes, counts):
    
    # Convert to numpy arrays if not already
    sizes = np.array(sizes)
    counts = np.array(counts)
    
    # Filter out non-positive sizes and counts
    valid = (sizes > 0) & (counts > 0)
    valid_sizes = sizes[valid]
    valid_counts = counts[valid]
    
    if len(valid_sizes) > 0 and len(valid_counts) > 0:
        # Perform linear fit on log-log data
        fit, cov = np.polyfit(np.log10(valid_sizes), np.log10(valid_counts), 1, cov=True)
        r2 = r2_score(np.log10(valid_counts), fit[0] * np.log10(valid_sizes) + fit[1])
        d_value = -fit[0]  # Fractal dimension
    else:
        # Handle cases with insufficient data
        valid_sizes = np.array([])
        valid_counts = np.array([])
        d_value = np.nan
        fit = np.array([np.nan, np.nan])
        r2 = np.nan
    
    return valid_sizes, valid_counts, d_value, fit, r2
