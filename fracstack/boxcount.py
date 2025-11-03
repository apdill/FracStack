import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from scipy.stats import t
import time
from numba import njit, prange
from .image_processing import pad_image_for_boxcounting

def generate_random_offsets(sizes, num_offsets, seed=None):
    """
    Generate random grid offsets for box counting to avoid thread safety issues in numba parallel functions.
    
    For each box size, generates a set of random starting positions for the box counting grid.
    The first offset is always the geometric center of the image for consistency. Additional
    offsets are generated randomly within the box size constraints to sample different
    grid alignments and reduce bias from edge effects.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (in pixels) for which to generate offsets
    num_offsets : int
        Number of offset positions to generate for each box size
    seed : int, optional
        Random seed for reproducible results. If None, uses current random state
        
    Returns
    -------
    np.ndarray
        Array of shape (len(sizes), num_offsets, 2) containing (x_offset, y_offset) pairs.
        Values are of type uint16 to save memory. For each size, the first offset is
        always the geometric center, followed by random offsets within the box bounds.
        If num_offsets exceeds the number of possible unique positions (size²), 
        the remaining slots are filled with the last valid offset.
    """
    if seed is not None: 
        np.random.seed(seed)
    
    off = np.zeros((len(sizes), num_offsets, 2), dtype=np.uint16)
    
    for i, s in enumerate(sizes):
        # First offset is always the geometric center
        off[i, 0] = (s // 2, s // 2)
        
        # Generate additional random offsets
        j = 0  # Track last valid offset index
        for j in range(1, min(num_offsets, s * s)):
            r = np.random.randint(0, s * s)
            off[i, j] = (r % s, r // s)
        
        # Fill remaining slots with last valid offset if needed
        if j + 1 < num_offsets:
            off[i, j + 1:] = off[i, j]
            
    return off

def get_pairwise_slopes(sizes, measures, return_second_order=False):
    """
    Calculate pairwise slopes between consecutive points in log-log scaling data.
    
    This function computes the local slopes between consecutive points in log-log
    scaling data, which is essential for analyzing the local scaling behavior
    of fractal systems. It can also compute second-order derivatives to detect
    changes in scaling behavior.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (scale parameters) in ascending order
    measures : array-like
        Array of corresponding measures (box counts, entropies, etc.)
    return_second_order : bool, default False
        Whether to also compute second-order slope derivatives
        
    Returns
    -------
    np.ndarray or tuple
        If return_second_order=False:
            slopes : np.ndarray - Pairwise slopes between consecutive log-transformed points
        If return_second_order=True:
            (slopes, second_order_slopes) : tuple - Both first and second-order slopes
            
    Notes
    -----
    Mathematical Details:
    
    The function computes slopes as:
    slopes[i] = (log(measures[i+1]) - log(measures[i])) / (log(sizes[i+1]) - log(sizes[i]))
    
    For second-order derivatives:
    second_order_slopes[i] = (slopes[i+1] - slopes[i]) / (log(sizes[i+2]) - log(sizes[i]))
    
    These calculations are fundamental for:
    - Detecting regions of consistent scaling behavior
    - Identifying transitions between different scaling regimes
    - Plateau detection algorithms
    - Quality assessment of fractal dimension estimates
    
    Array Dimensions:
    - Input arrays: length n
    - First-order slopes: length n-1
    - Second-order slopes: length n-2
    
    Applications:
    - Fractal dimension analysis
    - Scaling behavior characterization
    - Plateau detection
    - Quality control in box counting
    - Identification of scaling transitions
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic scaling data
    >>> sizes = np.array([1, 2, 4, 8, 16, 32])
    >>> measures = sizes**(-1.5)  # Perfect scaling with D=1.5
    >>> 
    >>> # Calculate pairwise slopes
    >>> slopes = get_pairwise_slopes(sizes, measures)
    >>> print(f"Slopes: {slopes}")  # Should be approximately -1.5
    >>> 
    >>> # Include second-order analysis
    >>> slopes, second_order = get_pairwise_slopes(sizes, measures, return_second_order=True)
    >>> print(f"Second-order slopes: {second_order}")  # Should be near zero for perfect scaling
    >>> 
    >>> # Analyze real box counting data
    >>> from fracstack.boxcount import boxcount
    >>> binary_img = np.random.choice([0, 1], size=(256, 256), p=[0.8, 0.2])
    >>> sizes, counts = boxcount(binary_img, mode='D0')
    >>> slopes = get_pairwise_slopes(sizes, counts)
    >>> median_slope = np.median(slopes)
    >>> print(f"Median slope: {median_slope:.3f}, Implied D0: {-median_slope:.3f}")
    
    See Also
    --------
    detect_scaling_plateau : Uses pairwise slopes for plateau detection
    detect_plateau_pairwise : Uses slope differences for plateau detection
    plot_pairwise_slopes : Visualization of pairwise slopes 
    """
    log_eps = np.log(sizes)
    log_N   = np.log(measures)
    slopes = np.diff(log_N) / np.diff(log_eps)
    if return_second_order:
        second_order_slopes = np.diff(slopes) / np.diff(log_eps[1:])
        return slopes, second_order_slopes
    return slopes

def detect_scaling_plateau(sizes,
                           measures,
                           window: int = 4,
                           tol: float = 0.01,
                           min_pts: int = 8):
    """
    Detect scaling plateau by finding the longest region where slopes are stable around the median.
    
    This function identifies regions in log-log scaling data where the local slopes are
    consistent with the overall median slope, indicating good fractal scaling behavior.
    It uses relative tolerance to ensure fairness across different fractal dimensions.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (scale parameters) in ascending order
    measures : array-like
        Array of corresponding measures (box counts, entropies, etc.) 
    window : int, default 4
        Minimum window size for local slope calculation (currently unused)
    tol : float, default 0.01
        Relative tolerance for slope stability as fraction of median slope magnitude.
        For example, 0.01 means slopes must be within 1% of the median slope
    min_pts : int, default 8
        Minimum number of data points required for a valid plateau
        
    Returns
    -------
    tuple
        (start_index, stop_index) defining the plateau region in the input arrays,
        or (None, None) if no suitable plateau is found
        
    Notes
    -----
    The algorithm works by:
    1. Computing two-point slopes between consecutive log-transformed data points
    2. Finding the median slope as a reference
    3. Identifying slopes that are within the relative tolerance of the median
    4. Finding the longest contiguous run of stable slopes
    5. Converting slope indices back to data point indices
    
    The relative tolerance approach ensures that the same percentage deviation is
    allowed regardless of whether the fractal dimension is close to 1 or 2.
    """
    if len(sizes) < window + 2:
        return None, None

    slopes = get_pairwise_slopes(sizes, measures)

    # Apply stability criterion using relative tolerance
    med = np.median(slopes)
    if np.abs(med) < 1e-10:  # Handle near-zero median slope
        stable = np.abs(slopes - med) < tol
    else:
        # Use relative tolerance: |slope_i - median| < tol * |median|
        stable = np.abs(slopes - med) < tol * np.abs(med)

    # Find the longest contiguous run of stable slopes
    best_len = 0
    best_i = None
    cur_len = 0
    cur_start = 0
    
    for i, ok in enumerate(stable):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len and cur_len + 1 >= min_pts:
                best_len = cur_len
                best_i = cur_start
        else:
            cur_len = 0

    if best_i is None:
        return None, None

    # Convert slope indices to data point indices
    # Need +1 because slopes connect consecutive points
    return best_i, best_i + best_len + 1

def detect_plateau_pairwise(sizes,
                            measures,
                            window: int = 4,
                            d_tol: float = 0.01,
                            min_pts: int = 8):
    """
    Detect scaling plateau using pairwise slope differences to identify staircasing artifacts.
    
    This method examines the smoothness of the log-log curve by looking at differences
    between consecutive slopes. It's particularly effective at detecting staircasing
    artifacts where slopes change abruptly between relatively flat regions, which
    can indicate non-fractal or composite scaling behavior.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (scale parameters) in ascending order
    measures : array-like
        Array of corresponding measures (box counts, entropies, etc.)
    window : int, default 4
        Minimum window size for local slope calculation (currently unused)
    d_tol : float, default 0.01
        Relative tolerance for pairwise slope differences as fraction of median slope magnitude.
        For example, 0.01 means consecutive slopes must differ by less than 1% of the median slope
    min_pts : int, default 8
        Minimum number of data points required for a valid plateau
        
    Returns
    -------
    tuple
        (start_index, stop_index) defining the plateau region in the input arrays,
        or (None, None) if no suitable plateau is found
        
    Notes
    -----
    The algorithm works by:
    1. Computing two-point slopes between consecutive log-transformed data points
    2. Calculating absolute differences between consecutive slopes
    3. Using the median slope magnitude to set a relative tolerance threshold
    4. Finding the longest contiguous run where slope differences are below threshold
    5. Converting slope difference indices back to data point indices
    
    This approach is complementary to the median-based method and can catch different
    types of scaling problems, particularly those involving abrupt transitions between
    different scaling regimes.
    """
    if len(sizes) < window + 2:
        return None, None

    slopes = get_pairwise_slopes(sizes, measures)
    
    if len(slopes) < 2:
        return None, None
    
    # Calculate absolute differences between consecutive slopes
    slope_diffs = np.abs(np.diff(slopes))
    
    # Apply stability criterion using relative tolerance
    med_slope = np.median(slopes)
    if np.abs(med_slope) < 1e-10:  # Handle near-zero median slope
        stable = slope_diffs < d_tol
    else:
        # Use relative tolerance: |slope_diff| < d_tol * |median_slope|
        stable = slope_diffs < d_tol * np.abs(med_slope)
    
    # Find the longest contiguous run of stable slope differences
    best_len = 0
    best_i = None
    cur_len = 0
    cur_start = 0
    
    for i, ok in enumerate(stable):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            # Check if this run is long enough and better than previous best
            # cur_len + 2 because slope differences correspond to cur_len + 2 data points
            if cur_len > best_len and cur_len + 2 >= min_pts:
                best_len = cur_len
                best_i = cur_start
        else:
            cur_len = 0

    if best_i is None:
        return None, None

    # Convert from slope difference indices to data point indices
    start_idx = best_i
    stop_idx = best_i + best_len + 2  # +2 because slope differences span 2 more points
    
    return start_idx, stop_idx

def detect_plateau_hybrid(sizes,
                          measures,
                          window: int = 4,
                          median_tol: float = 0.01,
                          pairwise_tol: float = 0.01,
                          min_pts: int = 8,
                          method: str = 'pairwise_first'):
    """
    Combine median-based and pairwise plateau detection methods for robust scaling region identification.
    
    This function provides multiple strategies for combining the two plateau detection approaches,
    allowing for more robust identification of scaling regions by leveraging the strengths of
    both methods. The median-based method is good for overall stability, while the pairwise
    method is better at detecting staircasing artifacts.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (scale parameters) in ascending order
    measures : array-like
        Array of corresponding measures (box counts, entropies, etc.)
    window : int, default 4
        Minimum window size for local slope calculation
    median_tol : float, default 0.01
        Relative tolerance for median-based plateau detection as fraction of median slope magnitude
    pairwise_tol : float, default 0.01
        Relative tolerance for pairwise slope difference detection as fraction of median slope magnitude
    min_pts : int, default 8
        Minimum number of data points required for a valid plateau
    method : str, default 'pairwise_first'
        Strategy for combining the two methods:
        - 'pairwise_first': Try pairwise method first, fallback to median if it fails
        - 'median_first': Try median method first, fallback to pairwise if it fails
        - 'intersection': Use intersection of both methods (most conservative)
        - 'longest': Use whichever method gives the longer plateau
            
    Returns
    -------
    tuple
        (start_index, stop_index, method_used) where method_used indicates which
        detection method was ultimately used ('median', 'pairwise', or 'intersection'),
        or (None, None, None) if no plateau found
        
    Notes
    -----
    The different combination strategies serve different purposes:
    - 'pairwise_first' and 'median_first' provide fallback behavior
    - 'intersection' is most conservative, requiring both methods to agree
    - 'longest' maximizes the scaling range while maintaining quality
    
    Both methods use relative tolerance to ensure fairness across different fractal dimensions.
    """
    if len(sizes) < window + 2:
        return None, None, None

    # Try both plateau detection methods
    pairwise_result = detect_plateau_pairwise(sizes, measures, window, pairwise_tol, min_pts)
    median_result = detect_scaling_plateau(sizes, measures, window, median_tol, min_pts)
    
    if method == 'pairwise_first':
        if pairwise_result[0] is not None:
            return pairwise_result[0], pairwise_result[1], 'pairwise'
        elif median_result[0] is not None:
            return median_result[0], median_result[1], 'median'
        else:
            return None, None, None
            
    elif method == 'median_first':
        if median_result[0] is not None:
            return median_result[0], median_result[1], 'median'
        elif pairwise_result[0] is not None:
            return pairwise_result[0], pairwise_result[1], 'pairwise'
        else:
            return None, None, None
            
    elif method == 'longest':
        pairwise_len = 0 if pairwise_result[0] is None else pairwise_result[1] - pairwise_result[0]
        median_len = 0 if median_result[0] is None else median_result[1] - median_result[0]
        
        if pairwise_len >= median_len and pairwise_result[0] is not None:
            return pairwise_result[0], pairwise_result[1], 'pairwise'
        elif median_result[0] is not None:
            return median_result[0], median_result[1], 'median'
        else:
            return None, None, None
            
    elif method == 'intersection':
        # Use intersection of both methods if they both succeed
        if pairwise_result[0] is not None and median_result[0] is not None:
            # Find overlapping region
            start = max(pairwise_result[0], median_result[0])
            stop = min(pairwise_result[1], median_result[1])
            
            if stop - start >= min_pts:
                return start, stop, 'intersection'
            else:
                # If intersection is too small, use the longer of the two
                pairwise_len = pairwise_result[1] - pairwise_result[0]
                median_len = median_result[1] - median_result[0]
                
                if pairwise_len >= median_len:
                    return pairwise_result[0], pairwise_result[1], 'pairwise'
                else:
                    return median_result[0], median_result[1], 'median'
        elif pairwise_result[0] is not None:
            return pairwise_result[0], pairwise_result[1], 'pairwise'
        elif median_result[0] is not None:
            return median_result[0], median_result[1], 'median'
        else:
            return None, None, None
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pairwise_first', 'median_first', 'intersection', or 'longest'")

def filter_by_occupancy(array, sizes, counts,
                        occ_low=0.05, occ_high=0.95):
    """
    Filter out box sizes with extreme occupancy rates that don't provide useful fractal information.
    
    Occupancy is defined as the fraction of boxes in the grid that contain at least one
    non-zero pixel. Very low occupancy (sparse sampling) can lead to noisy dimension estimates,
    while very high occupancy (over-sampling) indicates that the box size is too small to
    capture meaningful scaling behavior. This function removes both extremes to improve
    the quality of fractal dimension estimates.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary image array used for box counting
    sizes : np.ndarray
        1D array of box sizes (in pixels) 
    counts : np.ndarray
        1D array of box counts N(ε) corresponding to each size
    occ_low : float, default 0.05
        Lower occupancy threshold. Scales with occupancy ≤ occ_low are removed
    occ_high : float, default 0.95
        Upper occupancy threshold. Scales with occupancy ≥ occ_high are removed
        
    Returns
    -------
    tuple
        (filtered_sizes, filtered_counts) - Arrays containing only the sizes and counts
        for scales with occupancy in the acceptable range (occ_low, occ_high)
        
    Notes
    -----
    Occupancy is calculated as:
        occupancy = N(ε) / (total_boxes_in_grid)
    where total_boxes_in_grid = floor(H/ε) * floor(W/ε)
    
    Typical filtering removes:
    - Under-sampled scales (occupancy < 5%): Too few boxes contain structure
    - Over-sampled scales (occupancy > 95%): Almost all boxes contain structure
    
    The floor division matches the behavior of the numba box counting kernels.
    """
    H, W = array.shape
    
    # Calculate total number of boxes in grid for each scale
    # Use integer division to match numba kernel behavior
    grid_boxes = (np.floor_divide(H, sizes) * np.floor_divide(W, sizes))
    
    # Calculate occupancy (fraction of boxes that contain fractal structure)
    occupancy = counts / grid_boxes
    
    # Keep only scales with occupancy in useful range
    mask = (occupancy > occ_low) & (occupancy < occ_high)
    
    return sizes[mask], counts[mask]

@njit(nogil=True, parallel=True, cache=True)
def numba_d0(array, sizes, offsets, use_min_count=False):
    """
    Compute capacity dimension (D0) box counts using basic numba-optimized algorithm.
    
    This is the unoptimized version of the D0 box counting algorithm, provided primarily
    for benchmarking and debugging purposes. It processes all boxes in the grid without
    spatial optimizations. For production use, prefer numba_d0_optimized.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in pixels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 2) containing pre-generated
        (x_offset, y_offset) pairs for each size
    use_min_count : bool, default False
        If True, return minimum count across all offsets for each size.
        If False, return average count across all offsets for each size.
        
    Returns
    -------
    np.ndarray
        1D array of box counts for each size. If use_min_count=True, contains
        minimum counts; if False, contains average counts (may be fractional).
        
    Notes
    -----
    The algorithm works by:
    1. For each box size, testing multiple grid offset positions
    2. For each offset, counting boxes that contain at least one non-zero pixel
    3. Combining results across offsets (minimum or average based on use_min_count)
    
    This function is compiled with numba for performance but lacks spatial optimizations
    like bounding box culling. It processes every possible box position regardless of
    whether the box might contain structure.
    
    See Also
    --------
    numba_d0_optimized : Optimized version with bounding box culling
    numba_d0_sparse : Specialized version for very sparse arrays
    """
    results = np.empty(len(sizes), dtype=np.int64)
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        H, W = array.shape
        
        # Calculate actual centered offsets for this array size
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        
        if use_min_count:
            # Find minimum count across offsets
            min_count = np.inf
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                # Box counting logic
                count = 0
                max_x = x_off + ((H - x_off) // size) * size
                max_y = y_off + ((W - y_off) // size) * size
                
                for x in range(x_off, max_x, size):
                    for y in range(y_off, max_y, size):
                        count += array[x:x+size, y:y+size].any()
                
                if count < min_count:
                    min_count = count
            
            results[idx] = min_count if min_count != np.inf else 0
        else:
            # Average count across offsets
            count_sum = 0
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                # Box counting logic
                count = 0
                max_x = x_off + ((H - x_off) // size) * size
                max_y = y_off + ((W - y_off) // size) * size
                
                for x in range(x_off, max_x, size):
                    for y in range(y_off, max_y, size):
                        count += array[x:x+size, y:y+size].any()
                
                count_sum += count
            
            results[idx] = count_sum / total_offsets if total_offsets > 0 else 0
    
    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d1(array, sizes, offsets):
    """
    Compute information dimension (D1) entropy values using basic numba-optimized algorithm.
    
    This is the unoptimized version of the D1 information dimension algorithm, provided
    primarily for benchmarking and debugging purposes. It processes all boxes in the grid
    without spatial optimizations. For production use, prefer numba_d1_optimized.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in pixels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 2) containing pre-generated
        (x_offset, y_offset) pairs for each size
        
    Returns
    -------
    np.ndarray
        1D array of entropy values H(ε) for each box size, averaged across all offsets.
        Values are in bits (using log2).
        
    Notes
    -----
    The information dimension algorithm works by:
    1. Computing the total mass M = sum of all pixel values in the array
    2. For each box size and offset, calculating the Shannon entropy:
       H = -Σ p_i * log2(p_i) where p_i = (box_sum_i / M)
    3. Averaging entropy values across all offsets for each size
    
    The information dimension D1 is then computed as the negative slope of H(ε) vs log2(ε).
    
    This function handles edge cases:
    - Returns 0 if the array is empty (M = 0)
    - Returns 0 if box size is 0
    - Skips boxes with zero sum (they don't contribute to entropy)
    
    See Also
    --------
    numba_d1_optimized : Optimized version with bounding box culling
    compute_dimension : Function to compute D1 from entropy values
    """
    results = np.empty(len(sizes), dtype=np.float64)
    M = array.sum()
    H, W = array.shape
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        if M == 0 or size == 0:
            results[idx] = 0.0
            continue
            
        # Calculate actual centered offsets for this array size
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        entropy_sum = 0.0
        
        for offset_idx in range(total_offsets):
            # Get offset coordinates
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                # Use pre-generated offsets
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size
            
            # Box processing
            max_x = x_off + ((H - x_off) // size) * size
            max_y = y_off + ((W - y_off) // size) * size
            entropy = 0.0
            
            for x in range(x_off, max_x, size):
                for y in range(y_off, max_y, size):
                    box = array[x:x+size, y:y+size]
                    box_sum = box.sum()
                    if box_sum > 0:
                        p = box_sum / M
                        entropy += -p * np.log2(p)
            
            entropy_sum += entropy
        
        # Average across offsets
        results[idx] = entropy_sum / total_offsets if total_offsets > 0 else 0.0
    
    return results

@njit(nogil=True, cache=True)
def get_bounding_box(array):
    """
    Compute the tight bounding box of all non-zero pixels in a 2D array.
    
    This function finds the smallest rectangle that contains all non-zero pixels,
    which is used by the optimized box counting algorithms to skip empty regions
    and improve performance.
    
    Parameters
    ----------
    array : np.ndarray
        2D array to analyze (typically binary, but works with any numeric array)
        
    Returns
    -------
    tuple
        (min_row, min_col, max_row, max_col) defining the bounding box where:
        - min_row, min_col: Top-left corner (inclusive)
        - max_row, max_col: Bottom-right corner (exclusive, suitable for slicing)
        
        Returns (0, 0, 0, 0) if the array contains no non-zero pixels.
        
    Notes
    -----
    The algorithm works by:
    1. Scanning all pixels to identify which rows and columns contain non-zero values
    2. Finding the first and last non-zero row and column
    3. Returning coordinates in a format suitable for array slicing
    
    The returned coordinates follow Python slicing conventions where max_row and
    max_col are exclusive bounds, so array[min_row:max_row, min_col:max_col] 
    gives the bounding box contents.
    
    This function is numba-compiled for performance as it's called frequently
    by the optimized box counting algorithms.
    """
    H, W = array.shape
    
    # Find rows and columns that contain non-zero pixels
    row_has_data = np.zeros(H, dtype=np.bool_)
    col_has_data = np.zeros(W, dtype=np.bool_)
    
    for i in range(H):
        for j in range(W):
            if array[i, j] > 0:
                row_has_data[i] = True
                col_has_data[j] = True
    
    # Find the bounds of non-zero regions
    min_row, max_row = H, -1
    min_col, max_col = W, -1
    
    for i in range(H):
        if row_has_data[i]:
            if min_row == H:
                min_row = i
            max_row = i
    
    for j in range(W):
        if col_has_data[j]:
            if min_col == W:
                min_col = j
            max_col = j
    
    if min_row == H:  # No non-zero pixels found
        return 0, 0, 0, 0
    
    return min_row, min_col, max_row + 1, max_col + 1

@njit(nogil=True, cache=True)
def box_intersects_bounds(row, col, size, min_row, min_col, max_row, max_col):
    """
    Check if a box intersects with a given bounding box region.
    
    This function is used by the optimized box counting algorithms to determine
    whether a box at a given position could potentially contain non-zero pixels,
    allowing the algorithm to skip empty regions for better performance.
    
    Parameters
    ----------
    row : int
        Top-left row coordinate of the box to test
    col : int
        Top-left column coordinate of the box to test
    size : int
        Size of the box (both width and height)
    min_row : int
        Minimum row of the bounding box (inclusive)
    min_col : int
        Minimum column of the bounding box (inclusive)
    max_row : int
        Maximum row of the bounding box (exclusive)
    max_col : int
        Maximum column of the bounding box (exclusive)
        
    Returns
    -------
    bool
        True if the box intersects with the bounding box, False otherwise
        
    Notes
    -----
    The function uses standard rectangle intersection logic:
    - Two rectangles don't intersect if one is completely to the left, right,
      above, or below the other
    - The function returns the negation of the non-intersection condition
    
    This is a performance-critical function that's called many times during
    box counting, so it's compiled with numba and kept simple.
    
    The box coordinates follow the same convention as the bounding box:
    - (row, col) is the top-left corner (inclusive)
    - The box extends to (row + size, col + size) (exclusive)
    """
    box_max_row = row + size
    box_max_col = col + size
    
    # Box doesn't intersect if it's completely outside the bounds
    return not (box_max_row <= min_row or row >= max_row or 
                box_max_col <= min_col or col >= max_col)

@njit(nogil=True, parallel=True, cache=True)
def numba_d0_optimized(array, sizes, offsets, use_min_count=False):
    """
    Compute capacity dimension (D0) box counts using bounding box optimization.
    
    This is the optimized version of the D0 box counting algorithm that uses spatial
    optimizations to improve performance. It precomputes a bounding box of non-zero
    pixels and skips boxes that don't intersect with this region, providing significant
    speedup for sparse arrays while producing identical results to the basic version.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in pixels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 2) containing pre-generated
        (x_offset, y_offset) pairs for each size
    use_min_count : bool, default False
        If True, return minimum count across all offsets for each size.
        If False, return average count across all offsets for each size.
        
    Returns
    -------
    np.ndarray
        1D array of box counts for each size. If use_min_count=True, contains
        minimum counts; if False, contains average counts (may be fractional).
        Results are identical to numba_d0 but computed more efficiently.
        
    Notes
    -----
    Optimizations over the basic numba_d0 function:
    1. Precomputes bounding box of non-zero pixels once before processing
    2. Uses box_intersects_bounds() to skip empty boxes during counting
    3. Early exit for completely empty arrays
    
    The algorithm works by:
    1. Computing the tight bounding box of all non-zero pixels
    2. For each box size and offset, only processing boxes that intersect the bounding box
    3. Counting boxes that contain at least one non-zero pixel
    4. Combining results across offsets (minimum or average based on use_min_count)
    
    Performance improvements are most significant for sparse arrays where many boxes
    contain no structure. For dense arrays, the overhead of bounding box checks may
    slightly reduce performance, but results remain identical.
    
    See Also
    --------
    numba_d0 : Basic unoptimized version
    numba_d0_sparse : Specialized version for very sparse arrays
    get_bounding_box : Function that computes the bounding box
    box_intersects_bounds : Function that checks box-bounding box intersection
    """
    results = np.empty(len(sizes), dtype=np.int64)
    
    # Precompute bounding box once for all sizes
    min_row, min_col, max_row, max_col = get_bounding_box(array)
    
    # Early exit for completely empty arrays
    if min_row == max_row and min_col == max_col:
        results.fill(0)
        return results
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        H, W = array.shape
        
        # Calculate actual centered offsets for this array size
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        
        if use_min_count:
            # Find minimum count across offsets
            min_count = np.inf
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                # Box counting logic with bounding box optimization
                count = 0
                max_x = x_off + ((H - x_off) // size) * size
                max_y = y_off + ((W - y_off) // size) * size
                
                for x in range(x_off, max_x, size):
                    for y in range(y_off, max_y, size):
                        # Skip boxes that don't intersect with bounding box
                        if box_intersects_bounds(x, y, size, min_row, min_col, max_row, max_col):
                            count += array[x:x+size, y:y+size].any()
                
                if count < min_count:
                    min_count = count
            
            results[idx] = min_count if min_count != np.inf else 0
        else:
            # Average count across offsets
            count_sum = 0
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                # Box counting logic with bounding box optimization
                count = 0
                max_x = x_off + ((H - x_off) // size) * size
                max_y = y_off + ((W - y_off) // size) * size
                
                for x in range(x_off, max_x, size):
                    for y in range(y_off, max_y, size):
                        # Skip boxes that don't intersect with bounding box
                        if box_intersects_bounds(x, y, size, min_row, min_col, max_row, max_col):
                            count += array[x:x+size, y:y+size].any()
                
                count_sum += count
            
            results[idx] = count_sum / total_offsets if total_offsets > 0 else 0
    
    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d1_optimized(array, sizes, offsets):
    """
    Compute information dimension (D1) entropy values using bounding box optimization.
    
    This is the optimized version of the D1 information dimension algorithm that uses
    spatial optimizations to improve performance. It precomputes a bounding box of
    non-zero pixels and skips boxes that don't intersect with this region, providing
    significant speedup for sparse arrays while producing identical results to the
    basic version.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in pixels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 2) containing pre-generated
        (x_offset, y_offset) pairs for each size
        
    Returns
    -------
    np.ndarray
        1D array of entropy values H(ε) for each box size, averaged across all offsets.
        Values are in bits (using log2). Results are identical to numba_d1 but
        computed more efficiently.
        
    Notes
    -----
    Optimizations over the basic numba_d1 function:
    1. Precomputes bounding box of non-zero pixels once before processing
    2. Uses box_intersects_bounds() to skip empty boxes during entropy calculation
    3. Early exit for completely empty arrays
    
    The algorithm works by:
    1. Computing the total mass M = sum of all pixel values in the array
    2. Computing the tight bounding box of all non-zero pixels
    3. For each box size and offset, only processing boxes that intersect the bounding box
    4. Calculating Shannon entropy: H = -Σ p_i * log2(p_i) where p_i = (box_sum_i / M)
    5. Averaging entropy values across all offsets for each size
    
    Performance improvements are most significant for sparse arrays where many boxes
    contain no structure. For dense arrays, the overhead of bounding box checks may
    slightly reduce performance, but results remain identical.
    
    The information dimension D1 is computed as the negative slope of H(ε) vs log2(ε).
    
    See Also
    --------
    numba_d1 : Basic unoptimized version
    get_bounding_box : Function that computes the bounding box
    box_intersects_bounds : Function that checks box-bounding box intersection
    compute_dimension : Function to compute D1 from entropy values
    """
    results = np.empty(len(sizes), dtype=np.float64)
    M = array.sum()
    H, W = array.shape
    
    # Precompute bounding box once for all sizes
    min_row, min_col, max_row, max_col = get_bounding_box(array)
    
    # Early exit for completely empty arrays
    if M == 0 or (min_row == max_row and min_col == max_col):
        results.fill(0.0)
        return results
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        if size == 0:
            results[idx] = 0.0
            continue
            
        # Calculate actual centered offsets for this array size
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        entropy_sum = 0.0
        
        for offset_idx in range(total_offsets):
            # Get offset coordinates
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                # Use pre-generated offsets
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size
            
            # Box processing with bounding box optimization
            max_x = x_off + ((H - x_off) // size) * size
            max_y = y_off + ((W - y_off) // size) * size
            entropy = 0.0
            
            for x in range(x_off, max_x, size):
                for y in range(y_off, max_y, size):
                    # Skip boxes that don't intersect with bounding box
                    if box_intersects_bounds(x, y, size, min_row, min_col, max_row, max_col):
                        box = array[x:x+size, y:y+size]
                        box_sum = box.sum()
                        if box_sum > 0:
                            p = box_sum / M
                            entropy += -p * np.log2(p)
            
            entropy_sum += entropy
        
        # Average across offsets
        results[idx] = entropy_sum / total_offsets if total_offsets > 0 else 0.0
    
    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d2(array, sizes, offsets):
    """
    Compute correlation (mass) dimension measure (D2) using basic algorithm.

    For each box size and offset, this function partitions the array into
    non-overlapping boxes, computes the probability mass p_i of each box
    (box mass divided by total mass), and accumulates the correlation sum
    Σ p_i². Results are averaged across offsets.
    """
    results = np.empty(len(sizes), dtype=np.float64)
    total_mass = array.sum()
    H, W = array.shape

    if total_mass <= 0.0:
        results.fill(0.0)
        return results

    for idx in prange(len(sizes)):
        size = sizes[idx]
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        corr_sum = 0.0

        for offset_idx in range(total_offsets):
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size

            max_x = x_off + ((H - x_off) // size) * size
            max_y = y_off + ((W - y_off) // size) * size
            corr_offset = 0.0

            for x in range(x_off, max_x, size):
                for y in range(y_off, max_y, size):
                    mass = array[x:x+size, y:y+size].sum()
                    if mass > 0.0:
                        p = mass / total_mass
                        corr_offset += p * p

            corr_sum += corr_offset

        results[idx] = corr_sum / total_offsets if total_offsets > 0 else 0.0

    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d2_optimized(array, sizes, offsets):
    """
    Optimized correlation (mass) dimension measure using bounding box culling.

    This version mirrors numba_d1_optimized: it precomputes the bounding box of
    non-zero pixels to skip obviously empty boxes while producing identical
    results to the basic implementation.
    """
    results = np.empty(len(sizes), dtype=np.float64)
    total_mass = array.sum()
    H, W = array.shape

    min_row, min_col, max_row, max_col = get_bounding_box(array)

    if total_mass <= 0.0 or (min_row == max_row and min_col == max_col):
        results.fill(0.0)
        return results

    for idx in prange(len(sizes)):
        size = sizes[idx]
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        corr_sum = 0.0

        for offset_idx in range(total_offsets):
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size

            max_x = x_off + ((H - x_off) // size) * size
            max_y = y_off + ((W - y_off) // size) * size
            corr_offset = 0.0

            for x in range(x_off, max_x, size):
                for y in range(y_off, max_y, size):
                    if box_intersects_bounds(x, y, size, min_row, min_col, max_row, max_col):
                        mass = array[x:x+size, y:y+size].sum()
                        if mass > 0.0:
                            p = mass / total_mass
                            corr_offset += p * p

            corr_sum += corr_offset

        results[idx] = corr_sum / total_offsets if total_offsets > 0 else 0.0

    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d2_gliding(array, sizes):
    """
    Compute gliding-window correlation dimension measure.

    Uses summed-area tables to evaluate every possible box position. Returns
    the mean correlation sum (Σ p_i² / N_windows) for each size along with
    the total number of evaluated windows.
    """
    H, W = array.shape
    total_mass = array.sum()
    n_sizes = len(sizes)
    results = np.empty(n_sizes, dtype=np.float64)
    window_counts = np.zeros(n_sizes, dtype=np.int64)

    if total_mass <= 0.0:
        results.fill(0.0)
        return results, window_counts

    integral = _compute_integral_image(array)

    for idx in prange(n_sizes):
        size = sizes[idx]
        if size <= 0 or size > H or size > W:
            results[idx] = 0.0
            window_counts[idx] = 0
            continue

        max_row = H - size + 1
        max_col = W - size + 1
        total_windows = max_row * max_col
        window_counts[idx] = total_windows

        if total_windows == 0:
            results[idx] = 0.0
            continue

        corr_sum = 0.0
        for i in range(max_row):
            for j in range(max_col):
                mass = _sum_from_integral(integral, i, j, size)
                if mass > 0.0:
                    p = mass / total_mass
                    corr_sum += p * p

        results[idx] = corr_sum / total_windows

    return results, window_counts

@njit(nogil=True, cache=True)
def get_sparse_coordinates(array):
    """
    Extract coordinates of all non-zero pixels from a 2D array.
    
    This function is used by the sparse optimization algorithm to preprocess
    very sparse arrays by collecting the coordinates of all non-zero pixels.
    This allows the sparse box counting algorithm to work with a compact
    representation rather than scanning the entire array.
    
    Parameters
    ----------
    array : np.ndarray
        2D array to analyze (typically binary, but works with any numeric array)
        
    Returns
    -------
    list
        List of (row, col) tuples representing the coordinates of all non-zero pixels.
        Empty list if no non-zero pixels are found.
        
    Notes
    -----
    This function is compiled with numba for performance. It performs a full
    scan of the array, so it's only beneficial when the resulting coordinate
    list is significantly smaller than the original array (i.e., for very
    sparse arrays).
    
    The coordinate list is used by count_sparse_boxes() to efficiently count
    boxes containing structure without having to check every box in the grid.
    
    See Also
    --------
    count_sparse_boxes : Function that uses these coordinates for box counting
    numba_d0_sparse : Main sparse optimization function
    """
    H, W = array.shape
    coords = []
    
    for i in range(H):
        for j in range(W):
            if array[i, j] > 0:
                coords.append((i, j))
    
    return coords

@njit(nogil=True, cache=True)
def count_sparse_boxes(coords, size, x_off, y_off, H, W):
    """
    Count boxes containing non-zero pixels using sparse coordinate representation.
    
    This function efficiently counts boxes that contain at least one non-zero pixel
    by working with a precomputed list of non-zero pixel coordinates rather than
    scanning the entire array. It's used by the sparse optimization algorithm for
    very sparse arrays.
    
    Parameters
    ----------
    coords : list
        List of (row, col) tuples representing coordinates of non-zero pixels
    size : int
        Size of the boxes (both width and height)
    x_off : int
        X-offset for the box grid starting position
    y_off : int
        Y-offset for the box grid starting position
    H : int
        Height of the original array
    W : int
        Width of the original array
        
    Returns
    -------
    int
        Number of boxes of the given size that contain at least one non-zero pixel
        
    Notes
    -----
    The algorithm works by:
    1. Creating a boolean grid to track which boxes contain structure
    2. For each non-zero pixel coordinate, determining which box it belongs to
    3. Marking that box as containing structure in the boolean grid
    4. Counting the number of marked boxes
    
    This approach is much more efficient than the standard algorithm when the
    array is very sparse, as it only processes pixels that actually contain
    structure rather than scanning every possible box.
    
    The function handles edge cases by only counting boxes that are fully within
    the array bounds and correctly accounts for the grid offset when mapping
    pixel coordinates to box grid coordinates.
    
    See Also
    --------
    get_sparse_coordinates : Function that generates the coordinate list
    numba_d0_sparse : Main sparse optimization function that uses this
    """
    if len(coords) == 0:
        return 0
    
    # Create boolean grid to track unique boxes containing structure
    max_boxes_x = (H // size) + 1
    max_boxes_y = (W // size) + 1
    box_grid = np.zeros((max_boxes_x, max_boxes_y), dtype=np.bool_)
    
    for coord in coords:
        x, y = coord
        # Calculate which box this coordinate belongs to
        box_x = ((x - x_off) // size) * size + x_off
        box_y = ((y - y_off) // size) * size + y_off
        
        # Only count if box is within bounds
        if (box_x >= 0 and box_y >= 0 and 
            box_x + size <= H and box_y + size <= W):
            # Account for offset in grid coordinate calculation
            grid_x = (box_x - x_off) // size
            grid_y = (box_y - y_off) // size
            if grid_x < max_boxes_x and grid_y < max_boxes_y:
                box_grid[grid_x, grid_y] = True
    
    return np.sum(box_grid)

@njit(nogil=True, cache=True)
def _compute_integral_image(array):
    """
    Compute summed-area table (integral image) for fast gliding box sums.

    The returned array has shape (H+1, W+1) where each entry contains the sum of
    all pixels above and to the left of the corresponding position in the input
    array. Using the extra row/column allows O(1) retrieval of any rectangular
    sum, which is critical for lacunarity calculations that require the mass of
    every gliding box position.
    """
    H, W = array.shape
    integral = np.zeros((H + 1, W + 1), dtype=np.float64)

    for i in range(1, H + 1):
        row_sum = 0.0
        for j in range(1, W + 1):
            row_sum += array[i - 1, j - 1]
            integral[i, j] = integral[i - 1, j] + row_sum

    return integral

@njit(nogil=True, cache=True)
def _sum_from_integral(integral, top, left, size):
    """
    Retrieve sum of a size x size block using the integral image.
    """
    bottom = top + size
    right = left + size
    return (
        integral[bottom, right]
        - integral[top, right]
        - integral[bottom, left]
        + integral[top, left]
    )


@njit(nogil=True, parallel=True, cache=True)
def numba_lacunarity_gliding(array, sizes):
    """
    Compute gliding-box lacunarity following Allain & Cloitre (1991).

    Parameters
    ----------
    array : np.ndarray
        2D binary (or positive-valued) array to analyse.
    sizes : np.ndarray
        1D array of integer box sizes to evaluate. Sizes greater than the image
        extent are ignored (returning NaN lacunarity).

    Returns
    -------
    tuple of np.ndarray
        (lacunarity, mean_mass, variance, window_counts) each of length len(sizes).
        Lacunarity values follow Λ(r) = E[M^2] / E[M]^2 where M is the mass
        within a gliding box. Variance is unbiased mass variance across gliding
        positions. Window counts record the number of gliding positions
        contributing at each scale.
    """
    H, W = array.shape
    integral = _compute_integral_image(array)
    n_sizes = len(sizes)
    lacunarity = np.empty(n_sizes, dtype=np.float64)
    mean_mass = np.empty(n_sizes, dtype=np.float64)
    variance = np.empty(n_sizes, dtype=np.float64)
    window_counts = np.zeros(n_sizes, dtype=np.int64)

    for idx in prange(n_sizes):
        size = sizes[idx]
        if size <= 0 or size > H or size > W:
            lacunarity[idx] = np.nan
            mean_mass[idx] = 0.0
            variance[idx] = 0.0
            window_counts[idx] = 0
            continue

        max_row = H - size + 1
        max_col = W - size + 1
        total_windows = max_row * max_col
        window_counts[idx] = total_windows

        sum_mass = 0.0
        sum_mass_sq = 0.0

        for i in range(max_row):
            top = i
            for j in range(max_col):
                left = j
                mass = _sum_from_integral(integral, top, left, size)
                sum_mass += mass
                sum_mass_sq += mass * mass

        if total_windows == 0 or sum_mass <= 0.0:
            lacunarity[idx] = np.nan
            mean_mass[idx] = 0.0
            variance[idx] = 0.0
            continue

        mean_val = sum_mass / total_windows
        second_moment = sum_mass_sq / total_windows
        var_val = second_moment - mean_val * mean_val
        if var_val < 0.0:
            var_val = 0.0

        mean_mass[idx] = mean_val
        variance[idx] = var_val
        lacunarity[idx] = second_moment / (mean_val * mean_val)

    return lacunarity, mean_mass, variance, window_counts

@njit(nogil=True, parallel=True, cache=True)
def numba_lacunarity_boxcount(array, sizes, offsets, use_min=False, conditional=False):
    """
    Compute box-count (non-gliding) lacunarity following Plotnick et al. (1996).

    For each box size, this routine evaluates non-overlapping grids defined by
    the provided offsets and computes lacunarity according to:

        Λ_box(r) = N(r) * Σ n_i^2 / (Σ n_i)^2

    where N(r) is the number of boxes in the partition and n_i the mass per
    box. When `conditional=True`, only occupied boxes (n_i > 0) contribute,
    producing the conditional version described by Plotnick et al. When
    `conditional=False`, all boxes are included (unconditional lacunarity),
    capturing both mass and void statistics.

    Parameters
    ----------
    array : np.ndarray
        2D binary (or positive-valued) array to analyse.
    sizes : np.ndarray
        1D array of integer box sizes to evaluate.
    offsets : np.ndarray
        Pre-generated offsets of shape (len(sizes), num_offsets, 2). Identical
        to the structure used for the D0 kernels.
    use_min : bool, default False
        If True, select the minimum lacunarity across offsets. Otherwise,
        return the mean lacunarity across valid offsets.
    conditional : bool, default False
        If True, compute conditional lacunarity (occupied boxes only).
        Otherwise include empty boxes (unconditional lacunarity).

    Returns
    -------
    tuple of np.ndarray
        (lacunarity, mean_mass, variance, window_counts) where each array has
        length len(sizes). Mean and variance correspond to the set of boxes
        used (occupied-only when conditional=True, otherwise all boxes).
        window_counts reports the number of boxes contributing per size.
    """
    H, W = array.shape
    integral = _compute_integral_image(array)
    min_row, min_col, max_row, max_col = get_bounding_box(array)
    n_sizes = len(sizes)

    lacunarity = np.empty(n_sizes, dtype=np.float64)
    mean_mass = np.empty(n_sizes, dtype=np.float64)
    variance = np.empty(n_sizes, dtype=np.float64)
    window_counts = np.zeros(n_sizes, dtype=np.int64)

    for idx in prange(n_sizes):
        size = sizes[idx]
        if size <= 0 or size > H or size > W:
            lacunarity[idx] = np.nan
            mean_mass[idx] = 0.0
            variance[idx] = 0.0
            window_counts[idx] = 0
            continue

        total_offsets = min(offsets.shape[1], size * size)
        if total_offsets <= 0:
            lacunarity[idx] = np.nan
            mean_mass[idx] = 0.0
            variance[idx] = 0.0
            window_counts[idx] = 0
            continue

        # Aggregation variables
        best_lac = np.inf
        best_mean = 0.0
        best_var = 0.0
        best_windows = 0

        sum_lac = 0.0
        sum_mean = 0.0
        sum_var = 0.0
        sum_windows = 0.0
        valid_offsets = 0

        for offset_idx in range(total_offsets):
            if offset_idx == 0:
                x_off = (H % size) // 2
                y_off = (W % size) // 2
            else:
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size

            max_x = x_off + ((H - x_off) // size) * size
            max_y = y_off + ((W - y_off) // size) * size

            if max_x <= x_off or max_y <= y_off:
                continue

            occ = 0
            total_boxes = 0
            sum_mass = 0.0
            sum_mass_sq = 0.0

            for x in range(x_off, max_x, size):
                for y in range(y_off, max_y, size):
                    total_boxes += 1
                    mass = 0.0
                    if box_intersects_bounds(x, y, size, min_row, min_col, max_row, max_col):
                        mass = _sum_from_integral(integral, x, y, size)

                    if conditional:
                        if mass > 0.0:
                            occ += 1
                            sum_mass += mass
                            sum_mass_sq += mass * mass
                    else:
                        if mass > 0.0:
                            occ += 1
                        sum_mass += mass
                        sum_mass_sq += mass * mass

            boxes_used = occ if conditional else total_boxes

            if boxes_used == 0 or sum_mass <= 0.0:
                continue

            inv_boxes = 1.0 / boxes_used
            mean_val = sum_mass * inv_boxes
            second_moment = sum_mass_sq * inv_boxes
            var_val = second_moment - mean_val * mean_val
            if var_val < 0.0:
                var_val = 0.0
            lac_val = (boxes_used * sum_mass_sq) / (sum_mass * sum_mass)

            valid_offsets += 1

            if use_min:
                if lac_val < best_lac:
                    best_lac = lac_val
                    best_mean = mean_val
                    best_var = var_val
                    best_windows = boxes_used
            else:
                sum_lac += lac_val
                sum_mean += mean_val
                sum_var += var_val
                sum_windows += boxes_used

        if valid_offsets == 0:
            lacunarity[idx] = np.nan
            mean_mass[idx] = 0.0
            variance[idx] = 0.0
            window_counts[idx] = 0
            continue

        if use_min:
            lacunarity[idx] = best_lac
            mean_mass[idx] = best_mean
            variance[idx] = best_var
            window_counts[idx] = best_windows
        else:
            inv = 1.0 / valid_offsets
            lacunarity[idx] = sum_lac * inv
            mean_mass[idx] = sum_mean * inv
            variance[idx] = sum_var * inv
            window_counts[idx] = int(np.round(sum_windows * inv))

    return lacunarity, mean_mass, variance, window_counts

@njit(nogil=True, parallel=True, cache=True)
def numba_d0_sparse(array, sizes, offsets, sparsity_threshold=0.01, use_min_count=False):
    """
    Compute capacity dimension (D0) box counts using sparse coordinate optimization.
    
    This function provides an additional optimization level for very sparse arrays
    by working with a coordinate-based representation of non-zero pixels rather
    than scanning the entire array. It automatically falls back to the bounding
    box optimization if the array is not sparse enough to benefit from this approach.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in pixels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 2) containing pre-generated
        (x_offset, y_offset) pairs for each size
    sparsity_threshold : float, default 0.01
        Sparsity threshold below which to use coordinate-based processing.
        Arrays with sparsity (non_zero_pixels / total_pixels) above this threshold
        will fall back to numba_d0_optimized
    use_min_count : bool, default False
        If True, return minimum count across all offsets for each size.
        If False, return average count across all offsets for each size.
        
    Returns
    -------
    np.ndarray
        1D array of box counts for each size. If use_min_count=True, contains
        minimum counts; if False, contains average counts (may be fractional).
        Results are identical to other D0 functions but computed more efficiently
        for very sparse arrays.
        
    Notes
    -----
    This function implements a three-tier optimization strategy:
    1. Calculates array sparsity (fraction of non-zero pixels)
    2. If sparsity > threshold, falls back to numba_d0_optimized
    3. If sparsity ≤ threshold, uses coordinate-based processing
    
    The sparse algorithm works by:
    1. Extracting coordinates of all non-zero pixels once
    2. For each box size and offset, using count_sparse_boxes() to efficiently
       count boxes containing structure
    3. Combining results across offsets (minimum or average)
    
    Performance characteristics:
    - Most beneficial for arrays with sparsity < 1% (very sparse)
    - Provides diminishing returns as sparsity increases
    - May be slower than bounding box optimization for dense arrays
    - Memory usage scales with number of non-zero pixels
    
    The function handles edge cases including completely empty arrays and
    ensures identical results to other D0 implementations.
    
    See Also
    --------
    numba_d0_optimized : Bounding box optimization (fallback)
    get_sparse_coordinates : Extracts non-zero pixel coordinates
    count_sparse_boxes : Counts boxes using coordinate representation
    """
    results = np.empty(len(sizes), dtype=np.int64)
    H, W = array.shape
    total_pixels = H * W
    non_zero_count = 0
    
    # Count non-zero pixels to determine sparsity
    for i in range(H):
        for j in range(W):
            if array[i, j] > 0:
                non_zero_count += 1
    
    # Fall back to bounding box method if array is too dense
    sparsity = non_zero_count / total_pixels
    if sparsity > sparsity_threshold:
        return numba_d0_optimized(array, sizes, offsets, use_min_count)
    
    # Extract sparse coordinates once for all sizes
    coords = get_sparse_coordinates(array)
    
    if len(coords) == 0:
        results.fill(0)
        return results
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**2)
        
        if use_min_count:
            # Find minimum count across offsets
            min_count = np.inf
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                count = count_sparse_boxes(coords, size, x_off, y_off, H, W)
                
                if count < min_count:
                    min_count = count
            
            results[idx] = min_count if min_count != np.inf else 0
        else:
            # Average count across offsets
            count_sum = 0
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                
                count = count_sparse_boxes(coords, size, x_off, y_off, H, W)
                count_sum += count
            
            results[idx] = count_sum / total_offsets if total_offsets > 0 else 0
    
    return results

def lacunarity_boxcount(array,
                        num_sizes=10,
                        min_size=None,
                        max_size=None,
                        sizes=None,
                        pad_factor=None,
                        pad_kwargs=None,
                        method='gliding',
                        num_offsets=1,
                        use_min=False,
                        seed=None,
                        conditional=False):
    """
    Compute lacunarity using either gliding-box or non-gliding (box-count) methods.

    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyse. Non-binary inputs are cast to float32.
    num_sizes : int, default 10
        Number of box sizes to generate when explicit `sizes` are not provided.
    min_size : int, optional
        Minimum box size. Defaults to 1.
    max_size : int, optional
        Maximum box size. Defaults to min(array.shape).
    sizes : array-like, optional
        Explicit iterable of box sizes to evaluate. When provided, `num_sizes`,
        `min_size`, and `max_size` are ignored.
    pad_factor : float, optional
        Optional padding factor applied via pad_image_for_boxcounting to reduce
        edge artefacts prior to lacunarity calculation. Set to None to disable.
    pad_kwargs : dict, optional
        Additional keyword arguments forwarded to pad_image_for_boxcounting.
    method : {'gliding', 'box'}, default 'gliding'
        Choose between gliding-box lacunarity (Allain & Cloitre, 1991) and
        non-gliding box-count lacunarity (Plotnick et al., 1996).
    num_offsets : int, default 1
        Number of grid offsets to evaluate for the box-count method. Ignored
        when `method='gliding'`.
    use_min : bool, default False
        For the box-count method, return the minimum lacunarity across offsets
        instead of the mean.
    seed : int, optional
        Random seed for reproducible offset generation (box-count method).
    conditional : bool, default False
        When `method='box'`, compute conditional lacunarity (occupied boxes
        only). The default includes empty boxes (unconditional variant).

    Returns
    -------
    tuple
        (sizes, lacunarity, mean_mass, variance, window_counts) where all arrays
        are aligned with the evaluated box sizes. Lacunarity entries may be NaN
        when a scale has no occupied windows.
    """
    if pad_kwargs is None:
        pad_kwargs = {}

    padded_array = array
    if pad_factor is not None:
        max_dim = np.max(array.shape) if max_size is None else max_size
        padded_array = pad_image_for_boxcounting(array, max_dim, pad_factor=pad_factor, **pad_kwargs)

    padded_array = np.ascontiguousarray(padded_array.astype(np.float32))
    H, W = padded_array.shape

    if sizes is not None:
        size_list = sorted({int(s) for s in sizes if s is not None})
    else:
        min_size = 1 if min_size is None else int(min_size)
        max_dim = min(H, W) if max_size is None else min(int(max_size), min(H, W))
        min_clamped = max(1, min_size)
        if max_dim < min_clamped:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
            )
        size_list = get_sizes(num_sizes, min_clamped, max_dim)

    # Filter invalid sizes (<=0 or exceeding current image dimensions)
    valid_sizes = [s for s in size_list if s is not None and s > 0 and s <= min(H, W)]
    if not valid_sizes:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    sizes_arr = np.array(valid_sizes, dtype=np.int32)

    if method == 'gliding':
        lac, mean_mass, var, window_counts = numba_lacunarity_gliding(padded_array, sizes_arr)
    elif method == 'box':
        offsets = generate_random_offsets(sizes_arr, num_offsets, seed=seed)
        lac, mean_mass, var, window_counts = numba_lacunarity_boxcount(
            padded_array,
            sizes_arr,
            offsets,
            use_min=use_min,
            conditional=conditional,
        )
    else:
        raise ValueError("method must be 'gliding' or 'box'")

    return sizes_arr, lac, mean_mass, var, window_counts

def correlation_dimension(array,
                          num_sizes=10,
                          min_size=None,
                          max_size=None,
                          sizes=None,
                          pad_factor=None,
                          pad_kwargs=None,
                          method='box',
                          num_offsets=1,
                          seed=None,
                          use_optimization=True):
    """
    Compute correlation (mass) dimension measures with box-count or gliding windows.

    Parameters
    ----------
    array : np.ndarray
        2D binary or grayscale array.
    num_sizes : int, default 10
        Number of box sizes to generate when explicit `sizes` not provided.
    min_size, max_size : int, optional
        Minimum and maximum box sizes.
    sizes : array-like, optional
        Explicit box sizes.
    pad_factor : float, optional
        Optional padding factor applied via pad_image_for_boxcounting.
    pad_kwargs : dict, optional
        Additional padding kwargs.
    method : {'box', 'gliding'}, default 'box'
        Choose non-overlapping tilings or gliding windows.
    num_offsets : int, default 1
        Number of offsets for box-count method.
    seed : int, optional
        Seed for offset generation.
    use_optimization : bool, default True
        Whether to use optimized correlation kernels for method='box'.

    Returns
    -------
    tuple
        (sizes, correlation_sums, window_counts) where correlation_sums is Σ p_i²
        (averaged across offsets for method='box') and window_counts gives the
        number of windows evaluated at each scale.
    """
    if pad_kwargs is None:
        pad_kwargs = {}

    padded_array = array
    if pad_factor is not None:
        max_dim = np.max(array.shape) if max_size is None else max_size
        padded_array = pad_image_for_boxcounting(array, max_dim, pad_factor=pad_factor, **pad_kwargs)

    padded_array = np.ascontiguousarray(padded_array.astype(np.float32))
    H, W = padded_array.shape

    if sizes is not None:
        size_list = sorted({int(s) for s in sizes if s is not None})
    else:
        min_size = 1 if min_size is None else int(min_size)
        max_dim = min(H, W) if max_size is None else min(int(max_size), min(H, W))
        min_clamped = max(1, min_size)
        if max_dim < min_clamped:
            return (
                np.array([], dtype=np.int32),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.int64),
            )
        size_list = get_sizes(num_sizes, min_clamped, max_dim)

    valid_sizes = [s for s in size_list if s is not None and s > 0 and s <= min(H, W)]
    if not valid_sizes:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    sizes_arr = np.array(valid_sizes, dtype=np.int32)

    if method == 'gliding':
        print('Using gliding method')
        corr, window_counts = numba_d2_gliding(padded_array, sizes_arr)
    elif method == 'box':
        print('Using box method')
        offsets = generate_random_offsets(sizes_arr, num_offsets, seed=seed)
        if use_optimization:
            corr = numba_d2_optimized(padded_array, sizes_arr, offsets)
        else:
            corr = numba_d2(padded_array, sizes_arr, offsets)
        window_counts = (np.floor_divide(H, sizes_arr) * np.floor_divide(W, sizes_arr)).astype(np.int64)
    else:
        raise ValueError("method must be 'box' or 'gliding'")

    return sizes_arr, corr, window_counts

def boxcount(array, mode='D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1, 
             use_optimization=True, sparse_threshold=0.01, use_min_count=True, seed=None):
    """
    Perform box counting analysis with automatic optimization level selection.
    
    This is the main interface function for box counting analysis. It automatically
    selects the most appropriate optimization level based on array characteristics
    and provides a unified interface for both capacity dimension (D0) and information
    dimension (D1) calculations.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze. Will be converted to contiguous float32 format.
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension (box counts)
        - 'D1': Information dimension (entropy values)
    num_sizes : int, default 10
        Number of box sizes to test, distributed geometrically between min_size and max_size
    min_size : int, optional
        Minimum box size in pixels. Defaults to 1 if not specified.
    max_size : int, optional
        Maximum box size in pixels. Defaults to min(array.shape)//5 if not specified.
    num_offsets : int, default 1
        Number of grid offset positions to test for each box size. More offsets
        reduce bias from grid alignment effects but increase computation time.
    use_optimization : bool, default True
        Whether to use optimized algorithms. If True, automatically selects between
        sparse, bounding box, and basic optimizations based on array characteristics.
        If False, uses basic unoptimized algorithms.
    sparse_threshold : float, default 0.01
        Sparsity threshold (fraction of non-zero pixels) below which to use
        sparse coordinate optimization. Only applies when use_optimization=True.
    use_min_count : bool, default False
        For D0 mode only: whether to use minimum count across offsets (True) or
        average count across offsets (False). Averaging is generally recommended.
    seed : int, optional
        Random seed for reproducible grid offset generation. If None, uses
        current random state.
        
    Returns
    -------
    tuple
        (sizes, counts) where:
        - sizes: List of box sizes used in the analysis
        - counts: List of corresponding measures (box counts for D0, entropy values for D1)
        
    Notes
    -----
    Optimization Strategy:
    1. If use_optimization=False: Uses basic numba_d0 or numba_d1
    2. If use_optimization=True and mode='D0':
       - Sparsity ≤ sparse_threshold: Uses numba_d0_sparse
       - Sparsity > sparse_threshold: Uses numba_d0_optimized
    3. If use_optimization=True and mode='D1': Uses numba_d1_optimized
    
    The function handles array preprocessing (type conversion, contiguity),
    size range generation, and offset generation automatically. Box sizes are
    distributed geometrically and adjusted to ensure they are unique integers.
    
    For fractal dimension calculation, pass the returned sizes and counts to
    compute_dimension().
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple binary array
    >>> array = np.random.choice([0, 1], size=(100, 100), p=[0.9, 0.1])
    >>> sizes, counts = boxcount(array, mode='D0', num_sizes=20, num_offsets=10)
    >>> # Compute dimension
    >>> from fracstack.boxcount import compute_dimension
    >>> valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(sizes, counts, mode='D0')
    
    See Also
    --------
    compute_dimension : Compute fractal dimension from box counting results
    dynamic_boxcount : Advanced box counting with automatic scaling range detection
    numba_d0, numba_d0_optimized, numba_d0_sparse : Low-level D0 implementations
    numba_d1, numba_d1_optimized : Low-level D1 implementations
    """
    array = np.ascontiguousarray(array.astype(np.float32))
    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape)//5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    sizes_arr = np.array(sizes)
    
    # Pre-generate random offsets to avoid thread safety issues in numba parallel functions
    offsets = generate_random_offsets(sizes_arr, num_offsets, seed=seed)
    
    if use_optimization:
        # Calculate sparsity to choose optimization strategy
        total_pixels = array.size
        non_zero_pixels = np.count_nonzero(array)
        sparsity = non_zero_pixels / total_pixels
        
        # Use sparse optimization for very sparse arrays (D0 only for now)
        if mode == 'D0' and sparsity <= sparse_threshold:
            counts = numba_d0_sparse(array, sizes_arr, offsets, sparse_threshold, use_min_count)
        elif mode == 'D0':
            counts = numba_d0_optimized(array, sizes_arr, offsets, use_min_count)
        elif mode == 'D1':
            counts = numba_d1_optimized(array, sizes_arr, offsets)
        elif mode == 'D2':
            counts = numba_d2_optimized(array, sizes_arr, offsets)
        else:
            raise ValueError("Invalid mode, use 'D0', 'D1', or 'D2'")
    else:
        # Fall back to original implementation
        if mode == 'D0':
            counts = numba_d0(array, sizes_arr, offsets, use_min_count)
        elif mode == 'D1':
            counts = numba_d1(array, sizes_arr, offsets)
        elif mode == 'D2':
            counts = numba_d2(array, sizes_arr, offsets)
        else:
            raise ValueError("Invalid mode, use 'D0', 'D1', or 'D2'")
    
    return sizes, counts.tolist()

def get_sizes(num_sizes, minsize, maxsize):
    """
    Generate a geometric sequence of box sizes for fractal analysis.
    
    Creates a sequence of box sizes distributed geometrically between minimum and
    maximum values, with adjustments to ensure all sizes are unique integers.
    This distribution is optimal for fractal analysis as it provides good coverage
    across different scales while maintaining approximately equal spacing in log space.
    
    Parameters
    ----------
    num_sizes : int
        Number of box sizes to generate
    minsize : int
        Minimum box size (inclusive)
    maxsize : int
        Maximum box size (inclusive)
        
    Returns
    -------
    list
        List of unique integer box sizes in ascending order. May contain fewer than
        num_sizes elements if the range is too small to accommodate all unique sizes.
        
    Notes
    -----
    The algorithm works by:
    1. Generating a geometric sequence using numpy.geomspace
    2. Rounding to nearest integers
    3. Ensuring all sizes are unique by incrementing duplicates
    4. Truncating if maximum size is reached
    
    The geometric distribution ensures that the ratio between consecutive sizes
    is approximately constant, which is important for fractal analysis where
    we're interested in power-law scaling relationships.
    
    Examples
    --------
    >>> get_sizes(5, 1, 100)
    [1, 3, 10, 32, 100]
    
    >>> get_sizes(10, 2, 5)  # Small range
    [2, 3, 4, 5]  # Returns fewer sizes due to range limitation
    
    See Also
    --------
    boxcount : Main function that uses this to generate size sequences
    """
    sizes = list(np.around(np.geomspace(minsize, maxsize, num_sizes)).astype(int))
    for index in range(1, len(sizes)):
        size = sizes[index]
        prev_size = sizes[index - 1]
        if size <= prev_size:
            sizes[index] = prev_size + 1
            if prev_size == maxsize:
                return sizes[:index]
    return sizes

def compute_dimension(sizes, measures, mode='D0', use_weighted_fit=True, use_bootstrap_ci=True,
                      bootstrap_method='residual', n_bootstrap=1000, alpha=0.05,
                      random_seed=None, embedding_dim=2):
    """
    Compute fractal dimension from box counting results using robust statistical methods.
    
    This function performs the critical step of converting box counting measurements into
    fractal dimension estimates. It uses weighted least squares to address heteroscedasticity
    in log-log scaling data and provides multiple methods for confidence interval estimation,
    including bootstrap methods that don't rely on normality assumptions.
    
    Parameters
    ----------
    sizes : array-like
        Box sizes (scale parameters) used in the box counting analysis
    measures : array-like
        Corresponding measures for each box size:
        - For D0: box counts N(ε)
        - For D1: entropy values H(ε)
        - For D2: correlation sums C(ε) = Σ p_i²
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension from log10(N) vs log10(ε)
        - 'D1': Information dimension from H vs log2(ε)
        - 'D2': Correlation (mass) dimension from log10(C) vs log10(ε)
    use_weighted_fit : bool, default True
        Whether to use weighted least squares (WLS) instead of ordinary least squares (OLS).
        WLS addresses heteroscedasticity where measurement variance increases at finer scales.
    use_bootstrap_ci : bool, default True
        Whether to use bootstrap confidence intervals instead of traditional t-based intervals.
        Bootstrap methods are more robust and don't assume normality of residuals.
    bootstrap_method : str, default 'residual'
        Bootstrap method to use:
        - 'residual': Residual bootstrap (recommended) - resamples residuals while keeping
          x-values fixed, preserving the structure of the regression
        - 'offset': Not yet implemented - would resample offset positions
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 gives 95% CI)
    random_seed : int, optional
        Random seed for bootstrap reproducibility. If None, uses current random state.
    embedding_dim : float, default 2
        Embedding dimension correction subtracted from the fitted slope when
        computing D2. For gliding-window correlation sums in 2D, pass 2.0.
        
    Returns
    -------
    tuple
        (valid_sizes, valid_measures, d_value, fit, r2, ci_low, ci_high) where:
        - valid_sizes : np.ndarray - Box sizes after filtering out non-positive values
        - valid_measures : np.ndarray - Measures after filtering out non-positive values
        - d_value : float - Computed fractal dimension (-slope of the fit)
        - fit : np.ndarray - Linear fit parameters [slope, intercept]
        - r2 : float - R-squared value indicating goodness of fit
        - ci_low : float - Lower bound of confidence interval for dimension
        - ci_high : float - Upper bound of confidence interval for dimension
        
    Notes
    -----
    Statistical Methods:
    
    Weighted Least Squares (WLS):
    - For D0: Uses box counts N(ε) as weights, addressing the fact that variance of
      log(N) increases as ε decreases (heteroscedasticity)
    - For D1: Uses inverse entropy values as approximate weights
    - Provides more reliable estimates than OLS for fractal data
    
    Bootstrap Confidence Intervals:
    - Residual bootstrap resamples residuals while keeping x-values fixed
    - More robust than t-based intervals, especially for small samples
    - Doesn't assume normality of residuals or homoscedasticity
    - Provides better coverage probabilities for fractal dimension estimates
    
    The fractal dimension is computed as:
    - D0: -slope of log10(N(ε)) vs log10(ε)
    - D1: -slope of H(ε) vs log2(ε)
    
    Edge Cases:
    - Returns NaN values if insufficient valid data points (< 3 points)
    - Filters out non-positive sizes and measures before fitting
    - Falls back to OLS if WLS matrix inversion fails
    
    Examples
    --------
    >>> sizes = [1, 2, 4, 8, 16, 32]
    >>> counts = [1000, 250, 63, 16, 4, 1]  # Approximate D=2 scaling
    >>> valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(
    ...     sizes, counts, mode='D0', use_bootstrap_ci=True)
    >>> print(f"Dimension: {d_value:.3f} ± {(ci_high-ci_low)/2:.3f}")
    
    See Also
    --------
    bootstrap_residual : Residual bootstrap implementation
    bootstrap_dimension : Standard bootstrap for dimension estimation
    boxcount : Function that generates the input sizes and measures
    """
    
    # Convert to numpy arrays if not already
    sizes = np.array(sizes)
    measures = np.array(measures)
    
    # Filter out non-positive sizes and counts
    valid = (sizes > 0) & (measures > 0)
    valid_sizes = sizes[valid]
    valid_measures = measures[valid]
    
    if len(valid_sizes) > 2 and len(valid_measures) > 2:  # Need at least 3 points for WLS
        if use_weighted_fit:
            # Weighted Least Squares (WLS) approach
            if mode == 'D0':
                # For D0: x = log10(ε), y = log10(N(ε)), w = N(ε)
                x = np.log10(valid_sizes)
                y = np.log10(valid_measures)
                w = valid_measures # Weight proportional to N(ε)
                
                # Weighted least squares: (X^T W X)^{-1} X^T W y
                # where X = [x, ones], W = diag(w)
                X = np.vstack([x, np.ones_like(x)]).T  # Design matrix
                W = np.diag(w)  # Weight matrix
                
                # Compute weighted coefficients
                XtWX = X.T @ W @ X
                XtWy = X.T @ W @ y
                
                try:
                    beta = np.linalg.solve(XtWX, XtWy)
                    slope, intercept = beta
                    fit = np.array([slope, intercept])
                    

                    y_pred = slope * x + intercept
                    
                    residuals = y - y_pred
                    if use_weighted_fit:
                        residuals -= np.average(residuals, weights=w)
                    else:
                        residuals -= residuals.mean()
                        
                    y_mean_weighted = np.sum(w * y) / np.sum(w)  # Weighted mean
                    ss_res = np.sum(w * (y - y_pred)**2)  # Weighted residual sum of squares
                    ss_tot = np.sum(w * (y - y_mean_weighted)**2)  # Weighted total sum of squares
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Compute covariance matrix for confidence intervals
                    # Cov(β) = (X^T W X)^{-1} * σ²
                    # For weighted regression, σ² is estimated from weighted residuals
                    
                    n = len(x)   
                    sigma_sq = np.sum(w * residuals**2) / (n - 2)  # Weighted MSE
                    cov = np.linalg.inv(XtWX) * sigma_sq
                    
                except np.linalg.LinAlgError:
                    # Fallback to OLS if WLS fails
                    fit, cov = np.polyfit(x, y, 1, cov=True)
                    r2 = r2_score(y, fit[0] * x + fit[1])
                    
            elif mode == 'D1':
                # For D1: x = log2(ε), y = H(ε), w = N(ε) (need box counts for weights)
                # Since measures are entropy values, we need to approximate weights
                # Use inverse of measures as weights (entropy is roughly proportional to log(N))
                x = np.log2(valid_sizes)
                y = valid_measures
                w = 1.0 / (valid_measures + 1e-10)  # Small constant to avoid division by zero
                
                # Weighted least squares
                X = np.vstack([x, np.ones_like(x)]).T
                W = np.diag(w)
                
                try:
                    XtWX = X.T @ W @ X
                    XtWy = X.T @ W @ y
                    beta = np.linalg.solve(XtWX, XtWy)
                    slope, intercept = beta
                    fit = np.array([slope, intercept])
                    
                    # Compute R² for weighted fit
                    y_pred = slope * x + intercept
                    y_mean_weighted = np.sum(w * y) / np.sum(w)
                    ss_res = np.sum(w * (y - y_pred)**2)
                    ss_tot = np.sum(w * (y - y_mean_weighted)**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Compute covariance matrix
                    n = len(x)
                    residuals = y - y_pred
                    if use_weighted_fit:
                        residuals -= np.average(residuals, weights=w)
                    else:
                        residuals -= residuals.mean()
                        
                    sigma_sq = np.sum(w * residuals**2) / (n - 2)
                    cov = np.linalg.inv(XtWX) * sigma_sq
                    
                except np.linalg.LinAlgError:
                    # Fallback to OLS if WLS fails
                    fit, cov = np.polyfit(x, y, 1, cov=True)
                    r2 = r2_score(y, fit[0] * x + fit[1])
                    
            elif mode == 'D2':
                x = np.log10(valid_sizes)
                y = np.log10(valid_measures)
                w = np.ones_like(x)
                
                X = np.vstack([x, np.ones_like(x)]).T
                W = np.diag(w)
                
                try:
                    XtWX = X.T @ W @ X
                    XtWy = X.T @ W @ y
                    beta = np.linalg.solve(XtWX, XtWy)
                    slope, intercept = beta
                    fit = np.array([slope, intercept])
                    
                    y_pred = slope * x + intercept
                    y_mean = np.sum(w * y) / np.sum(w)
                    ss_res = np.sum(w * (y - y_pred)**2)
                    ss_tot = np.sum(w * (y - y_mean)**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    n = len(x)
                    residuals = y - y_pred
                    if use_weighted_fit:
                        residuals -= np.average(residuals, weights=w)
                    else:
                        residuals -= residuals.mean()
                    sigma_sq = np.sum(w * residuals**2) / (n - 2)
                    cov = np.linalg.inv(XtWX) * sigma_sq
                except np.linalg.LinAlgError:
                    fit, cov = np.polyfit(x, y, 1, cov=True)
                    r2 = r2_score(y, fit[0] * x + fit[1])
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'D0', 'D1', or 'D2'")
        else:
            # Original OLS approach (for comparison)
            if mode == 'D0':
                fit, cov = np.polyfit(np.log10(valid_sizes), np.log10(valid_measures), 1, cov=True)
                r2 = r2_score(np.log10(valid_measures), fit[0] * np.log10(valid_sizes) + fit[1])
            elif mode == 'D1':
                fit, cov = np.polyfit(np.log2(valid_sizes), valid_measures, 1, cov=True)
                r2 = r2_score(valid_measures, fit[0] * np.log2(valid_sizes) + fit[1])
            elif mode == 'D2':
                fit, cov = np.polyfit(np.log10(valid_sizes), np.log10(valid_measures), 1, cov=True)
                r2 = r2_score(np.log10(valid_measures), fit[0] * np.log10(valid_sizes) + fit[1])
            else:
                raise ValueError(f"Invalid mode: {mode}. Use 'D0', 'D1', or 'D2'")
        
        if mode in ('D0', 'D1'):
            d_value = -fit[0]
        elif mode == 'D2':
            d_value = fit[0] - embedding_dim
        else:
            d_value = -fit[0]

        # --- Compute confidence intervals ---
        if use_bootstrap_ci and bootstrap_method == 'residual':
            # Residual Bootstrap Confidence Intervals
            _, ci_low, ci_high = bootstrap_residual(
                valid_sizes, valid_measures, mode=mode, 
                n_bootstrap=n_bootstrap, alpha=alpha, 
                use_weighted_fit=use_weighted_fit, random_seed=random_seed,
                embedding_dim=embedding_dim
            )
            
        elif use_bootstrap_ci and bootstrap_method == 'offset':
            # Offset bootstrap will be implemented later
            raise NotImplementedError("Offset bootstrap not yet implemented. Use bootstrap_method='residual'")
            
        else:
            # Traditional t-based confidence intervals
            # 1) Number of points and degrees of freedom
            N = len(valid_sizes)
            dof = N - 2  # for a linear fit with two parameters

            # 2) Critical t-value
            t_crit = t.ppf(1 - alpha/2, dof)  # two-sided

            # 3) Standard errors for slope (fit[0]) and intercept (fit[1])
            slope_se = np.sqrt(cov[0, 0])
            intercept_se = np.sqrt(cov[1, 1])

            # 4) Confidence intervals
            slope_ci = (fit[0] - t_crit * slope_se, fit[0] + t_crit * slope_se)
            #intercept_ci = (fit[1] - t_crit * intercept_se, fit[1] + t_crit * intercept_se)

            # 5) If you want the CI for d_value = -slope, just flip signs and ensure lower value comes first
            if mode in ('D0', 'D1'):
                ci_low =  min(-slope_ci[1], -slope_ci[0])
                ci_high = max(-slope_ci[1], -slope_ci[0])
            elif mode == 'D2':
                ci_low = slope_ci[0] - embedding_dim
                ci_high = slope_ci[1] - embedding_dim
            else:
                ci_low =  min(-slope_ci[1], -slope_ci[0])
                ci_high = max(-slope_ci[1], -slope_ci[0])
    
    else:
        # Handle cases with insufficient data
        valid_sizes = np.array([])
        valid_measures = np.array([])
        d_value = np.nan
        fit = np.array([np.nan, np.nan])
        r2 = np.nan
        ci_low = np.nan
        ci_high = np.nan
    
    return valid_sizes, valid_measures, d_value, fit, r2, ci_low, ci_high

def bootstrap_residual(
        sizes,
        measures,
        mode: str = "D0",
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        use_weighted_fit: bool = True,
        random_seed: Optional[int] = None,
        studentized: bool = True,
        embedding_dim: float = 2.0,
    ):
    """
    Compute bootstrap confidence intervals for fractal dimensions using residual resampling.
    
    The residual bootstrap is a powerful non-parametric method for estimating confidence
    intervals in regression problems. It's particularly well-suited for fractal dimension
    estimation because it doesn't assume normality of residuals or homoscedasticity, which
    are often violated in log-log scaling data. This method preserves the structure of the
    regression relationship while accounting for uncertainty in the residuals.
    
    Parameters
    ----------
    sizes : array-like
        Box sizes ε used in the scaling analysis
    measures : array-like
        Corresponding measures (box counts N for D0, entropy values H for D1)
    mode : str, default "D0"
        Type of dimension analysis:
        - "D0": Capacity dimension using log10(N) vs log10(ε)
        - "D1": Information dimension using H vs log2(ε)
        - "D2": Correlation dimension using log10(C) vs log10(ε)
    n_bootstrap : int, default 1000
        Number of bootstrap resamples. More resamples give more precise confidence
        intervals but increase computation time.
    alpha : float, default 0.05
        Significance level for confidence intervals. Default 0.05 gives 95% CI.
    use_weighted_fit : bool, default True
        Whether to use weighted least squares (recommended) or ordinary least squares.
        WLS addresses heteroscedasticity common in fractal scaling data.
    random_seed : int, optional
        Random seed for reproducible bootstrap results. If None, uses current random state.
    studentized : bool, default True
        Whether to use studentized (t-type) confidence intervals. If True, uses the
        bootstrap distribution of t-statistics; if False, uses percentile method.
        Studentized intervals generally have better coverage properties.
    embedding_dim : float, default 0
        Embedding dimension correction to subtract from the fitted slope when
        mode='D2'. Use 2.0 for gliding estimates in 2D.
        
    Returns
    -------
    tuple
        (d_hat, ci_low, ci_high) where:
        - d_hat : float - Point estimate of fractal dimension
        - ci_low : float - Lower bound of confidence interval
        - ci_high : float - Upper bound of confidence interval
        
    Notes
    -----
    Theory and Method:
    
    The residual bootstrap works by:
    1. Fitting the original regression model to get fitted values and residuals
    2. Resampling residuals with replacement to create new datasets
    3. Adding resampled residuals to fitted values: y* = ŷ + ε*
    4. Refitting the model to each bootstrap sample
    5. Computing confidence intervals from the bootstrap distribution
    
    Advantages over traditional t-based confidence intervals:
    - Doesn't assume normality of residuals
    - Doesn't assume homoscedasticity (constant variance)
    - More robust for small sample sizes
    - Better coverage probabilities for non-normal data
    - Naturally handles weighted regression
    
    Studentized vs Percentile Bootstrap:
    - Studentized: Uses bootstrap distribution of t-statistics, generally more accurate
    - Percentile: Uses direct percentiles of bootstrap estimates, simpler but less robust
    
    The method is particularly important for fractal analysis because:
    - Log-log scaling data often violates normality assumptions
    - Measurement errors can be heteroscedastic across scales
    - Sample sizes are often small (10-50 data points)
    - Traditional confidence intervals may be unreliable
    
    Examples
    --------
    >>> sizes = [1, 2, 4, 8, 16, 32]
    >>> counts = [1000, 250, 63, 16, 4, 1]
    >>> d_est, ci_low, ci_high = bootstrap_residual(sizes, counts, mode="D0", n_bootstrap=1000)
    >>> print(f"D0 = {d_est:.3f} [{ci_low:.3f}, {ci_high:.3f}]")
    
    See Also
    --------
    bootstrap_dimension : Standard bootstrap resampling data points
    compute_dimension : Main function that calls this for bootstrap CI
    """
    rng = np.random.default_rng(random_seed)

    sizes = np.asarray(sizes, dtype=float)
    measures = np.asarray(measures, dtype=float)

    # Prepare x, y and weights for the chosen mode
    if mode == "D0":
        x = np.log10(sizes)
        y = np.log10(measures)
        w = measures if use_weighted_fit else np.ones_like(x)
    elif mode == "D1":
        x = np.log2(sizes)
        y = measures
        w = 1.0 / (measures + 1e-10) if use_weighted_fit else np.ones_like(x)
    elif mode == "D2":
        x = np.log10(sizes)
        y = np.log10(measures)
        w = np.ones_like(x)
    else:
        raise ValueError("mode must be 'D0', 'D1', or 'D2'")

    # Small jitter to avoid singular XtWX if duplicate x values exist
    x = x + 1e-12 * rng.standard_normal(x.shape)

    X = np.column_stack((x, np.ones_like(x)))          # design matrix

    def _wls(Xmat, yvec, weight):
        """Return beta, residuals, cov, sigma² in fast diagonal-weight form."""
        WX = weight[:, None] * Xmat                    # (n,2)
        XtWX = Xmat.T @ WX
        XtWy = Xmat.T @ (weight * yvec)
        beta = np.linalg.solve(XtWX, XtWy)             # slope, intercept
        resid = yvec - Xmat @ beta
        # Unbiased sigma² under 'known weights' assumption
        sigma2 = np.sum(weight * resid**2) / (len(yvec) - 2)
        cov = np.linalg.inv(XtWX) * sigma2
        return beta, resid, cov, sigma2

    # Fit original model
    if use_weighted_fit:
        beta_hat, resid_hat, cov_hat, sigma2_hat = _wls(X, y, w)
    else:
        beta_hat = np.polyfit(x, y, 1)
        resid_hat = y - (beta_hat[0] * x + beta_hat[1])
        sigma2_hat = resid_hat.var(ddof=2)
        # OLS covariance via (XᵀX)⁻¹ σ²
        cov_hat = np.linalg.inv(X.T @ X) * sigma2_hat

    slope_hat, intercept_hat = beta_hat
    se_hat = np.sqrt(cov_hat[0, 0])                    # s.e. of slope
    y_hat = X @ beta_hat

    if mode in ("D0", "D1"):
        sign = -1.0
        correction = 0.0
    else:  # D2 or other positive-slope modes
        sign = 1.0
        correction = embedding_dim

    d_hat = sign * slope_hat - correction

    # Bootstrap loop
    slopes   = np.empty(n_bootstrap)
    t_stats  = np.empty(n_bootstrap) if studentized else None

    for b in range(n_bootstrap):
        # Resample residuals with replacement
        resampled = rng.choice(resid_hat, size=resid_hat.size, replace=True)
        y_star = y_hat + resampled

        try:
            if use_weighted_fit:
                beta_b, _, cov_b, _ = _wls(X, y_star, w)
            else:
                beta_b = np.polyfit(x, y_star, 1)
                cov_b  = np.linalg.inv(X.T @ X) * \
                         np.sum((y_star - X @ beta_b)**2) / (len(y_star) - 2)
        except np.linalg.LinAlgError:                  # rare singular fallback
            beta_b = np.polyfit(x + 1e-10, y_star, 1)
            cov_b  = np.linalg.inv(X.T @ X) * \
                     np.sum((y_star - X @ beta_b)**2) / (len(y_star) - 2)

        slope_b = beta_b[0]
        se_b = np.sqrt(cov_b[0, 0])

        slopes[b] = slope_b
        if studentized:
            t_stats[b] = (slope_b - slope_hat) / se_b

    # Compute confidence intervals
    if studentized:
        # Studentized bootstrap confidence intervals
        q_low, q_high = np.quantile(t_stats, [1 - alpha/2, alpha/2])
        slope_low  = slope_hat - q_low  * se_hat
        slope_high = slope_hat - q_high * se_hat
    else:
        # Percentile bootstrap confidence intervals
        slope_low, slope_high = np.percentile(slopes, [100*(alpha/2),
                                                       100*(1-alpha/2)])

    if mode in ("D0", "D1"):
        ci_low = -slope_high
        ci_high = -slope_low
    else:
        ci_low = slope_low - embedding_dim
        ci_high = slope_high - embedding_dim

    return d_hat, ci_low, ci_high

def bootstrap_dimension(sizes, measures, mode='D0', n_bootstrap=1000, alpha=0.05,
                        embedding_dim=2.0):
    """
    Compute fractal dimension confidence intervals using standard bootstrap resampling.
    
    This function implements the standard (non-parametric) bootstrap method for estimating
    confidence intervals of fractal dimensions. Unlike residual bootstrap, this method
    resamples the original data points (sizes, measures) with replacement, which can be
    useful when the regression model assumptions are more severely violated or when you
    want to account for uncertainty in both x and y variables.
    
    Parameters
    ----------
    sizes : array-like
        Box sizes used in the scaling analysis
    measures : array-like
        Corresponding measures (box counts for D0, entropy values for D1, correlation sums for D2)
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension using log10(N) vs log10(ε)
        - 'D1': Information dimension using H vs log2(ε)
        - 'D2': Correlation dimension using log10(C) vs log10(ε)
    n_bootstrap : int, default 1000
        Number of bootstrap resamples. More resamples give more stable confidence
        intervals but increase computation time.
    alpha : float, default 0.05
        Significance level for confidence intervals. Default 0.05 gives 95% CI.
    embedding_dim : float, default 2
        Embedding dimension correction to subtract from slopes when mode='D2'.
        
    Returns
    -------
    tuple
        (d_median, d_lower, d_upper) where:
        - d_median : float - Median fractal dimension over all bootstrap samples
        - d_lower : float - Lower bound of confidence interval
        - d_upper : float - Upper bound of confidence interval
        
    Notes
    -----
    Standard Bootstrap vs Residual Bootstrap:
    
    Standard Bootstrap (this function):
    - Resamples (x,y) pairs with replacement
    - Accounts for uncertainty in both box sizes and measures
    - More robust when regression assumptions are severely violated
    - Can handle cases where x-values have measurement error
    - May be less efficient than residual bootstrap for well-behaved data
    
    Residual Bootstrap (bootstrap_residual):
    - Resamples residuals while keeping x-values fixed
    - Assumes x-values are known exactly
    - More efficient for typical regression problems
    - Better preserves the structure of the regression relationship
    - Generally preferred for fractal dimension estimation
    
    When to Use Standard Bootstrap:
    - When box sizes have significant measurement uncertainty
    - When the regression relationship is highly non-linear
    - When residual bootstrap assumptions are violated
    - For exploratory analysis or robustness checks
    
    The method uses ordinary least squares for simplicity and speed, making it
    suitable for quick confidence interval estimation. For more sophisticated
    statistical analysis, consider using compute_dimension with residual bootstrap.
    
    Examples
    --------
    >>> sizes = [1, 2, 4, 8, 16, 32]
    >>> counts = [1000, 250, 63, 16, 4, 1]
    >>> d_med, d_low, d_high = bootstrap_dimension(sizes, counts, mode='D0', n_bootstrap=1000)
    >>> print(f"D0 = {d_med:.3f} [{d_low:.3f}, {d_high:.3f}]")
    
    See Also
    --------
    bootstrap_residual : Residual bootstrap method (generally preferred)
    compute_dimension : Main function with multiple bootstrap options
    """
    sizes = np.array(sizes)
    measures = np.array(measures)
    n = len(sizes)

    # Store dimension from each bootstrap iteration
    d_values = []

    for _ in range(n_bootstrap):
        # Randomly resample data points with replacement
        resample_idx = np.random.randint(0, n, size=n)
        resampled_sizes = sizes[resample_idx]
        resampled_measures = measures[resample_idx]

        # Fit slope based on the chosen mode using OLS
        if mode == 'D0':
            slope, _ = np.polyfit(np.log10(resampled_sizes), np.log10(resampled_measures), 1)
            d_values.append(-slope)
        elif mode == 'D1':
            slope, _ = np.polyfit(np.log2(resampled_sizes), resampled_measures, 1)
            d_values.append(-slope)
        elif mode == 'D2':
            slope, _ = np.polyfit(np.log10(resampled_sizes), np.log10(resampled_measures), 1)
            d_values.append(slope - embedding_dim)
        else:
            raise ValueError("Invalid mode. Use 'D0', 'D1', or 'D2'.")

    # Compute statistics from bootstrap distribution
    d_values = np.array(d_values)
    d_median = np.median(d_values)
    d_lower = np.percentile(d_values, 100 * (alpha / 2))
    d_upper = np.percentile(d_values, 100 * (1 - alpha / 2))

    return d_median, d_lower, d_upper

def dynamic_boxcount(array, 
                     mode='D0', 
                     global_min_size=2, 
                     global_max_size=None, 
                     min_decade_span=1.0, 
                     min_points=10,
                     num_sizes=100, 
                     num_offsets=50,
                     show_progress=True, 
                     stretch=False, 
                     min_R2=0.9985, 
                     pad_factor=1.5,
                     use_plateau_detection=False, 
                     plateau_window=4, 
                     plateau_tol=0.01,
                     plateau_method='pairwise_first', 
                     pairwise_tol=0.01,
                     use_occupancy_filter=False, 
                     occ_low=0.05, 
                     occ_high=0.95, 
                     verbose=False,
                     use_min_count=False, 
                     seed=None, 
                     use_bootstrap_ci=False,
                     bootstrap_method='residual', 
                     n_bootstrap=1000, 
                     bootstrap_seed=None,
                     alpha=0.05):
    """
    Advanced box counting analysis with automatic optimal scaling range detection.
    
    This function addresses a critical challenge in fractal analysis: determining the
    optimal range of box sizes for dimension calculation. Rather than using the entire
    available range, it intelligently searches for the size range that exhibits the
    strongest linear scaling relationship, providing more reliable and accurate fractal
    dimension estimates.
    
    The function offers multiple strategies for range selection, including automatic
    plateau detection, occupancy filtering, and optimization for either highest R² or
    largest scaling range meeting quality criteria.
    
    Parameters
    ----------
    array : np.ndarray
        2D binary array to analyze. Will be padded and converted to contiguous float32.
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension (box counts)
        - 'D1': Information dimension (entropy values)
    global_min_size : int, default 2
        Absolute minimum box size to consider in the analysis
    global_max_size : int, optional
        Absolute maximum box size to consider. Defaults to min(array.shape)//4.
    min_decade_span : float, default 1.0
        Minimum span in orders of magnitude (log10 units) for a valid scaling range.
        1.0 means the range must span at least one decade (10x size ratio).
    min_points : int, default 10
        Minimum number of data points required for reliable dimension fitting
    num_sizes : int, default 100
        Number of box sizes to sample across the global range for dense sampling
    num_offsets : int, default 50
        Number of grid offset positions to test for each box size
    show_progress : bool, default True
        Whether to display progress bars during computation
    stretch : bool, default False
        Optimization strategy:
        - False: Find range with highest R² (quality optimization)
        - True: Find largest range meeting min_R2 threshold (range optimization)
    min_R2 : float, default 0.9985
        Minimum R² threshold for stretch mode. Only ranges with R² ≥ min_R2
        are considered when stretch=True.
    pad_factor : float, default 1.5
        Padding factor for edge effect mitigation. Array is padded to
        pad_factor * max_box_size to reduce boundary effects.
    use_plateau_detection : bool, default False
        Whether to use automatic plateau detection to find optimal range.
        If successful, bypasses the sliding window search for efficiency.
    plateau_window : int, default 4
        Minimum window size for plateau detection algorithms
    plateau_tol : float, default 0.01
        Relative tolerance for plateau detection as fraction of median slope magnitude
    plateau_method : str, default 'pairwise_first'
        Plateau detection strategy:
        - 'pairwise_first': Try pairwise method first, fallback to median
        - 'median_first': Try median method first, fallback to pairwise
        - 'intersection': Use intersection of both methods
        - 'longest': Use whichever method gives longer plateau
    pairwise_tol : float, default 0.01
        Relative tolerance for pairwise slope difference detection
    use_occupancy_filter : bool, default False
        Whether to filter out scales with extreme occupancy rates
    occ_low : float, default 0.05
        Lower occupancy threshold - scales with occupancy ≤ occ_low are removed
    occ_high : float, default 0.95
        Upper occupancy threshold - scales with occupancy ≥ occ_high are removed
    verbose : bool, default False
        Whether to print detailed information about filtering and plateau detection
    use_min_count : bool, default False
        For D0 mode: whether to use minimum count across offsets (True) or
        average count across offsets (False, recommended)
    seed : int, optional
        Random seed for reproducible grid offset generation
    use_bootstrap_ci : bool, default False
        Whether to use bootstrap confidence intervals for dimension estimates
    bootstrap_method : str, default 'residual'
        Bootstrap method - 'residual' or 'standard'
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility
    alpha : float, default 0.05
        Significance level for confidence intervals (0.05 gives 95% CI)
        
    Returns
    -------
    dict
        Comprehensive analysis results containing:
        
        Core Results:
        - 'optimal_range' : tuple - (min_size, max_size) of best scaling range
        - 'D_value' : float - Fractal dimension in optimal range
        - 'R2' : float - R-squared value in optimal range
        - 'optimal_sizes' : np.ndarray - Box sizes used in optimal fit
        - 'optimal_measures' : np.ndarray - Measures used in optimal fit
        - 'decade_span' : float - Span of optimal range in log10 units
        
        Confidence Intervals (if use_bootstrap_ci=True):
        - 'confidence_interval' : tuple - (ci_low, ci_high)
        - 'ci_low' : float - Lower confidence bound
        - 'ci_high' : float - Upper confidence bound
        - 'ci_type' : str - Type of confidence interval used
        
        Analysis Details:
        - 'all_results' : list - All tested ranges with their metrics
        - 'qualified_results' : list - Ranges meeting quality criteria
        - 'global_sizes' : np.ndarray - All sizes tested globally
        - 'global_measures' : np.ndarray - All measures from global box counting
        - 'num_candidates_tested' : int - Total number of ranges tested
        - 'num_qualified' : int - Number of ranges meeting criteria
        
        Metadata:
        - 'analysis_type' : str - Type of analysis performed
        - 'optimization_mode' : str - Optimization strategy used
        - 'min_R2_threshold' : float - R² threshold used (stretch mode)
        - 'threshold_met' : bool - Whether threshold was met
        - 'fallback_mode' : bool - Whether fallback was used
        
    Notes
    -----
    Algorithm Overview:
    
    1. **Global Box Counting**: Performs dense sampling across the full size range
    2. **Optional Filtering**: Applies occupancy filtering to remove saturated scales
    3. **Plateau Detection**: Optionally uses automatic plateau detection for efficiency
    4. **Sliding Window Search**: Systematically tests all possible size ranges
    5. **Optimization**: Selects optimal range based on chosen strategy
    
    Optimization Strategies:
    
    - **Quality Mode** (stretch=False): Finds range with highest R² value
    - **Range Mode** (stretch=True): Finds largest range with R² ≥ min_R2
    
    Advanced Features:
    
    - **Plateau Detection**: Automatically identifies stable scaling regions
    - **Occupancy Filtering**: Removes under-sampled and over-sampled scales
    - **Bootstrap Confidence Intervals**: Provides uncertainty estimates
    - **Edge Effect Mitigation**: Padding reduces boundary artifacts
    
    Performance Considerations:
    
    - Uses optimized numba functions for maximum speed
    - Plateau detection can significantly reduce computation time
    - Sliding window search scales as O(n²) with number of sizes
    - Bootstrap confidence intervals add computational overhead
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a fractal-like pattern
    >>> array = np.random.choice([0, 1], size=(512, 512), p=[0.7, 0.3])
    >>> 
    >>> # Basic usage - find highest R² range
    >>> result = dynamic_boxcount(array, mode='D0', show_progress=True)
    >>> print(f"Optimal range: {result['optimal_range']}")
    >>> print(f"Dimension: {result['D_value']:.3f}, R²: {result['R2']:.4f}")
    >>> 
    >>> # Advanced usage - find largest high-quality range
    >>> result = dynamic_boxcount(array, mode='D0', stretch=True, min_R2=0.999,
    ...                          use_plateau_detection=True, use_bootstrap_ci=True)
    >>> print(f"Range: {result['optimal_range']}, Span: {result['decade_span']:.2f} decades")
    >>> print(f"D0: {result['D_value']:.3f} ± {(result['ci_high']-result['ci_low'])/2:.3f}")
    
    See Also
    --------
    boxcount : Basic box counting interface
    compute_dimension : Dimension computation with statistical methods
    detect_scaling_plateau : Plateau detection algorithm
    filter_by_occupancy : Occupancy-based scale filtering
    """
    
    # Apply padding if specified
    if pad_factor is not None and pad_factor > 1.0:
        # Calculate max_size for padding if not already set
        padding_max_size = global_max_size if global_max_size is not None else min(array.shape) // 4
        array = pad_image_for_boxcounting(array, padding_max_size, pad_factor=pad_factor)
    
    # Prepare array for numba functions
    array = np.ascontiguousarray(array.astype(np.float32))
    
    # Set reasonable defaults
    if global_max_size is None:
        global_max_size = min(array.shape) // 4
        
    # Step 1: Global box counting with dense sampling using existing numba functions
    if show_progress:
        print(f"Performing global box counting from {global_min_size} to {global_max_size}...")
    
    sizes = get_sizes(num_sizes, global_min_size, global_max_size)
    sizes_arr = np.array(sizes)
    
    # Pre-generate random offsets to avoid thread safety issues in numba parallel functions
    offsets = generate_random_offsets(sizes_arr, num_offsets, seed=seed)
    
    # Use optimized numba functions for performance by default
    if mode == 'D0':
        counts = numba_d0_optimized(array, sizes_arr, offsets, use_min_count)
    elif mode == 'D1':
        counts = numba_d1_optimized(array, sizes_arr, offsets)
    else:
        raise ValueError("Invalid mode, use 'D0' or 'D1'")
    
    # Convert to numpy arrays and filter positive values
    sizes = np.array(sizes)
    measures = np.array(counts) # Use counts as measures for D0
    valid_mask = (sizes > 0) & (measures > 0)
    sizes = sizes[valid_mask]
    measures = measures[valid_mask]
    
    if len(sizes) < min_points:
        raise ValueError(f"Insufficient valid data points: {len(sizes)} < {min_points}")
    
    # Step 2: Occupancy filtering (if enabled)
    if use_occupancy_filter:
        if show_progress:
            print(f"Applying occupancy filter (occ_low={occ_low}, occ_high={occ_high})...")
        
        # Save original data for fallback
        sizes_original = sizes.copy()
        measures_original = measures.copy()
        original_count = len(sizes)
        
        # For D1 mode, we need to get the actual box counts for occupancy filtering
        # The measures for D1 are entropy values, but occupancy is based on box counts
        if mode == 'D1':
            # Get box counts for occupancy calculation
            # dynamic_boxcount always uses optimized functions internally
            # Generate offsets for the filtered sizes
            filtered_offsets = generate_random_offsets(sizes, num_offsets, seed=seed)
            box_counts = numba_d0_optimized(array, sizes, filtered_offsets, use_min_count)
            
            if verbose:
                # Show detailed occupancy calculations for D1 mode
                H, W = array.shape
                grid_boxes = (np.floor_divide(H, sizes) * np.floor_divide(W, sizes))
                occupancy = box_counts / grid_boxes
                
                print(f"  D1 Mode: Using box counts for occupancy calculation")
                print(f"  Original scales: {len(sizes)} sizes from {sizes[0]:.1f} to {sizes[-1]:.1f}")
                print(f"  Occupancy range: {occupancy.min():.4f} to {occupancy.max():.4f}")
                
                # Show which scales are being filtered
                low_occ_mask = occupancy <= occ_low
                high_occ_mask = occupancy >= occ_high
                excluded_mask = low_occ_mask | high_occ_mask
                
                if np.any(excluded_mask):
                    excluded_sizes = sizes[excluded_mask]
                    excluded_occ = occupancy[excluded_mask]
                    print(f"  Excluding {np.sum(excluded_mask)} scales:")
                    for sz, occ, reason in zip(excluded_sizes, excluded_occ, 
                                               ['low' if o <= occ_low else 'high' for o in excluded_occ]):
                        print(f"    Size {sz:6.1f}: occupancy {occ:.4f} ({reason})")
                else:
                    print(f"No scales excluded by occupancy filter")
            
            # Filter based on box counts, but apply mask to both sizes and entropy measures
            filtered_sizes, _ = filter_by_occupancy(array, sizes, box_counts, occ_low, occ_high)
            # Create mask to apply to entropy measures
            size_mask = np.isin(sizes, filtered_sizes)
            sizes = sizes[size_mask]
            measures = measures[size_mask]
        else:
            # For D0, measures are already box counts
            if verbose:
                # Show detailed occupancy calculations for D0 mode
                H, W = array.shape
                grid_boxes = (np.floor_divide(H, sizes) * np.floor_divide(W, sizes))
                occupancy = measures / grid_boxes
                
                print(f"  D0 Mode: Using box counts directly for occupancy")
                print(f"  Original scales: {len(sizes)} sizes from {sizes[0]:.1f} to {sizes[-1]:.1f}")
                print(f"  Occupancy range: {occupancy.min():.4f} to {occupancy.max():.4f}")
                
                # Show which scales are being filtered
                low_occ_mask = occupancy <= occ_low
                high_occ_mask = occupancy >= occ_high
                excluded_mask = low_occ_mask | high_occ_mask
                
                if np.any(excluded_mask):
                    excluded_sizes = sizes[excluded_mask]
                    excluded_occ = occupancy[excluded_mask]
                    print(f"  Excluding {np.sum(excluded_mask)} scales:")
                    for sz, occ, reason in zip(excluded_sizes, excluded_occ, 
                                               ['low' if o <= occ_low else 'high' for o in excluded_occ]):
                        print(f"    Size {sz:6.1f}: occupancy {occ:.4f} ({reason})")
                else:
                    print(f"  No scales excluded by occupancy filter")
            
            sizes, measures = filter_by_occupancy(array, sizes, measures, occ_low, occ_high)
        
        if show_progress:
            print(f"Occupancy filter: {original_count} → {len(sizes)} scales retained")
        
        if verbose and len(sizes) > 0:
            print(f"  Retained scale range: {sizes[0]:.1f} to {sizes[-1]:.1f}")
        
        if len(sizes) < min_points:
            if show_progress:
                print(f"Warning: Occupancy filter removed too many scales ({len(sizes)} < {min_points})")
                print("Continuing with original data...")
            # Fall back to original data if too much was filtered
            sizes = sizes_original
            measures = measures_original
    
    # Step 3: Automatic plateau detection (if enabled)
    if use_plateau_detection:
        if show_progress:
            print(f"Attempting automatic plateau detection (method={plateau_method}, window={plateau_window}, tol={plateau_tol})...")
        
        if verbose:
            print(f"  Input data for plateau detection: {len(sizes)} scales")
            print(f"  Scale range: {sizes[0]:.1f} to {sizes[-1]:.1f}")
            print(f"  Plateau method: {plateau_method}")
            
            # Show slope analysis
            log_eps = np.log(sizes)
            log_N = np.log(measures)
            slopes = np.diff(log_N) / np.diff(log_eps)
            median_slope = np.median(slopes)
            print(f"  Two-point slopes range: {slopes.min():.4f} to {slopes.max():.4f}")
            print(f"  Median slope: {median_slope:.4f}")
            print(f"  Median tolerance threshold: ±{plateau_tol*100:.1f}% of |median_slope| = ±{plateau_tol * np.abs(median_slope):.4f}")
            
            # Show pairwise slope differences if using pairwise method
            if 'pairwise' in plateau_method:
                if len(slopes) > 1:
                    slope_diffs = np.abs(np.diff(slopes))
                    print(f"  Pairwise slope differences range: {slope_diffs.min():.4f} to {slope_diffs.max():.4f}")
                    print(f"  Pairwise tolerance threshold: {pairwise_tol*100:.1f}% of |median_slope| = {pairwise_tol * np.abs(median_slope):.4f}")
                    stable_pairwise = slope_diffs < pairwise_tol * np.abs(median_slope) if np.abs(median_slope) > 1e-10 else slope_diffs < pairwise_tol
                    print(f"  Stable pairwise slopes: {np.sum(stable_pairwise)}/{len(stable_pairwise)} ({np.sum(stable_pairwise)/len(stable_pairwise)*100:.1f}%)")
            
            # Show median stability analysis
            stable = np.abs(slopes - median_slope) < plateau_tol * np.abs(median_slope) if np.abs(median_slope) > 1e-10 else np.abs(slopes - median_slope) < plateau_tol
            print(f"  Stable slopes (median): {np.sum(stable)}/{len(stable)} ({np.sum(stable)/len(stable)*100:.1f}%)")
            if np.any(stable):
                stable_indices = np.where(stable)[0]
                print(f"  Stable regions at indices: {stable_indices}")
        
        # Use the appropriate plateau detection method
        if plateau_method == 'median':
            pl_start, pl_stop = detect_scaling_plateau(sizes, measures,
                                                       window=plateau_window, 
                                                       tol=plateau_tol, 
                                                       min_pts=min_points)
            method_used = 'median'
        elif plateau_method == 'pairwise':
            pl_start, pl_stop = detect_plateau_pairwise(sizes, measures,
                                                        window=plateau_window, 
                                                        d_tol=pairwise_tol, 
                                                        min_pts=min_points)
            method_used = 'pairwise'
        else:
            # Use hybrid method
            pl_start, pl_stop, method_used = detect_plateau_hybrid(sizes, measures,
                                                                   window=plateau_window, 
                                                                   median_tol=plateau_tol,
                                                                   pairwise_tol=pairwise_tol,
                                                                   min_pts=min_points,
                                                                   method=plateau_method)
        
        if pl_start is not None:
            subset_sizes = sizes[pl_start:pl_stop]
            subset_measures = measures[pl_start:pl_stop]
            
            if verbose:
                print(f"  Plateau detected at indices {pl_start}:{pl_stop} using {method_used} method")
                print(f"  Plateau size range: {subset_sizes[0]:.1f} to {subset_sizes[-1]:.1f}")
                print(f"  Plateau contains {len(subset_sizes)} points")
                print(f"  Decade span: {np.log10(subset_sizes[-1]/subset_sizes[0]):.2f}")
            
            try:
                valid_sizes, valid_measures, d_value, fit, r2, ci_low, ci_high = compute_dimension(
                    subset_sizes, subset_measures, mode=mode, use_bootstrap_ci=use_bootstrap_ci, 
                    bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed
                )
                
                if verbose:
                    print(f"  Dimension computation results:")
                    print(f"    D_{mode[1:]}: {d_value:.4f}")
                    print(f"    R²: {r2:.6f}")
                    print(f"    Confidence interval: [{ci_low:.4f}, {ci_high:.4f}]")
                    print(f"    Valid points after filtering: {len(valid_sizes)}")
                
                if not np.isnan(r2) and len(valid_sizes) >= min_points:
                    # Package result identical to what the brute-force code builds
                    best_result = {
                        'size_range': (subset_sizes[0], subset_sizes[-1]),
                        'D_value': d_value,
                        'R2': r2,
                        'num_points': len(subset_sizes),
                        'decade_span': np.log10(subset_sizes[-1]/subset_sizes[0]),
                        'valid_sizes': valid_sizes,
                        'valid_measures': valid_measures,
                        'fit_params': fit,
                        'ci_low': ci_low,
                        'ci_high': ci_high
                    }
                    
                    if show_progress:
                        print(f"Plateau detection successful ({method_used}): range {best_result['size_range']} with R² = {r2:.6f}")
                    
                    if verbose:
                        print(f"  SUCCESS: Plateau detection found optimal range using {method_used} method!")
                        print(f"  Bypassing sliding window search (saved significant computation)")
                    
                    # Return immediately with plateau detection result
                    return {
                        'optimal_range': best_result['size_range'],
                        'D_value': best_result['D_value'],
                        'R2': best_result['R2'],
                        'optimal_sizes': best_result['valid_sizes'],
                        'optimal_measures': best_result['valid_measures'],
                        'decade_span': best_result['decade_span'],
                        'confidence_interval': (best_result['ci_low'], best_result['ci_high']),
                        'ci_type': bootstrap_method,
                        'ci_low': best_result['ci_low'],
                        'ci_high': best_result['ci_high'],
                        'all_results': [],  # Not computed in fast path
                        'qualified_results': [],  # Not computed in fast path
                        'global_sizes': sizes,
                        'global_measures': measures,
                        'num_candidates_tested': 1,
                        'num_qualified': 1,
                        'analysis_type': 'plateau_auto',
                        'optimization_mode': f'auto_plateau_{method_used}',
                        'min_R2_threshold': min_R2,
                        'threshold_met': True,
                        'fallback_mode': False
                    }
                else:
                    if verbose:
                        if np.isnan(r2):
                            print(f"  FAILED: Dimension computation produced NaN R²")
                        if len(valid_sizes) < min_points:
                            print(f"  FAILED: Too few valid points ({len(valid_sizes)} < {min_points})")
                    
            except Exception as e:
                if show_progress:
                    print(f"Plateau detection failed during dimension computation: {e}")
                    print("Falling back to sliding window search...")
                if verbose:
                    print(f"  FAILED: Exception during dimension computation: {e}")
        else:
            if show_progress:
                print("No stable plateau found, falling back to sliding window search...")
            if verbose:
                print(f"  FAILED: No stable plateau found with current parameters")
                print(f"  Try adjusting plateau_window (currently {plateau_window}) or plateau_tol (currently {plateau_tol})")
    
         # Step 4: Sliding window optimization (fallback or primary method)
    optimization_mode = "largest_stretch" if stretch else "highest_R2"
    if show_progress:
        if stretch:
            print(f"Searching for largest scaling range with R² ≥ {min_R2:.4f}...")
        else:
            print("Searching for optimal scaling range...")
    
    all_results = []
    best_r2 = -np.inf
    best_decade_span = -np.inf
    best_result = None
    
    # Progress bar for the sliding window search
    total_combinations = sum(max(0, len(sizes) - i - min_points + 1) 
                           for i in range(len(sizes)))
    
    if show_progress:
        pbar = tqdm(total=total_combinations, desc='Testing size ranges')
    
    # Try all possible starting points
    for i in range(len(sizes)):
        min_size_candidate = sizes[i]
        
        # Find all valid ending points that satisfy decade span constraint
        for j in range(i + min_points - 1, len(sizes)):
            max_size_candidate = sizes[j]
            
            if show_progress:
                pbar.update(1)
            
            # Check decade span constraint
            size_ratio = max_size_candidate / min_size_candidate
            if size_ratio < 10**min_decade_span:
                continue
                
            # Extract subset and compute dimension
            subset_sizes = sizes[i:j+1]
            subset_measures = measures[i:j+1]
            
            try:
                valid_sizes, valid_measures, d_value, fit, r2, ci_low, ci_high = compute_dimension(
                    subset_sizes, subset_measures, mode=mode, use_bootstrap_ci=use_bootstrap_ci, 
                    bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed
                )
                
                if np.isnan(r2) or len(valid_sizes) < min_points:
                    continue
                    
                result = {
                    'size_range': (min_size_candidate, max_size_candidate),
                    'D_value': d_value,
                    'R2': r2,
                    'num_points': len(valid_sizes),
                    'decade_span': np.log10(size_ratio),
                    'valid_sizes': valid_sizes,
                    'valid_measures': valid_measures,
                    'fit_params': fit,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_type': bootstrap_method
                }
                
                all_results.append(result)
                
                # Track best result based on optimization mode
                if stretch:
                    # Stretch mode: prioritize largest decade span among results meeting R² threshold
                    if r2 >= min_R2:
                        decade_span = np.log10(size_ratio)
                        if decade_span > best_decade_span:
                            best_decade_span = decade_span
                            best_r2 = r2
                            best_result = result
                else:
                    # Normal mode: prioritize highest R²
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = result
                    
            except Exception as e:
                # Skip problematic ranges
                continue
    
    if show_progress:
        pbar.close()
    
    if best_result is None:
        if stretch:
            # In stretch mode, if no results meet threshold, fall back to highest R²
            if all_results:
                best_result = max(all_results, key=lambda x: x['R2'])
                best_r2 = best_result['R2']
                if show_progress:
                    print(f"Warning: No ranges found meeting R² ≥ {min_R2:.4f} threshold.")
                    print(f"Falling back to highest R² range: {best_result['size_range']} with R² = {best_r2:.6f}")
            else:
                raise ValueError("No valid scaling ranges found meeting basic criteria")
        else:
            raise ValueError("No valid scaling ranges found meeting criteria")
        
    # Step 3: Compile results
    if show_progress:
        if stretch:
            if best_decade_span > -np.inf:  # Found qualifying result
                print(f"Largest range found: {best_result['size_range']} with R² = {best_r2:.6f}, span = {best_decade_span:.2f} decades")
            # If we fell back to highest R², the warning was already printed above
        else:
            print(f"Optimal range found: {best_result['size_range']} with R² = {best_r2:.6f}")
    
    # Sort results differently based on mode
    if stretch:
        # In stretch mode, sort by decade span (descending) among results meeting R² threshold
        qualified_results = [r for r in all_results if r['R2'] >= min_R2]
        sorted_results = sorted(qualified_results, key=lambda x: x['decade_span'], reverse=True)
        # Include all results but mark which ones qualified
        all_sorted = sorted(all_results, key=lambda x: x['decade_span'], reverse=True)
    else:
        # In normal mode, sort by R² (descending)
        sorted_results = sorted(all_results, key=lambda x: x['R2'], reverse=True)
        all_sorted = sorted_results
    
    # Check if we're in fallback mode (stretch mode but no results met threshold)
    threshold_met = True
    if stretch and best_decade_span == -np.inf:
        threshold_met = False
    
    return {
        'optimal_range': best_result['size_range'],
        'D_value': best_result['D_value'], 
        'R2': best_result['R2'],
        'optimal_sizes': best_result['valid_sizes'],
        'optimal_measures': best_result['valid_measures'],
        'decade_span': best_result['decade_span'],
        'confidence_interval': (best_result['ci_low'], best_result['ci_high']),
        'ci_type': bootstrap_method,
        'ci_low': best_result['ci_low'],
        'ci_high': best_result['ci_high'],
        'all_results': all_sorted,
        'qualified_results': sorted_results if stretch else all_sorted,
        'global_sizes': sizes,
        'global_measures': measures,
        'num_candidates_tested': len(all_results),
        'num_qualified': len([r for r in all_results if r['R2'] >= min_R2]) if stretch else len(all_results),
        'analysis_type': 'dynamic_boxcount_stretch' if stretch else 'dynamic_boxcount',
        'optimization_mode': optimization_mode,
        'min_R2_threshold': min_R2 if stretch else None,
        'threshold_met': threshold_met,
        'fallback_mode': not threshold_met if stretch else False
    }

def benchmark_boxcount_optimizations(array, mode='D0', num_sizes=20, min_size=16, max_size=None, num_offsets=10, use_min_count=False, seed=None):
    """
    Benchmark different boxcount optimization levels to show performance improvements.
    
    Args:
        array: 2D binary array to analyze
        mode: 'D0' or 'D1' 
        num_sizes: Number of box sizes to test
        min_size: Minimum box size
        max_size: Maximum box size
        num_offsets: Number of grid offsets
        use_min_count: For D0 mode, whether to use minimum count across offsets (True) or average count (False, default)
        seed: Random seed for reproducible results
        
    Returns:
        dict: Performance comparison results
    """
    if max_size is None:
        max_size = min(array.shape) // 5
    
    results = {}
    
    # Calculate array properties
    total_pixels = array.size
    non_zero_pixels = np.count_nonzero(array)
    sparsity = non_zero_pixels / total_pixels
    
    print(f"Array properties:")
    print(f"  Shape: {array.shape}")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Non-zero pixels: {non_zero_pixels:,}")
    print(f"  Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
    print(f"  Box size range: {min_size} - {max_size}")
    print(f"  Num sizes: {num_sizes}, Num offsets: {num_offsets}")
    print(f"  Use min count: {use_min_count}")
    print(f"  Random seed: {seed}")
    print()
    
    # Test original implementation
    print("Testing original implementation...")
    start_time = time.time()
    sizes1, counts1 = boxcount(array, mode=mode, num_sizes=num_sizes, min_size=min_size, 
                              max_size=max_size, num_offsets=num_offsets, use_optimization=False, use_min_count=use_min_count, seed=seed)
    original_time = time.time() - start_time
    results['original'] = {'time': original_time, 'sizes': sizes1, 'counts': counts1}
    print(f"  Time: {original_time:.3f} seconds")
    
    # Test bounding box optimization
    print("Testing bounding box optimization...")
    start_time = time.time()
    sizes2, counts2 = boxcount(array, mode=mode, num_sizes=num_sizes, min_size=min_size, 
                              max_size=max_size, num_offsets=num_offsets, use_optimization=True, sparse_threshold=0, use_min_count=use_min_count, seed=seed)
    bbox_time = time.time() - start_time
    results['bounding_box'] = {'time': bbox_time, 'sizes': sizes2, 'counts': counts2}
    print(f"  Time: {bbox_time:.3f} seconds")
    print(f"  Speedup: {original_time/bbox_time:.2f}x")
    
    # Test sparse optimization (D0 only)
    if mode == 'D0' and sparsity <= 0.1:  # Only test if reasonably sparse
        print("Testing sparse optimization...")
        start_time = time.time()
        sizes3, counts3 = boxcount(array, mode=mode, num_sizes=num_sizes, min_size=min_size, 
                                  max_size=max_size, num_offsets=num_offsets, use_optimization=True, sparse_threshold=0.1, use_min_count=use_min_count, seed=seed)
        sparse_time = time.time() - start_time
        results['sparse'] = {'time': sparse_time, 'sizes': sizes3, 'counts': counts3}
        print(f"  Time: {sparse_time:.3f} seconds")
        print(f"  Speedup vs original: {original_time/sparse_time:.2f}x")
        print(f"  Speedup vs bounding box: {bbox_time/sparse_time:.2f}x")
    else:
        print("Skipping sparse optimization (array not sparse enough or mode != D0)")
    
    # Verify results are consistent
    print("\nVerifying consistency...")
    if np.allclose(counts1, counts2, rtol=1e-10):
        print("Bounding box optimization produces identical results")
    else:
        print("WARNING: Bounding box optimization results differ!")
        
    if 'sparse' in results and np.allclose(counts1, counts3, rtol=1e-10):
        print("Sparse optimization produces identical results")
    elif 'sparse' in results:
        print("WARNING: Sparse optimization results differ!")
    
    # Summary
    print(f"\nPerformance Summary:")
    print(f"  Original: {original_time:.3f}s")
    print(f"  Bounding box: {bbox_time:.3f}s ({original_time/bbox_time:.2f}x speedup)")
    if 'sparse' in results:
        sparse_time = results['sparse']['time']
        print(f"  Sparse: {sparse_time:.3f}s ({original_time/sparse_time:.2f}x speedup)")
    
    return results

def plot_dynamic_boxcount_results(dynamic_result, figsize=(15, 10)):
    """
    Visualize the results of dynamic box counting analysis.
    
    Args:
        dynamic_result (dict): Output from dynamic_boxcount function
        figsize (tuple): Figure size for the plot
    """
    
    if 'error' in dynamic_result:
        print(f"Cannot plot: {dynamic_result['error']}")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Plot 1: Global scaling with optimal range highlighted
    ax1.scatter(np.log10(dynamic_result['global_sizes']), 
               np.log10(dynamic_result['global_measures']), 
               alpha=0.5, color='lightgray', label='All data')
    ax1.scatter(np.log10(dynamic_result['optimal_sizes']), 
               np.log10(dynamic_result['optimal_measures']), 
               color='red', s=50, label='Optimal range')
    
    # Plot optimal fit line
    fit_x = np.log10(dynamic_result['optimal_sizes'])
    fit = np.polyfit(fit_x, np.log10(dynamic_result['optimal_measures']), 1)
    ax1.plot(fit_x, fit[0] * fit_x + fit[1], 'r-', linewidth=2, 
             label=f'D = {dynamic_result["D_value"]:.3f}, R² = {dynamic_result["R2"]:.4f}')
    
    ax1.set_xlabel('Log₁₀(Box Size)')
    ax1.set_ylabel('Log₁₀(Box Count)')
    ax1.set_title('Global vs Optimal Scaling Range')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R² vs range size
    all_results = dynamic_result['all_results']
    
    # Handle plateau detection case where all_results is empty
    if len(all_results) == 0:
        # Create a single point representing the plateau result
        decade_spans = [dynamic_result['decade_span']]
        r2_values = [dynamic_result['R2']]
    else:
        decade_spans = [r['decade_span'] for r in all_results]
        r2_values = [r['R2'] for r in all_results]
    
    # Color code points based on mode
    if dynamic_result.get('optimization_mode') == 'largest_stretch':
        min_R2 = dynamic_result.get('min_R2_threshold', 0.9985)
        qualified_results = dynamic_result.get('qualified_results', [])
        qualified_spans = [r['decade_span'] for r in qualified_results]
        qualified_r2s = [r['R2'] for r in qualified_results]
        
        # Plot all points in light gray
        ax2.scatter(decade_spans, r2_values, alpha=0.3, color='lightgray', label='Below threshold')
        # Plot qualified points in blue
        if qualified_spans:
            ax2.scatter(qualified_spans, qualified_r2s, alpha=0.7, color='blue', label=f'R² ≥ {min_R2:.4f}')
        # Add horizontal line for R² threshold
        ax2.axhline(y=min_R2, color='orange', linestyle='--', alpha=0.7, label=f'R² threshold = {min_R2:.4f}')
        ax2.set_title(f'R² vs Scaling Range Width (Stretch Mode)')
    else:
        ax2.scatter(decade_spans, r2_values, alpha=0.6, color='blue')
        ax2.set_title('R² vs Scaling Range Width')
    
    # Highlight optimal result
    ax2.scatter(dynamic_result['decade_span'], dynamic_result['R2'], 
               color='red', s=100, zorder=5, label='Selected')
    ax2.set_xlabel('Decade Span (log₁₀ units)')
    ax2.set_ylabel('R² Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Size range visualization
    if len(all_results) == 0:
        # Handle plateau detection case with single point
        min_sizes = [dynamic_result['optimal_range'][0]]
        max_sizes = [dynamic_result['optimal_range'][1]]
        scatter = ax3.scatter(min_sizes, max_sizes, c=[dynamic_result['R2']], 
                             cmap='viridis', alpha=0.7, s=100)
    else:
        min_sizes = [r['size_range'][0] for r in all_results]
        max_sizes = [r['size_range'][1] for r in all_results]
        scatter = ax3.scatter(min_sizes, max_sizes, c=[r['R2'] for r in all_results], 
                             cmap='viridis', alpha=0.7)
    ax3.scatter(dynamic_result['optimal_range'][0], dynamic_result['optimal_range'][1], 
               color='red', s=100, marker='x', linewidth=3, label='Optimal')
    
    # Add diagonal lines for constant decade spans
    size_range = np.logspace(np.log10(min(min_sizes)), np.log10(max(max_sizes)), 100)
    for decade in [1, 1.5, 2]:
        ax3.plot(size_range, size_range * 10**decade, '--', alpha=0.5, 
                label=f'{decade} decades')
    
    ax3.set_xlabel('Min Size')
    ax3.set_ylabel('Max Size')
    ax3.set_title('Size Range Map (colored by R²)')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='R²')
    
    # Plot 4: Top candidates comparison
    if len(all_results) == 0:
        # Handle plateau detection case with single result
        ax4.scatter(np.log10(dynamic_result['optimal_sizes']), 
                   np.log10(dynamic_result['optimal_measures']), 
                   alpha=1.0, color='red', s=40, label='Plateau result')
        ax4.legend()
    else:
        top_results = sorted(all_results, key=lambda x: x['R2'], reverse=True)[:10]
        
        for i, result in enumerate(top_results):
            alpha = 1.0 if i == 0 else 0.3
            color = 'red' if i == 0 else 'gray'
            ax4.scatter(np.log10(result['valid_sizes']), 
                       np.log10(result['valid_measures']), 
                       alpha=alpha, color=color, s=20)
    
    ax4.set_xlabel('Log₁₀(Box Size)')
    ax4.set_ylabel('Log₁₀(Box Count)')
    ax4.set_title('Top 10 Scaling Ranges (Best in Red)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    mode_str = "Stretch Mode" if dynamic_result.get('optimization_mode') == 'largest_stretch' else "Optimization Mode"
    
    # Check if this is a plateau detection result
    if dynamic_result.get('optimization_mode') == 'auto_plateau':
        mode_str = "Automatic Plateau Detection"
    
    print(f"\n=== Dynamic Box Counting Results ({mode_str}) ===")
    print(f"Selected size range: {dynamic_result['optimal_range'][0]:.1f} - {dynamic_result['optimal_range'][1]:.1f}")
    print(f"Fractal dimension: {dynamic_result['D_value']:.4f} ± {(dynamic_result['confidence_interval'][1] - dynamic_result['confidence_interval'][0])/2:.4f}")
    print(f"R² value: {dynamic_result['R2']:.6f}")
    print(f"Decade span: {dynamic_result['decade_span']:.2f}")
    print(f"Data points in fit: {len(dynamic_result['optimal_sizes'])}")
    print(f"Total candidates tested: {dynamic_result['num_candidates_tested']}")
    
    if dynamic_result.get('optimization_mode') == 'auto_plateau':
        print(f"Plateau detection successful - no sliding window search needed")
    elif dynamic_result.get('optimization_mode') == 'largest_stretch':
        min_R2 = dynamic_result.get('min_R2_threshold', 0.9985)
        num_qualified = dynamic_result.get('num_qualified', 0)
        threshold_met = dynamic_result.get('threshold_met', True)
        print(f"R² threshold: {min_R2:.4f}")
        print(f"Qualified candidates: {num_qualified}/{dynamic_result['num_candidates_tested']}")
        if not threshold_met:
            print(f"WARNING: Fallback mode: No ranges met R² threshold, selected best available")

def bootstrap_pairs(sizes,
                    measures,
                    mode='D0',
                    n_bootstrap=1000,
                    alpha=0.05,
                    use_weighted_fit=True,
                    random_seed=None,
                    studentized=False,
                ):
    """
    Compute fractal dimension confidence intervals using standard bootstrap resampling.
    
    This function implements the standard (non-parametric) bootstrap method for estimating
    confidence intervals of fractal dimensions. Unlike residual bootstrap, this method
    resamples the original data points (sizes, measures) with replacement, which can be
    useful when the regression model assumptions are more severely violated or when you
    want to account for uncertainty in both x and y variables.
    
    Parameters
    ----------
    sizes : array-like
        Box sizes used in the scaling analysis
    measures : array-like
        Corresponding measures (box counts for D0, entropy values for D1)
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension using log10(N) vs log10(ε)
        - 'D1': Information dimension using H vs log2(ε)
    n_bootstrap : int, default 1000
        Number of bootstrap resamples. More resamples give more stable confidence
        intervals but increase computation time.
    alpha : float, default 0.05
        Significance level for confidence intervals. Default 0.05 gives 95% CI.
        
    Returns
    -------
    tuple
        (d_median, d_lower, d_upper) where:
        - d_median : float - Median fractal dimension over all bootstrap samples
        - d_lower : float - Lower bound of confidence interval
        - d_upper : float - Upper bound of confidence interval
        
    Notes
    -----
    Standard Bootstrap vs Residual Bootstrap:
    
    Standard Bootstrap (this function):
    - Resamples (x,y) pairs with replacement
    - Accounts for uncertainty in both box sizes and measures
    - More robust when regression assumptions are severely violated
    - Can handle cases where x-values have measurement error
    - May be less efficient than residual bootstrap for well-behaved data
    
    Residual Bootstrap (bootstrap_residual):
    - Resamples residuals while keeping x-values fixed
    - Assumes x-values are known exactly
    - More efficient for typical regression problems
    - Better preserves the structure of the regression relationship
    - Generally preferred for fractal dimension estimation
    
    When to Use Standard Bootstrap:
    - When box sizes have significant measurement uncertainty
    - When the regression relationship is highly non-linear
    - When residual bootstrap assumptions are violated
    - For exploratory analysis or robustness checks
    
    The method uses ordinary least squares for simplicity and speed, making it
    suitable for quick confidence interval estimation. For more sophisticated
    statistical analysis, consider using compute_dimension with residual bootstrap.
    
    Examples
    --------
    >>> sizes = [1, 2, 4, 8, 16, 32]
    >>> counts = [1000, 250, 63, 16, 4, 1]
    >>> d_med, d_low, d_high = bootstrap_pairs(sizes, counts, mode='D0', n_bootstrap=1000)
    >>> print(f"D0 = {d_med:.3f} [{d_low:.3f}, {d_high:.3f}]")
    
    See Also
    --------
    bootstrap_residual : Residual bootstrap method (generally preferred)
    compute_dimension : Main function with multiple bootstrap options
    """
    rng = np.random.default_rng(random_seed)
    sizes = np.asarray(sizes, dtype=float)
    measures = np.asarray(measures, dtype=float)
    n = sizes.size

    # helpers to match your residual function’s conventions
    def prepare_xyw(sz, ms, mode, use_w):
        if mode == 'D0':
            x = np.log10(sz)
            y = np.log10(ms)
            w = ms if use_w else np.ones_like(x)            # var(log N) ~ 1/N
        elif mode == 'D1':
            x = np.log2(sz)
            y = ms
            w = (1.0 / (ms + 1e-10)) if use_w else np.ones_like(x)
        else:
            raise ValueError("mode must be 'D0' or 'D1'")
        return x, y, w

    def wls_slope(x, y, w=None, ridge=1e-12):
        X = np.column_stack((x, np.ones_like(x)))
        if w is None:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            yhat = X @ beta
            # OLS cov
            rss = np.sum((y - yhat)**2)
            sigma2 = rss / (len(y) - 2)
            cov = np.linalg.inv(X.T @ X) * sigma2
            return beta[0], beta, yhat, cov
        # WLS via sqrt(w) scaling
        sw = np.sqrt(np.clip(w, 0.0, np.inf))
        Xw = X * sw[:, None]
        yw = y * sw
        XtX = Xw.T @ Xw
        XtX.flat[::XtX.shape[0]+1] += ridge
        beta = np.linalg.solve(XtX, Xw.T @ yw)
        yhat = X @ beta
        # Unbiased sigma^2 under precision weights
        sigma2 = np.sum(w * (y - yhat)**2) / (len(y) - 2)
        cov = np.linalg.inv(X.T @ (w[:, None] * X)) * sigma2
        return beta[0], beta, yhat, cov

    # Point estimate on original data
    x0, y0, w0 = prepare_xyw(sizes, measures, mode, use_weighted_fit)
    slope_hat, beta_hat, yhat, cov_hat = wls_slope(x0, y0, (w0 if use_weighted_fit else None))
    d_hat = -slope_hat
    se_hat = float(np.sqrt(cov_hat[0, 0]))

    d_samples = []
    t_stats = [] if studentized else None

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # resample pairs
        s_b = sizes[idx]
        m_b = measures[idx]
        x_b, y_b, w_b = prepare_xyw(s_b, m_b, mode, use_weighted_fit)

        try:
            slope_b, _, _, cov_b = wls_slope(x_b, y_b, (w_b if use_weighted_fit else None))
        except np.linalg.LinAlgError:
            # skip rare singular draws
            continue

        d_b = -slope_b
        d_samples.append(d_b)

        if studentized:
            se_b = float(np.sqrt(cov_b[0, 0]))
            # studentized pivot for slope, then map to D with a sign:
            # t* = (slope_b - slope_hat)/se_b, same as -(d_b - d_hat)/se_b
            t_stats.append((slope_b - slope_hat) / max(se_b, 1e-20))

    d_samples = np.array(d_samples)

    if d_samples.size == 0:
        return d_hat, np.nan, np.nan

    if studentized:
        q_lo, q_hi = np.quantile(np.array(t_stats), [1 - alpha/2, alpha/2])
        slope_low  = slope_hat - q_lo  * se_hat
        slope_high = slope_hat - q_hi * se_hat
        ci_low, ci_high = -slope_high, -slope_low
    else:
        # percentile CI on D directly
        ci_low, ci_high = np.quantile(d_samples, [alpha/2, 1 - alpha/2])

    return float(d_hat), float(ci_low), float(ci_high)

def integral_image_d0(mask, sizes, offsets, use_min_count=False):
    """
    2‑D capacity‑dimension (D0) box counts using a summed‑area table.

    Parameters
    ----------
    mask : (H, W) array_like
        Binary image (non‑zeros are foreground).  Will be cast to uint8.
    sizes : (S,) 1‑D int array
        Box sizes in pixels.
    offsets : (S, M, 2) int array
        For each size `sizes[i]`, offsets[i] is an (M, 2) array of (x_off, y_off).
        Values may be larger than the box size; they are reduced modulo `size`.
    use_min_count : bool, optional
        If True, take the minimum count over the M offsets; otherwise take the mean.

    Returns
    -------
    counts : (S,) float64
        Box count for each size.
    """
    # --- build summed‑area table with 1‑pixel zero border
    mask = mask.astype(np.uint8, copy=False)
    H, W = mask.shape
    sat = np.zeros((H + 1, W + 1), dtype=np.uint64)
    sat[1:, 1:] = mask
    
    start_time = time.perf_counter()
    
    sat = sat.cumsum(0).cumsum(1)           # O(N²) but runs in C

    end_time = time.perf_counter()

    print(f"Time taken to build summed area table: {end_time - start_time} seconds")

    out = np.empty(len(sizes), dtype=np.float64)

    start_time = time.perf_counter()

    for i, s in enumerate(sizes):
        m_offsets = offsets[i]
        counts = np.empty(len(m_offsets), dtype=np.int64)

        for j, (x_raw, y_raw) in enumerate(m_offsets):
            # normalise so 0 <= offset < s
            x_off = int(x_raw) % s
            y_off = int(y_raw) % s

            # four‑corner inclusion–exclusion on a stride‑s grid
            br = sat[x_off + s : H + 1 : s, y_off + s : W + 1 : s]
            bl = sat[x_off + s : H + 1 : s, y_off       : W + 1 - s : s]
            tr = sat[x_off       : H + 1 - s : s, y_off + s : W + 1 : s]
            tl = sat[x_off       : H + 1 - s : s, y_off       : W + 1 - s : s]

            counts[j] = (br + tl - tr - bl > 0).sum()

        out[i] = counts.min() if use_min_count else counts.mean()

    end_time = time.perf_counter()

    print(f"Time taken to compute box counts: {end_time - start_time} seconds")

    return out

def build_sat(mask: np.ndarray) -> np.ndarray:
    """Summed‑area table with 1‑pixel zero border (NumPy only)."""
    H, W = mask.shape
    sat = np.zeros((H + 1, W + 1), dtype=np.uint32)
    sat[1:, 1:] = mask.astype(np.uint8)
    # two in‑place passes, still C‑speed
    sat.cumsum(axis=0, out=sat)
    sat.cumsum(axis=1, out=sat)
    return sat

@njit(parallel=True, fastmath=True, cache=True)
def counts_from_sat(sat, sizes, offsets, use_min):
    """
    Parallel box counts from a summed‑area table (SAT).

    Parameters
    ----------
    sat      : (H+1, W+1) uint32  –  summed‑area table
    sizes    : (S,)      int32    –  box sizes
    offsets  : (S, M, 2) int32    –  (x_off, y_off) pairs for each size
    use_min  : bool                 minimise vs. average across offsets
    """
    S = sizes.shape[0]
    out = np.empty(S, dtype=np.float64)
    H = sat.shape[0] - 1
    W = sat.shape[1] - 1

    for i in prange(S):                    # ← parallel over box sizes
        s     = sizes[i]
        offs  = offsets[i]
        m     = offs.shape[0]

        best  = 1e18          # big number for min‑reduction
        acc   = 0.0           # running sum for mean‑reduction

        for j in prange(m):
            # normalise offsets so 0 ≤ x_off,y_off < s
            x_off = int(offs[j, 0]) % s
            y_off = int(offs[j, 1]) % s

            br = sat[x_off + s : H + 1 : s, y_off + s : W + 1 : s]
            bl = sat[x_off + s : H + 1 : s, y_off       : W + 1 - s : s]
            tr = sat[x_off       : H + 1 - s : s, y_off + s : W + 1 : s]
            tl = sat[x_off       : H + 1 - s : s, y_off       : W + 1 - s : s]

            # Scratch array the same shape as br
            buf = np.empty(br.shape, dtype=np.uint32)
            buf[:] = br         # copy br
            buf += tl           # buf = br + tl
            buf -= tr           # buf -= tr
            buf -= bl           # buf -= bl

            cnt = (buf > 0).sum()   # occupancy for this offset

            if use_min:
                if cnt < best:
                    best = cnt
            else:
                acc += cnt

        out[i] = best if use_min else acc / m

    return out