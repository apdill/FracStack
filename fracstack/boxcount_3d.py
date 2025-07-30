import numpy as np
from numba import njit, prange
from .boxcount import generate_random_offsets, get_sizes, compute_dimension

def generate_random_offsets_3d(sizes, num_offsets, seed=None):
    """
    Generate random grid offsets for 3D box counting.
    
    Parameters
    ----------
    sizes : array-like
        Array of box sizes (in voxels) for which to generate offsets
    num_offsets : int
        Number of offset positions to generate for each box size
    seed : int, optional
        Random seed for reproducible results
        
    Returns
    -------
    np.ndarray
        Array of shape (len(sizes), num_offsets, 3) containing (x_offset, y_offset, z_offset) triplets
    """
    if seed is not None: 
        np.random.seed(seed)
    
    off = np.zeros((len(sizes), num_offsets, 3), dtype=np.uint16)
    
    for i, s in enumerate(sizes):
        # First offset is always the geometric center
        off[i, 0] = (s // 2, s // 2, s // 2)
        
        # Generate additional random offsets
        j = 0
        for j in range(1, min(num_offsets, s * s * s)):
            r = np.random.randint(0, s * s * s)
            # Convert linear index to 3D coordinates
            z = r // (s * s)
            y = (r % (s * s)) // s
            x = r % s
            off[i, j] = (x, y, z)
        
        # Fill remaining slots with last valid offset if needed
        if j + 1 < num_offsets:
            off[i, j + 1:] = off[i, j]
            
    return off

@njit(nogil=True, cache=True)
def get_bounding_box_3d(array):
    """
    Compute the tight bounding box of all non-zero voxels in a 3D array.
    
    Parameters
    ----------
    array : np.ndarray
        3D array to analyze
        
    Returns
    -------
    tuple
        (min_x, min_y, min_z, max_x, max_y, max_z) defining the bounding box
    """
    D, H, W = array.shape
    
    # Find dimensions that contain non-zero voxels
    x_has_data = np.zeros(D, dtype=np.bool_)
    y_has_data = np.zeros(H, dtype=np.bool_)
    z_has_data = np.zeros(W, dtype=np.bool_)
    
    for i in range(D):
        for j in range(H):
            for k in range(W):
                if array[i, j, k] > 0:
                    x_has_data[i] = True
                    y_has_data[j] = True
                    z_has_data[k] = True
    
    # Find the bounds of non-zero regions
    min_x, max_x = D, -1
    min_y, max_y = H, -1
    min_z, max_z = W, -1
    
    for i in range(D):
        if x_has_data[i]:
            if min_x == D:
                min_x = i
            max_x = i
    
    for j in range(H):
        if y_has_data[j]:
            if min_y == H:
                min_y = j
            max_y = j
    
    for k in range(W):
        if z_has_data[k]:
            if min_z == W:
                min_z = k
            max_z = k
    
    if min_x == D:  # No non-zero voxels found
        return 0, 0, 0, 0, 0, 0
    
    return min_x, min_y, min_z, max_x + 1, max_y + 1, max_z + 1

@njit(nogil=True, cache=True)
def box_intersects_bounds_3d(x, y, z, size, min_x, min_y, min_z, max_x, max_y, max_z):
    """
    Check if a 3D box intersects with a given bounding box region.
    
    Parameters
    ----------
    x, y, z : int
        Top-left-front coordinates of the box to test
    size : int
        Size of the box (cubic)
    min_x, min_y, min_z : int
        Minimum coordinates of the bounding box (inclusive)
    max_x, max_y, max_z : int
        Maximum coordinates of the bounding box (exclusive)
        
    Returns
    -------
    bool
        True if the box intersects with the bounding box, False otherwise
    """
    box_max_x = x + size
    box_max_y = y + size
    box_max_z = z + size
    
    # Box doesn't intersect if it's completely outside the bounds
    return not (box_max_x <= min_x or x >= max_x or 
                box_max_y <= min_y or y >= max_y or
                box_max_z <= min_z or z >= max_z)

@njit(nogil=True, parallel=True, cache=True)
def numba_d0_3d(array, sizes, offsets, use_min_count=False):
    """
    Compute 3D capacity dimension (D0) box counts using numba optimization.
    
    Parameters
    ----------
    array : np.ndarray
        3D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in voxels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 3) containing pre-generated
        (x_offset, y_offset, z_offset) triplets for each size
    use_min_count : bool, default False
        If True, return minimum count across all offsets for each size.
        If False, return average count across all offsets for each size.
        
    Returns
    -------
    np.ndarray
        1D array of box counts for each size
    """
    results = np.empty(len(sizes), dtype=np.int64)
    
    # Precompute bounding box once for all sizes
    min_x, min_y, min_z, max_x, max_y, max_z = get_bounding_box_3d(array)
    
    # Early exit for completely empty arrays
    if min_x == max_x and min_y == max_y and min_z == max_z:
        results.fill(0)
        return results
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        D, H, W = array.shape
        
        # Calculate actual centered offsets for this array size
        centered_x = (D % size) // 2
        centered_y = (H % size) // 2
        centered_z = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**3)
        
        if use_min_count:
            # Find minimum count across offsets
            min_count = np.inf
            for offset_idx in range(total_offsets):
                if offset_idx == 0:
                    x_off = centered_x
                    y_off = centered_y
                    z_off = centered_z
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                    z_off = offsets[idx, offset_idx, 2] % size
                
                # 3D Box counting logic with bounding box optimization
                count = 0
                max_x_range = x_off + ((D - x_off) // size) * size
                max_y_range = y_off + ((H - y_off) // size) * size
                max_z_range = z_off + ((W - z_off) // size) * size
                
                for x in range(x_off, max_x_range, size):
                    for y in range(y_off, max_y_range, size):
                        for z in range(z_off, max_z_range, size):
                            # Skip boxes that don't intersect with bounding box
                            if box_intersects_bounds_3d(x, y, z, size, min_x, min_y, min_z, max_x, max_y, max_z):
                                count += array[x:x+size, y:y+size, z:z+size].any()
                
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
                    z_off = centered_z
                else:
                    # Use pre-generated offsets
                    x_off = offsets[idx, offset_idx, 0] % size
                    y_off = offsets[idx, offset_idx, 1] % size
                    z_off = offsets[idx, offset_idx, 2] % size
                
                # 3D Box counting logic with bounding box optimization
                count = 0
                max_x_range = x_off + ((D - x_off) // size) * size
                max_y_range = y_off + ((H - y_off) // size) * size
                max_z_range = z_off + ((W - z_off) // size) * size
                
                for x in range(x_off, max_x_range, size):
                    for y in range(y_off, max_y_range, size):
                        for z in range(z_off, max_z_range, size):
                            # Skip boxes that don't intersect with bounding box
                            if box_intersects_bounds_3d(x, y, z, size, min_x, min_y, min_z, max_x, max_y, max_z):
                                count += array[x:x+size, y:y+size, z:z+size].any()
                
                count_sum += count
            
            results[idx] = count_sum / total_offsets if total_offsets > 0 else 0
    
    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d1_3d(array, sizes, offsets):
    """
    Compute 3D information dimension (D1) entropy values using numba optimization.
    
    Parameters
    ----------
    array : np.ndarray
        3D binary array to analyze, should be contiguous float32 for numba compatibility
    sizes : np.ndarray
        1D array of box sizes (in voxels) to test
    offsets : np.ndarray
        3D array of shape (len(sizes), num_offsets, 3) containing pre-generated
        (x_offset, y_offset, z_offset) triplets for each size
        
    Returns
    -------
    np.ndarray
        1D array of entropy values H(ε) for each box size, averaged across all offsets
    """
    results = np.empty(len(sizes), dtype=np.float64)
    M = array.sum()
    D, H, W = array.shape
    
    # Precompute bounding box once for all sizes
    min_x, min_y, min_z, max_x, max_y, max_z = get_bounding_box_3d(array)
    
    # Early exit for completely empty arrays
    if M == 0 or (min_x == max_x and min_y == max_y and min_z == max_z):
        results.fill(0.0)
        return results
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        if size == 0:
            results[idx] = 0.0
            continue
            
        # Calculate actual centered offsets for this array size
        centered_x = (D % size) // 2
        centered_y = (H % size) // 2
        centered_z = (W % size) // 2
        total_offsets = min(offsets.shape[1], size**3)
        entropy_sum = 0.0
        
        for offset_idx in range(total_offsets):
            # Get offset coordinates
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
                z_off = centered_z
            else:
                # Use pre-generated offsets
                x_off = offsets[idx, offset_idx, 0] % size
                y_off = offsets[idx, offset_idx, 1] % size
                z_off = offsets[idx, offset_idx, 2] % size
            
            # 3D Box processing with bounding box optimization
            max_x_range = x_off + ((D - x_off) // size) * size
            max_y_range = y_off + ((H - y_off) // size) * size
            max_z_range = z_off + ((W - z_off) // size) * size
            entropy = 0.0
            
            for x in range(x_off, max_x_range, size):
                for y in range(y_off, max_y_range, size):
                    for z in range(z_off, max_z_range, size):
                        # Skip boxes that don't intersect with bounding box
                        if box_intersects_bounds_3d(x, y, z, size, min_x, min_y, min_z, max_x, max_y, max_z):
                            box = array[x:x+size, y:y+size, z:z+size]
                            box_sum = box.sum()
                            if box_sum > 0:
                                p = box_sum / M
                                entropy += -p * np.log2(p)
            
            entropy_sum += entropy
        
        # Average across offsets
        results[idx] = entropy_sum / total_offsets if total_offsets > 0 else 0.0
    
    return results

def boxcount_3d(array, mode='D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1, 
                use_min_count=False, seed=None):
    """
    Perform 3D box counting analysis.
    
    Parameters
    ----------
    array : np.ndarray
        3D binary array to analyze. Will be converted to contiguous float32 format.
    mode : str, default 'D0'
        Type of dimension to compute:
        - 'D0': Capacity dimension (box counts)
        - 'D1': Information dimension (entropy values)
    num_sizes : int, default 10
        Number of box sizes to test, distributed geometrically between min_size and max_size
    min_size : int, optional
        Minimum box size in voxels. Defaults to 1 if not specified.
    max_size : int, optional
        Maximum box size in voxels. Defaults to min(array.shape)//5 if not specified.
    num_offsets : int, default 1
        Number of grid offset positions to test for each box size
    use_min_count : bool, default False
        For D0 mode only: whether to use minimum count across offsets (True) or
        average count across offsets (False). Averaging is generally recommended.
    seed : int, optional
        Random seed for reproducible grid offset generation
        
    Returns
    -------
    tuple
        (sizes, counts) where:
        - sizes: List of box sizes used in the analysis
        - counts: List of corresponding measures (box counts for D0, entropy values for D1)
    """
    array = np.ascontiguousarray(array.astype(np.float32))
    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape)//5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    sizes_arr = np.array(sizes)
    
    # Pre-generate random offsets for 3D
    offsets = generate_random_offsets_3d(sizes_arr, num_offsets, seed=seed)
    
    if mode == 'D0':
        counts = numba_d0_3d(array, sizes_arr, offsets, use_min_count)
    elif mode == 'D1':
        counts = numba_d1_3d(array, sizes_arr, offsets)
    else:
        raise ValueError("Invalid mode, use 'D0' or 'D1'")
    
    return sizes, counts.tolist()

def measure_dimension_3d(input_array, 
                        mode='D0', 
                        num_sizes=50, 
                        min_size=2, 
                        max_size=None, 
                        num_offsets=50, 
                        use_min_count=False, 
                        seed=None, 
                        use_weighted_fit=True,
                        use_bootstrap_ci=False,
                        bootstrap_method='residual',
                        n_bootstrap=1000,
                        bootstrap_seed=None):
    """
    Measure 3D fractal dimension using box counting analysis.
    
    Parameters
    ----------
    input_array : np.ndarray
        3D binary array to analyze. Non-binary inputs are automatically
        converted to binary using boolean casting.
    mode : str, default 'D0'
        Type of fractal dimension to compute:
        - 'D0': Capacity dimension (box counting dimension)
        - 'D1': Information dimension (entropy-based)
    num_sizes : int, default 50
        Number of box sizes to test, distributed geometrically between
        min_size and max_size
    min_size : int, default 2
        Minimum box size in voxels
    max_size : int, optional
        Maximum box size in voxels. Defaults to min(array.shape)//5 if not specified.
    num_offsets : int, default 50
        Number of grid offset positions to test for each box size
    use_min_count : bool, default False
        For D0 mode only: whether to use minimum count across offsets (True)
        or average count across offsets (False, recommended).
    seed : int, optional
        Random seed for reproducible grid offset generation
    use_weighted_fit : bool, default True
        Whether to use weighted least squares instead of ordinary least squares
    use_bootstrap_ci : bool, default False
        Whether to compute bootstrap confidence intervals
    bootstrap_method : str, default 'residual'
        Bootstrap method for confidence intervals ('residual' or 'standard')
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility
        
    Returns
    -------
    dict
        Dictionary containing 3D fractal dimension analysis results:
        - 'D' : float - Computed fractal dimension
        - 'valid_sizes' : np.ndarray - Box sizes used in final fit after filtering
        - 'valid_counts' : np.ndarray - Corresponding measures used in final fit
        - 'fit' : np.ndarray - Linear fit parameters [slope, intercept]
        - 'R2' : float - R-squared value indicating goodness of fit
        - 'ci_low' : float - Lower bound of confidence interval (if bootstrap enabled)
        - 'ci_high' : float - Upper bound of confidence interval (if bootstrap enabled)
    """
    
    # Ensure binary input
    if not np.array_equal(input_array/np.max(input_array), input_array.astype(bool)):
        input_array = input_array.astype(bool).astype(np.uint8)
    
    if max_size is None:
        max_size = min(input_array.shape) // 5
        
    sizes, counts = boxcount_3d(input_array, mode=mode, min_size=min_size, max_size=max_size, 
                               num_sizes=num_sizes, num_offsets=num_offsets, 
                               use_min_count=use_min_count, seed=seed)
    
    valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(
        sizes, counts, mode=mode, use_weighted_fit=use_weighted_fit, 
        use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, 
        n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
    
    return {'D': d_value, 'valid_sizes': valid_sizes, 'valid_counts': valid_counts, 
            'fit': fit, 'R2': r2, 'ci_low': ci_low, 'ci_high': ci_high} 