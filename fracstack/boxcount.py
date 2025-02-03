import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import r2_score
from .image_processing import invert_array
import pandas as pd
import multiprocessing
from scipy.stats import t


from numba import njit, prange
import numpy as np
from multiprocessing import Pool
from tqdm.auto import tqdm

@njit(nogil=True, parallel=True, cache=True)
def numba_d0(array, sizes, num_offsets):
    results = np.empty(len(sizes), dtype=np.int64)
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        H, W = array.shape
        
        # Generate offsets directly in Numba
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(num_offsets, size**2)
        
        min_count = np.inf
        for offset_idx in range(total_offsets):
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                rand = np.random.randint(1, size**2)
                x_off = rand % size
                y_off = rand // size
            
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
    
    return results

@njit(nogil=True, parallel=True, cache=True)
def numba_d1(array, sizes, num_offsets):
    results = np.empty(len(sizes), dtype=np.float64)
    M = array.sum()
    H, W = array.shape
    
    for idx in prange(len(sizes)):
        size = sizes[idx]
        if M == 0 or size == 0:
            results[idx] = 0.0
            continue
            
        # Generate offsets directly in Numba
        centered_x = (H % size) // 2
        centered_y = (W % size) // 2
        total_offsets = min(num_offsets, size**2)
        entropy_sum = 0.0
        
        for offset_idx in range(total_offsets):
            # Get offset coordinates
            if offset_idx == 0:
                x_off = centered_x
                y_off = centered_y
            else:
                rand = np.random.randint(1, size**2)
                x_off = rand % size
                y_off = rand // size
            
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

def boxcount(array, mode='D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1):
    """Unified boxcount supporting both D0 and D1"""
    array = np.ascontiguousarray(array.astype(np.float32))
    min_size = 1 if min_size is None else min_size
    max_size = max(min_size + 1, min(array.shape)//5) if max_size is None else max_size
    sizes = get_sizes(num_sizes, min_size, max_size)
    sizes_arr = np.array(sizes)
    
    if mode == 'D0':
        counts = numba_d0(array, sizes_arr, num_offsets)
    elif mode == 'D1':
        counts = numba_d1(array, sizes_arr, num_offsets)
    else:
        raise ValueError("Invalid mode, use 'D0' or 'D1'")
    
    return sizes, counts.tolist()


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


def compute_dimension(sizes, measures, mode = 'D0'):
    """
    Compute a generalized dimension from box sizes and corresponding measures.
    
    Performs a linear fit on log-log data to determine the scaling exponent.
    The dimension is the negative slope of this fit. This function can be used
    for various dimension calculations including box-counting dimension,
    information dimension, and correlation dimension.
    
    Args:
        sizes (array-like): Box sizes used in the scaling analysis
        measures (array-like): Corresponding measures for each box size
            (e.g., box counts, entropy sums, correlation sums)
        
    Returns:
        tuple: Contains:
            - valid_sizes (np.ndarray): Box sizes after filtering out non-positive values
            - valid_measures (np.ndarray): Measures after filtering out non-positive values  
            - d_value (float): Computed dimension (-slope of log-log fit)
            - fit (np.ndarray): Parameters [slope, intercept] of linear fit
            - r2 (float): R-squared value indicating goodness of fit
            
    Notes:
        - Non-positive sizes and measures are filtered out before fitting
        - Returns NaN values if insufficient valid data points for fitting
    """
    
    # Convert to numpy arrays if not already
    sizes = np.array(sizes)
    measures = np.array(measures)
    
    # Filter out non-positive sizes and counts
    valid = (sizes > 0) & (measures > 0)
    valid_sizes = sizes[valid]
    valid_measures = measures[valid]
    
    if len(valid_sizes) > 0 and len(valid_measures) > 0:
        # Perform linear fit on log-log data
        if mode == 'D0':
            fit, cov = np.polyfit(np.log10(valid_sizes), np.log10(valid_measures), 1, cov=True)
            r2 = r2_score(np.log10(valid_measures), fit[0] * np.log10(valid_sizes) + fit[1])
        elif mode == 'D1':
            fit, cov = np.polyfit(-np.log2(1/valid_sizes), valid_measures, 1, cov=True)
            r2 = r2_score(valid_measures, fit[0] * -np.log2(1/valid_sizes) + fit[1])
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'D0' or 'D1'")
        
        d_value = -fit[0] 



        # --- Compute 95% CI for slope and intercept ---
        # 1) Number of points and degrees of freedom
        N = len(valid_sizes)
        dof = N - 2  # for a linear fit with two parameters

        # 2) Critical t-value at 95% confidence
        alpha = 0.05
        t_crit = t.ppf(1 - alpha/2, dof)  # two-sided

        # 3) Standard errors for slope (fit[0]) and intercept (fit[1])
        slope_se = np.sqrt(cov[0, 0])
        intercept_se = np.sqrt(cov[1, 1])

        # 4) Confidence intervals
        slope_ci = (fit[0] - t_crit * slope_se, fit[0] + t_crit * slope_se)
        #intercept_ci = (fit[1] - t_crit * intercept_se, fit[1] + t_crit * intercept_se)

        # 5) If you want the CI for d_value = -slope, just flip signs and ensure lower value comes first
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
    

## GRAVEYARD ##

# def get_coverage_mincount(array, size, mask=None, num_offsets=1):
#     """
#     Vectorized count of the minimum number of boxes covering non-zero regions of the array,
#     constrained by an optional mask, considering multiple random grid offsets.

#     Args:
#         array (np.ndarray): 2D binary numpy array to analyze.   
#         size (int): Size of the boxes (size x size).
#         mask (np.ndarray, optional): Optional 2D binary mask. Defaults to None.
#         num_offsets (int, optional): Number of random grid offsets to test (includes centered grid).
#                                      If num_offsets <= 1, only the centered grid is tested.
#                                      Defaults to 1.

#     Returns:
#         int: Minimum number of boxes covering non-zero regions across all offsets.
#     """
#     shape = array.shape
#     counts = []

#     # Step 1: Calculate the centered grid offset
#     centered_x_offset = (shape[0] % size) // 2
#     centered_y_offset = (shape[1] % size) // 2

#     # Step 2: Determine total possible unique offsets
#     total_possible_offsets = size ** 2
#     num_offsets = min(num_offsets, total_possible_offsets)  # Cap the number of offsets

#     # Step 3: Generate random offsets if num_offsets > 1
#     if num_offsets > 1 and total_possible_offsets > 1:
#         # Exclude the centered grid from random offsets
#         random_indices = np.random.choice(range(1, total_possible_offsets), size=num_offsets - 1, replace=False)
#         offsets_x = random_indices % size
#         offsets_y = random_indices // size
#     else:
#         offsets_x = np.array([], dtype=int)
#         offsets_y = np.array([], dtype=int)

#     # Include the centered grid as the first offset
#     offsets_x = np.concatenate(([centered_x_offset], offsets_x)) if num_offsets > 0 else np.array([centered_x_offset])
#     offsets_y = np.concatenate(([centered_y_offset], offsets_y)) if num_offsets > 0 else np.array([centered_y_offset])

#     for x_offset, y_offset in zip(offsets_x, offsets_y):
#         # Calculate the number of complete boxes along each dimension
#         num_boxes_x = (shape[0] - x_offset) // size
#         num_boxes_y = (shape[1] - y_offset) // size

#         if num_boxes_x == 0 or num_boxes_y == 0:
#             counts.append(0)
#             continue

#         # Calculate the end indices to ensure complete boxes
#         end_x = x_offset + num_boxes_x * size
#         end_y = y_offset + num_boxes_y * size

#         # Slice the array to extract complete boxes
#         sliced_array = array[x_offset:end_x, y_offset:end_y]

#         # Reshape to (num_boxes_x, num_boxes_y, size, size)
#         reshaped = sliced_array.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)

#         if mask is not None:
#             # Slice and reshape the mask similarly
#             sliced_mask = mask[x_offset:end_x, y_offset:end_y]
#             reshaped_mask = sliced_mask.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)

#             # Determine which boxes have at least one mask element and one non-zero element
#             box_has_mask = reshaped_mask.any(axis=(2, 3))
#             box_has_content = reshaped.any(axis=(2, 3))
#             valid_boxes = box_has_mask & box_has_content
#             count = np.sum(valid_boxes)
#         else:
#             # Determine which boxes have at least one non-zero element
#             box_has_content = reshaped.any(axis=(2, 3))
#             count = np.sum(box_has_content)

#         counts.append(count)

#     # Return the minimum count across all offsets
#     return min(counts) if counts else 0

# def get_coverage_entropies(array, size, mask=None, num_offsets=1):
#     shape = array.shape
#     M = np.sum(array)
#     offset_entropies = []

#     centered_x_offset = (shape[0] % size) // 2
#     centered_y_offset = (shape[1] % size) // 2

#     total_possible_offsets = size ** 2
#     num_offsets = min(num_offsets, total_possible_offsets)

#     if num_offsets > 1 and total_possible_offsets > 1:
#         random_indices = np.random.choice(range(1, total_possible_offsets), size=num_offsets - 1, replace=False)
#         offsets_x = random_indices % size
#         offsets_y = random_indices // size
#     else:
#         offsets_x, offsets_y = np.array([], dtype=int), np.array([], dtype=int)

#     offsets_x = np.concatenate(([centered_x_offset], offsets_x)) if num_offsets > 0 else np.array([centered_x_offset])
#     offsets_y = np.concatenate(([centered_y_offset], offsets_y)) if num_offsets > 0 else np.array([centered_y_offset])

#     for x_offset, y_offset in zip(offsets_x, offsets_y):
#         num_boxes_x = (shape[0] - x_offset) // size
#         num_boxes_y = (shape[1] - y_offset) // size

#         if num_boxes_x == 0 or num_boxes_y == 0:
#             continue

#         end_x = x_offset + num_boxes_x * size
#         end_y = y_offset + num_boxes_y * size

#         sliced_array = array[x_offset:end_x, y_offset:end_y]
#         reshaped = sliced_array.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)

#         if mask is not None:
#             sliced_mask = mask[x_offset:end_x, y_offset:end_y]
#             reshaped_mask = sliced_mask.reshape(num_boxes_x, size, num_boxes_y, size).transpose(0, 2, 1, 3)
#             box_has_mask = reshaped_mask.any(axis=(2, 3))
#             box_has_content = reshaped.any(axis=(2, 3))
#             valid_boxes = box_has_mask & box_has_content
#         else:
#             valid_boxes = reshaped.any(axis=(2, 3))

#         sum_per_box = reshaped.sum(axis=(2, 3))
#         p_per_box = sum_per_box / M
#         valid_p_mask = valid_boxes & (sum_per_box > 0)
#         valid_p = p_per_box[valid_p_mask]

#         if valid_p.size > 0:
#             shannon_entropies = -valid_p * np.log2(valid_p)
#             offset_entropy = np.sum(shannon_entropies)
#             # Assertion check remains if necessary
#             assert np.abs(np.sum(valid_p) - sum_per_box[valid_p_mask].sum() / M) < 1e-6, "Consistency check failed"
#         else:
#             offset_entropy = 0.0

#         offset_entropies.append(offset_entropy)

#     return np.mean(offset_entropies) if offset_entropies else 0.0

# def _process_size_for_boxcount(args):
#     """Helper function for multiprocessing box counting.
#     Args should be a tuple of (array, size, num_offsets, mode)"""
#     array, size, num_offsets, mode = args
#     if mode == 'D0':
#         return get_coverage_mincount(array, size, num_offsets=num_offsets)
#     elif mode == 'D1':
#         return get_coverage_entropies(array, size, num_offsets=num_offsets)
#     else:
#         raise ValueError(f"Invalid mode: {mode}. Use 'D0' or 'D1'")

# def multiprocess_boxcount(array, sizes, mode, num_offsets=1):
#     """
#     Run box counting in parallel for multiple box sizes.

#     Args:
#         array (np.ndarray): 2D binary array to analyze.
#         sizes (list): List of box sizes (int) to compute.
#         mode (str): Either 'D0' for capacity dimension or 'D1' for information dimension.
#         num_offsets (int, optional): Number of random grid offsets. Defaults to 1.

#     Returns:
#         tuple: (sizes, counts) arrays in sorted order
#     """
#     # Create argument tuples for each size
#     args = [(array, size, num_offsets, mode) for size in sizes]

#     # Launch multiprocessing for each size
#     with multiprocessing.Pool() as pool:
#         results = pool.map(_process_size_for_boxcount, args)

#     # Sort sizes and ensure counts match the order
#     sorted_sizes = np.array(sorted(sizes))
#     counts = np.array([results[list(sizes).index(size)] for size in sorted_sizes])
    
#     return sorted_sizes, counts

# def boxcount(array, mode = 'D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1, invert=False, mask=None, multiprocessing=True):
#     """
#     Perform box-count analysis, optionally constrained to a specific mask.

#     Args:
#         array (np.ndarray): 2D binary numpy array (counts elements that aren't 0, or elements that aren't 1 if inverted).
#         num_sizes (int): Number of box sizes.
#         min_size (int, optional): Smallest box size in pixels (defaults to 1).
#         max_size (int, optional): Largest box size in pixels (defaults to 1/5 smaller dimension of array).
#         invert (bool, optional): If True, invert the binary array. Defaults to False.
#         mask (np.ndarray, optional): Optional binary mask (1 for valid regions, 0 for excluded regions).
#         multiprocessing (bool, optional): If True, use multiprocessing for D0 calculation. Defaults to False.

#     Returns:
#         tuple: (sizes, counts) where sizes is a list of box sizes and counts is a list of box counts.
#     """
#     if invert:
#         array = invert_array(array)

#     # Apply mask if provided
#     if mask is not None:
#         array = array * mask  # Exclude regions outside the mask

#     min_size = 1 if min_size is None else min_size
#     max_size = max(min_size + 1, min(array.shape) // 5) if max_size is None else max_size
#     sizes = get_sizes(num_sizes, min_size, max_size)
#     counts = []


#     if multiprocessing == True:
#         sizes,counts = multiprocess_boxcount(array, sizes, mode, num_offsets)
    
#     else:
#         for size in tqdm(sizes, desc=f'Calculating {mode}', leave=False):
#             if mode == 'D0':
#                 counts.append(get_coverage_mincount(array, size, mask, num_offsets=num_offsets))
#             elif mode == 'D1':
#                 counts.append(get_coverage_entropies(array, size, mask, num_offsets=num_offsets))
#             else:
#                 raise ValueError(f"Invalid mode: {mode}. Use 'D0' or 'D1'")
        
#     return sizes, counts

# def sliding_decade_analysis(array, mode='D0', num_sizes=10, num_offsets=10, 
#                             invert=False, mask=None, min_decade=1.0):


#     # 1) Get global sizes/counts
#     sizes, counts = boxcount(
#         array, mode, num_sizes, 
#         min_size=8, 
#         max_size=min(array.shape)//5, 
#         num_offsets=num_offsets, 
#         invert=invert, mask=mask
#     )

#     # Make sure sizes are sorted ascending
#     # (They usually are from boxcount, but let's be safe.)
#     # We'll pair them in a single list to keep them in sync 
#     size_count_pairs = sorted(zip(sizes, counts), key=lambda x: x[0])
#     sizes_sorted = [sc[0] for sc in size_count_pairs]
#     counts_sorted = [sc[1] for sc in size_count_pairs]

#     # 2) Slide over possible i
#     i = 0
#     results = []
#     while i < len(sizes_sorted):
#         smin = sizes_sorted[i]
#         # If smin == 0, skip
#         if smin == 0:
#             i += 1
#             continue
        
#         # 3) Find the largest j s.t. sizes_sorted[j] / smin >= 10^min_decade
#         j = i
#         while j < len(sizes_sorted):
#             ratio = sizes_sorted[j] / smin
#             if ratio < 10**min_decade:
#                 j += 1
#             else:
#                 break
        
#         # If j is out of range, we won't find a full decade
#         if j >= len(sizes_sorted):
#             break

#         # 4) subrange from i..j
#         sub_sizes = sizes_sorted[i:j+1]
#         sub_counts = counts_sorted[i:j+1]
        
#         valid_sizes, valid_counts, d_value, fit, r2 = compute_dimension(
#             sub_sizes, sub_counts, mode
#         )
        
#         srange = (valid_sizes[0], valid_sizes[-1]) if len(valid_sizes)>0 else (None, None)

#         results.append({
#             'min_size': srange[0],
#             'max_size': srange[1],
#             'D': d_value,
#             'R2': r2
#         })

#         i += 1  # or i += some increment

#     # 6) Convert to DataFrame

#     results_df = pd.DataFrame(results)

#     return results_df

# def get_mass_scaling(array, size, mask=None, num_offsets=1):
#     """
#     Compute the total 'mass' in boxes of shape (size x size) for a 2D array,
#     optionally restricted by a binary mask, averaged over multiple offsets.

#     The 'mass' is defined as the sum of array values within each valid box.
#     In a binary image, this is simply the count of '1' pixels in each valid box.
    
#     Args:
#         array (np.ndarray): 2D numpy array (e.g., binary fractal image).
#         size (int): The box size (size x size).
#         mask (np.ndarray, optional): A 2D binary mask. If provided, a box is 
#                                      considered valid only if it overlaps 
#                                      mask=1. Defaults to None.
#         num_offsets (int, optional): Number of random grid offsets to test, 
#                                      in addition to a 'centered' offset.
#                                      Defaults to 1 (only centered grid).
    
#     Returns:
#         float: The average total mass (sum of array values in all valid boxes)
#                across all offsets tested. Returns 0.0 if no valid boxes.
#     """
#     shape = array.shape
#     offset_masses = []

#     # Step 1: Centered offset
#     centered_x_offset = (shape[0] % size) // 2
#     centered_y_offset = (shape[1] % size) // 2

#     # Step 2: Total possible offsets (size^2 distinct (x_offset, y_offset) pairs)
#     total_possible_offsets = size ** 2
#     num_offsets = min(num_offsets, total_possible_offsets)
    
#     # Step 3: Generate random offsets if needed
#     # We'll exclude the centered offset from random picks
#     if num_offsets > 1 and total_possible_offsets > 1:
#         random_indices = np.random.choice(
#             range(1, total_possible_offsets), 
#             size=num_offsets - 1, 
#             replace=False
#         )
#         offsets_x = random_indices % size
#         offsets_y = random_indices // size
#     else:
#         offsets_x = np.array([], dtype=int)
#         offsets_y = np.array([], dtype=int)

#     # Always include the centered grid as the first offset
#     offsets_x = np.concatenate(([centered_x_offset], offsets_x))
#     offsets_y = np.concatenate(([centered_y_offset], offsets_y))

#     # Loop over each offset
#     for x_offset, y_offset in zip(offsets_x, offsets_y):
#         # Number of complete boxes in x and y
#         num_boxes_x = (shape[0] - x_offset) // size
#         num_boxes_y = (shape[1] - y_offset) // size

#         if num_boxes_x == 0 or num_boxes_y == 0:
#             offset_masses.append(0.0)
#             continue
        
#         # End indices for slicing to get complete boxes
#         end_x = x_offset + num_boxes_x * size
#         end_y = y_offset + num_boxes_y * size
        
#         # Slice the array
#         sliced_array = array[x_offset:end_x, y_offset:end_y]
#         # Reshape -> (num_boxes_x, size, num_boxes_y, size) -> transpose
#         reshaped = sliced_array.reshape(num_boxes_x, size, num_boxes_y, size)
#         reshaped = reshaped.transpose(0, 2, 1, 3)  # shape: (Bx, By, size, size)

#         if mask is not None:
#             sliced_mask = mask[x_offset:end_x, y_offset:end_y]
#             reshaped_mask = sliced_mask.reshape(num_boxes_x, size, num_boxes_y, size)
#             reshaped_mask = reshaped_mask.transpose(0, 2, 1, 3)
            
#             # A valid box is one that overlaps mask=1
#             # and presumably has some mass in the array
#             box_has_mask = reshaped_mask.any(axis=(2,3))
#             box_mass = reshaped.sum(axis=(2,3))  # sum of array values
#             # We only sum the mass in boxes that overlap the mask
#             total_mass = np.sum(box_mass[box_has_mask])
#         else:
#             # Sum all array values in all boxes
#             total_mass = np.sum(reshaped)
        
#         offset_masses.append(total_mass)

#     # Return the average total mass across all tested offsets
#     if len(offset_masses) > 0:
#         return np.mean(offset_masses)
#     else:
#         return 0.0

