import time
import numpy as np
from numba import njit, prange
from .boxcount import generate_random_offsets, get_sizes, compute_dimension
import torch

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

def build_svt_torch(mask: torch.Tensor) -> torch.Tensor:
    """Build summed volume table with proper device management."""
    # Ensure we're working with the right dtype and device
    if mask.dtype != torch.uint8:
        mask = mask.to(torch.uint8)
    
    # Use more memory-efficient cumulative sum
    svt = mask.int()
    
    # Perform cumulative sums in-place to save memory
    svt = svt.cumsum(dim=0)
    svt = svt.cumsum(dim=1) 
    svt = svt.cumsum(dim=2)
    
    # Add 1-voxel guard band
    svt = torch.nn.functional.pad(svt, (1, 0, 1, 0, 1, 0))
    
    return svt

def counts_with_pool(mask, sizes, offsets, use_min):
    # mask is bool on CUDA
    counts = []
    for s, offs in zip(sizes, offsets):
        acc, best = 0, 1e18
        for z_off, y_off, x_off in offs:
            # slice so (z_off + k*s) fits; fastest on GPU if strides line up
            pooled = torch.nn.functional.max_pool3d(
                mask[...,
                     z_off:mask.size(0) - ((mask.size(0) - z_off) % s),
                     y_off:mask.size(1) - ((mask.size(1) - y_off) % s),
                     x_off:mask.size(2) - ((mask.size(2) - x_off) % s)],
                kernel_size=s, stride=s)
            cnt = pooled.sum().item()
            if use_min:
                best = cnt if cnt < best else best
            else:
                acc += cnt
        counts.append(best if use_min else acc / len(offs))
    return torch.as_tensor(counts, dtype=torch.float64, device='cpu')

def counts_from_svt_torch(svt: torch.Tensor, sizes, offsets, use_min):
    """
    Torch-based parallel cube counts from a summed-volume table (SVT).
    
    Parameters
    ----------
    svt : torch.Tensor
        (Z+1, Y+1, X+1) summed-volume table from build_svt_torch
    sizes : np.ndarray or torch.Tensor
        (S,) array of cube edge lengths
    offsets : np.ndarray or torch.Tensor
        (S, M, 3) array of (z_off, y_off, x_off) triplets for each size
    use_min : bool
        If True, take minimum count over offsets; otherwise take mean
        
    Returns
    -------
    torch.Tensor
        (S,) tensor of box counts for each size, on CPU as float64
    """
    # Convert inputs to torch tensors if needed
    if isinstance(sizes, np.ndarray):
        sizes = torch.from_numpy(sizes).to(svt.device)
    if isinstance(offsets, np.ndarray):
        offsets = torch.from_numpy(offsets).to(svt.device)
    
    S = sizes.shape[0]
    Zm, Ym, Xm = svt.shape[0] - 1, svt.shape[1] - 1, svt.shape[2] - 1
    
    # Pre-allocate output tensor
    out = torch.empty(S, dtype=torch.float64, device=svt.device)
    
    for i in range(S):
        s = sizes[i].item()
        offs = offsets[i]
        m = offs.shape[0]
        
        best = 1e18
        acc = 0.0
        
        for j in range(m):
            # Normalize offsets so 0 ≤ off < s (wraparound)
            z_off = int(offs[j, 0].item()) % s
            y_off = int(offs[j, 1].item()) % s
            x_off = int(offs[j, 2].item()) % s
            
            # Eight SVT corner sub-arrays using torch slicing
            c111 = svt[z_off + s : Zm + 1 : s,
                       y_off + s : Ym + 1 : s,
                       x_off + s : Xm + 1 : s]
            
            c011 = svt[z_off : Zm + 1 - s : s,
                       y_off + s : Ym + 1 : s,
                       x_off + s : Xm + 1 : s]
            
            c101 = svt[z_off + s : Zm + 1 : s,
                       y_off : Ym + 1 - s : s,
                       x_off + s : Xm + 1 : s]
            
            c110 = svt[z_off + s : Zm + 1 : s,
                       y_off + s : Ym + 1 : s,
                       x_off : Xm + 1 - s : s]
            
            c001 = svt[z_off : Zm + 1 - s : s,
                       y_off : Ym + 1 - s : s,
                       x_off + s : Xm + 1 : s]
            
            c010 = svt[z_off : Zm + 1 - s : s,
                       y_off + s : Ym + 1 : s,
                       x_off : Xm + 1 - s : s]
            
            c100 = svt[z_off + s : Zm + 1 : s,
                       y_off : Ym + 1 - s : s,
                       x_off : Xm + 1 - s : s]
            
            c000 = svt[z_off : Zm + 1 - s : s,
                       y_off : Ym + 1 - s : s,
                       x_off : Xm + 1 - s : s]
            
            # Inclusion-exclusion computation
            buf = c111 - c011 - c101 - c110 + c001 + c010 + c100 - c000
            
            cnt = (buf > 0).sum().item()  # occupied cubes for this offset
            
            if use_min:
                if cnt < best:
                    best = cnt
            else:
                acc += cnt
        
        out[i] = best if use_min else acc / m
    
    # Return on CPU as float64 for compatibility with existing code
    return out.cpu().double()

def chunked_boxcount_3d_torch(array, sizes, offsets, use_min_count=False, chunk_size=256):
    """
    Memory-efficient chunked processing for very large 3D arrays.
    
    Parameters
    ----------
    array : np.ndarray
        3D binary array to analyze
    sizes : np.ndarray
        Array of box sizes to test
    offsets : np.ndarray
        Offset arrays for each size
    use_min_count : bool
        Whether to use minimum count across offsets
    chunk_size : int
        Size of chunks to process (in voxels per dimension)
        
    Returns
    -------
    np.ndarray
        Box counts for each size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D, H, W = array.shape
    
    # Initialize results
    results = np.zeros(len(sizes), dtype=np.float64)
    
    # Process in chunks to avoid OOM
    for z_start in range(0, D, chunk_size):
        for y_start in range(0, H, chunk_size):
            for x_start in range(0, W, chunk_size):
                z_end = min(z_start + chunk_size, D)
                y_end = min(y_start + chunk_size, H)
                x_end = min(x_start + chunk_size, W)
                
                # Extract chunk
                chunk = array[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Skip empty chunks
                if not np.any(chunk):
                    continue
                
                # Process chunk with torch
                try:
                    chunk_torch = torch.from_numpy(chunk.astype(np.uint8)).to(device)
                    
                    # Build SVT for this chunk
                    svt_chunk = build_svt_torch(chunk_torch)
                    del chunk_torch
                    
                    # Adjust offsets for this chunk's coordinate system
                    chunk_offsets = offsets.copy()
                    
                    # Compute counts for this chunk
                    chunk_counts = counts_from_svt_torch(svt_chunk, sizes, chunk_offsets, use_min_count)
                    
                    # Accumulate results
                    results += chunk_counts.numpy()
                    
                    # Clean up
                    del svt_chunk, chunk_counts
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM on chunk at ({z_start}, {y_start}, {x_start}), skipping...")
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
    
    return results

def boxcount_3d(array, mode='D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1, 
                use_min_count=False, use_integral_image=False, use_torch=False, seed=None, 
                force_chunked=False, chunk_size=256):
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
    use_integral_image : bool, default False
        Whether to use integral image optimization. If True, uses SVT-based methods.
    use_torch : bool, default False
        Whether to use Torch implementation when use_integral_image=True.
        Requires PyTorch to be installed.
    seed : int, optional
        Random seed for reproducible grid offset generation
    force_chunked : bool, default False
        Force use of chunked processing for memory efficiency
    chunk_size : int, default 256
        Size of chunks for chunked processing (voxels per dimension)
        
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
    
    
    if array.ndim == 2 and array.shape[1] == 3:
        print("Detected coordinate inputs, using counts_from_points method")
        use_integral_image = False
        use_torch = False
    
    # Memory estimation for torch SVT
    if use_torch and use_integral_image:
        # Estimate memory usage for SVT (shape will be array.shape + 1)
        svt_elements = np.prod([d + 1 for d in array.shape])
        # SVT uses int32, so 4 bytes per element
        estimated_memory_gb = (svt_elements * 4) / (1024**3)
        print(f"Estimated SVT memory usage: {estimated_memory_gb:.2f} GB")
        
        # Force chunked processing for very large arrays
        if estimated_memory_gb > 8.0 or force_chunked:
            if not force_chunked:
                print("Large memory usage detected, forcing chunked processing...")
            force_chunked = True
    
    if mode == 'D0':
        
        if array.ndim == 2 and array.shape[1] == 3:
            print("Detected coordinate inputs, using point‑cloud box counter")

            # geometric progression of ε just like before
            sizes = get_sizes(num_sizes, min_size, max_size)

            rng = np.random.default_rng(seed)      # honour the seed argument
            counts = counts_from_points(
                        array.astype(np.float64),   # keep full precision
                        sizes,
                        num_offsets=num_offsets,
                        use_min=use_min_count,
                        rng=rng)

            return sizes, counts.tolist()
        
        if use_integral_image:
            if use_torch:
                # Check if we should use chunked processing
                if force_chunked:
                    print("Using chunked processing for memory efficiency...")
                    start_time = time.perf_counter()
                    counts = chunked_boxcount_3d_torch(array, sizes_arr, offsets, use_min_count, chunk_size)
                    end_time = time.perf_counter()
                    print(f"Time taken with chunked processing: {end_time - start_time} seconds")
                else:
                    # Torch-based SVT path with device management
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    print(f"Using device: {device}")
                    
                    try:
                        start_time = time.perf_counter()
                        mask_torch = torch.from_numpy(array.astype(np.uint8)).to(device)
                        
                        # For very large arrays, try to free memory and use chunked processing
                        if mask_torch.numel() > 500_000_000:  # ~500M elements
                            print("Large array detected, using memory-efficient processing...")
                            torch.cuda.empty_cache() if device.type == 'cuda' else None
                        
                        svt = build_svt_torch(mask_torch)
                        
                        # Free the original mask to save memory
                        del mask_torch
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        end_time = time.perf_counter()
                        print(f"Time taken to build summed volume table (Torch): {end_time - start_time} seconds")

                        start_time = time.perf_counter()
                        counts_torch = counts_from_svt_torch(svt, sizes_arr, offsets, use_min_count)
                        counts = counts_torch.numpy()
                        
                        # Clean up GPU memory
                        del svt, counts_torch
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        end_time = time.perf_counter()
                        print(f"Time taken to compute box counts (Torch): {end_time - start_time} seconds")
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("CUDA OOM detected, trying CPU fallback...")
                            torch.cuda.empty_cache()
                            device = torch.device("cpu")
                            
                            try:
                                # Retry with CPU
                                start_time = time.perf_counter()
                                mask_torch = torch.from_numpy(array.astype(np.uint8)).to(device)
                                svt = build_svt_torch(mask_torch)
                                del mask_torch
                                end_time = time.perf_counter()
                                print(f"Time taken to build summed volume table (CPU fallback): {end_time - start_time} seconds")

                                start_time = time.perf_counter()
                                counts_torch = counts_from_svt_torch(svt, sizes_arr, offsets, use_min_count)
                                counts = counts_torch.numpy()
                                del svt, counts_torch
                                end_time = time.perf_counter()
                                print(f"Time taken to compute box counts (CPU fallback): {end_time - start_time} seconds")
                                
                            except RuntimeError as e2:
                                if "out of memory" in str(e2).lower():
                                    print("CPU also OOM, falling back to chunked processing...")
                                    start_time = time.perf_counter()
                                    counts = chunked_boxcount_3d_torch(array, sizes_arr, offsets, use_min_count)
                                    end_time = time.perf_counter()
                                    print(f"Time taken with chunked processing: {end_time - start_time} seconds")
                                else:
                                    raise  # Re-raise non-OOM errors
                        else:
                            raise  # Re-raise non-OOM errors
            else:
                # NumPy/Numba SVT path
                start_time = time.perf_counter()
                sat = build_svt(array)
                end_time = time.perf_counter()
                print(f"Time taken to build summed area table: {end_time - start_time} seconds")

                start_time = time.perf_counter()
                counts = counts_from_svt(sat, sizes_arr, offsets, use_min_count)
                end_time = time.perf_counter()
                print(f"Time taken to compute box counts: {end_time - start_time} seconds")
        else:
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
                        use_integral_image=False,
                        use_torch=False,
                        seed=None, 
                        use_weighted_fit=True,
                        use_bootstrap_ci=False,
                        bootstrap_method='residual',
                        n_bootstrap=1000,
                        bootstrap_seed=None,
                        force_chunked=False,
                        chunk_size=256):
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
    use_integral_image : bool, default False
        Whether to use integral image optimization. If True, uses SVT-based methods.
    use_torch : bool, default False
        Whether to use Torch implementation when use_integral_image=True.
        Requires PyTorch to be installed.
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
    force_chunked : bool, default False
        Force use of chunked processing for memory efficiency
    chunk_size : int, default 256
        Size of chunks for chunked processing (voxels per dimension)
        
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
    if input_array.dtype != bool and input_array.dtype != np.uint8:
        # Check if values are already binary (0s and 1s only) without creating large arrays
        max_val = np.max(input_array)
        min_val = np.min(input_array)
        if max_val <= 1 and min_val >= 0 and np.all((input_array == 0) | (input_array == 1)):
            input_array = input_array.astype(np.uint8)
        else:
            input_array = input_array.astype(bool).astype(np.uint8)
    
    if max_size is None:
        max_size = min(input_array.shape) // 5
        
    sizes, counts = boxcount_3d(input_array, mode=mode, min_size=min_size, max_size=max_size, 
                               num_sizes=num_sizes, num_offsets=num_offsets, 
                               use_min_count=use_min_count, use_integral_image=use_integral_image, 
                               use_torch=use_torch, seed=seed, force_chunked=force_chunked, 
                               chunk_size=chunk_size)
    
    valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(
        sizes, counts, mode=mode, use_weighted_fit=use_weighted_fit, 
        use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, 
        n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
    
    return {'D': d_value, 'valid_sizes': valid_sizes, 'valid_counts': valid_counts, 
            'fit': fit, 'R2': r2, 'ci_low': ci_low, 'ci_high': ci_high} 
    
    

import numpy as np
from numba import njit, prange

# ──────────────────────────────────────────────────────────────────────────────
# Summed‑volume table (SVT) with a one‑voxel zero border
# ──────────────────────────────────────────────────────────────────────────────
def build_svt(mask: np.ndarray) -> np.ndarray:
    """Summed‑volume table with 1‑voxel zero border (NumPy‑only)."""
    Z, Y, X = mask.shape               # depth, height, width
    svt = np.zeros((Z + 1, Y + 1, X + 1), dtype=np.uint32)
    svt[1:, 1:, 1:] = mask.astype(np.uint8)

    # Three in‑place cumulative passes → C‑speed
    svt.cumsum(axis=0, out=svt)
    svt.cumsum(axis=1, out=svt)
    svt.cumsum(axis=2, out=svt)
    return svt


# ──────────────────────────────────────────────────────────────────────────────
# Parallel box counts from an SVT
# ──────────────────────────────────────────────────────────────────────────────
@njit(parallel=True, fastmath=True, cache=True)
def counts_from_svt(svt, sizes, offsets, use_min):
    """
    Parallel cube counts from a summed‑volume table (SVT).

    Parameters
    ----------
    svt      : (Z+1, Y+1, X+1) uint32  –  summed‑volume table
    sizes    : (S,)            int32   –  cube edge lengths
    offsets  : (S, M, 3)       int32   –  (z_off, y_off, x_off) triplets for each size
    use_min  : bool                     minimise vs. average across offsets
    """
    S  = sizes.shape[0]
    out = np.empty(S, dtype=np.float64)
    Zm, Ym, Xm = svt.shape[0] - 1, svt.shape[1] - 1, svt.shape[2] - 1   # mask extents

    for i in prange(S):                       # ← parallel over cube sizes
        s    = sizes[i]
        offs = offsets[i]
        m    = offs.shape[0]

        best = 1e18          # large number for min‑reduction
        acc  = 0.0           # running sum for mean‑reduction

        for j in prange(m):
            # Normalise offsets so 0 ≤ off < s (wraparound)
            z_off = int(offs[j, 0]) % s
            y_off = int(offs[j, 1]) % s
            x_off = int(offs[j, 2]) % s

            # Eight SVT corner sub‑arrays (all the same shape)
            c111 = svt[z_off + s : Zm + 1     : s,
                        y_off + s : Ym + 1     : s,
                        x_off + s : Xm + 1     : s]

            c011 = svt[z_off      : Zm + 1 - s : s,
                        y_off + s : Ym + 1     : s,
                        x_off + s : Xm + 1     : s]

            c101 = svt[z_off + s : Zm + 1     : s,
                        y_off      : Ym + 1 - s : s,
                        x_off + s : Xm + 1     : s]

            c110 = svt[z_off + s : Zm + 1     : s,
                        y_off + s : Ym + 1     : s,
                        x_off      : Xm + 1 - s : s]

            c001 = svt[z_off      : Zm + 1 - s : s,
                        y_off      : Ym + 1 - s : s,
                        x_off + s : Xm + 1     : s]

            c010 = svt[z_off      : Zm + 1 - s : s,
                        y_off + s : Ym + 1     : s,
                        x_off      : Xm + 1 - s : s]

            c100 = svt[z_off + s : Zm + 1     : s,
                        y_off      : Ym + 1 - s : s,
                        x_off      : Xm + 1 - s : s]

            c000 = svt[z_off      : Zm + 1 - s : s,
                        y_off      : Ym + 1 - s : s,
                        x_off      : Xm + 1 - s : s]

            # Inclusion–exclusion in‑place on a scratch buffer
            buf = np.empty(c111.shape, dtype=np.uint32)
            buf[:]  = c111
            buf -= c011
            buf -= c101
            buf -= c110
            buf += c001
            buf += c010
            buf += c100
            buf -= c000

            cnt = (buf > 0).sum()             # occupied cubes for this offset

            if use_min:
                if cnt < best:
                    best = cnt
            else:
                acc += cnt

        out[i] = best if use_min else acc / m

    return out

import numpy as np

def counts_from_points(points, sizes, num_offsets=1, use_min=False, rng=None):
    """
    Parameters
    ----------
    points      (N,3) float64/32  – raw coordinates
    sizes       1‑D array of ε (box edges, same unit as points)
    num_offsets int               – random grid shifts per ε
    use_min     bool              – True → min across offsets, False → mean
    rng         np.random.Generator or None

    Returns
    -------
    counts      1‑D float64       – N(ε) for every ε in `sizes`
    """
    if rng is None:
        rng = np.random.default_rng()

    # shift cloud so the min corner is at (0,0,0); keeps indices non‑negative
    pts = points - points.min(axis=0, keepdims=True)

    counts = []
    for eps in sizes:
        c_per_offset = []
        for _ in range(num_offsets):
            delta = rng.uniform(0, eps, size=(1, 3))      # random offset 0 ≤ Δ < ε
            idx   = np.floor((pts + delta) / eps).astype(np.int64)

            # unique rows – the slow but bullet‑proof way
            uniq = np.unique(idx.view([('', idx.dtype)]*3))
            c_per_offset.append(uniq.size)

        counts.append(min(c_per_offset) if use_min else np.mean(c_per_offset))

    return np.array(counts, dtype=float)
