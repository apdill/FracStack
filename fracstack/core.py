import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm # type: ignore
import matplotlib.pyplot as plt
import skimage.io as io # type: ignore

from .boxcount import boxcount, compute_dimension, dynamic_boxcount, get_pairwise_slopes
from .image_processing import process_image_to_array, pad_image_for_boxcounting, find_largest_smallest_objects
from .visualization import plot_scaling_results, show_image_info, plot_object_outlines

def measure_dimension(input_array, 
                      mode = 'D0', 
                      num_sizes=50, 
                      min_size=16, 
                      max_size=506, 
                      num_offsets=50, 
                      pad_factor=1.5, 
                      use_optimization=True, 
                      sparse_threshold=0.01, 
                      use_min_count=False, 
                      seed=None, 
                      use_weighted_fit=True,
                      use_bootstrap_ci=False,
                      bootstrap_method='residual',
                      n_bootstrap=1000,
                      bootstrap_seed=None):
    """
    Measure fractal dimension of a binary image using box counting analysis.
    
    This function provides a streamlined interface for fractal dimension calculation,
    handling the complete workflow from input validation through final dimension
    estimation. It automatically ensures binary input, applies padding for edge
    effect mitigation, performs optimized box counting, and computes robust
    statistical estimates with optional confidence intervals.
    
    Parameters
    ----------
    input_array : np.ndarray
        2D binary image array to analyze. Non-binary inputs are automatically
        converted to binary using boolean casting.
    mode : str, default 'D0'
        Type of fractal dimension to compute:
        - 'D0': Capacity dimension (box counting dimension)
        - 'D1': Information dimension (entropy-based)
    num_sizes : int, default 50
        Number of box sizes to test, distributed geometrically between
        min_size and max_size
    min_size : int, default 16
        Minimum box size in pixels. Should be larger than the finest
        structural features for reliable results.
    max_size : int, default 506
        Maximum box size in pixels. Should be smaller than the image
        size to ensure sufficient sampling.
    num_offsets : int, default 50
        Number of grid offset positions to test for each box size.
        More offsets reduce bias from grid alignment effects.
    pad_factor : float, default 1.5
        Padding factor for edge effect mitigation. Array is padded to
        pad_factor * max_size. Set to None to disable padding.
    use_optimization : bool, default True
        Whether to use optimized box counting algorithms. Automatically
        selects between sparse, bounding box, and basic optimizations.
    sparse_threshold : float, default 0.01
        Sparsity threshold for sparse optimization. Arrays with sparsity
        (non-zero fraction) below this use coordinate-based processing.
    use_min_count : bool, default False
        For D0 mode only: whether to use minimum count across offsets (True)
        or average count across offsets (False, recommended).
    seed : int, optional
        Random seed for reproducible grid offset generation. If None,
        uses current random state.
    use_weighted_fit : bool, default True
        Whether to use weighted least squares instead of ordinary least
        squares. WLS addresses heteroscedasticity in log-log scaling data.
    use_bootstrap_ci : bool, default False
        Whether to compute bootstrap confidence intervals for the dimension
        estimate. Provides more robust uncertainty quantification.
    bootstrap_method : str, default 'residual'
        Bootstrap method for confidence intervals:
        - 'residual': Residual bootstrap (recommended)
        - 'standard': Standard bootstrap resampling
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility. If None, uses current
        random state.
        
    Returns
    -------
    dict
        Dictionary containing fractal dimension analysis results:
        - 'D' : float - Computed fractal dimension
        - 'valid_sizes' : np.ndarray - Box sizes used in final fit after filtering
        - 'valid_counts' : np.ndarray - Corresponding measures used in final fit
        - 'fit' : np.ndarray - Linear fit parameters [slope, intercept]
        - 'R2' : float - R-squared value indicating goodness of fit
        - 'ci_low' : float - Lower bound of confidence interval (if bootstrap enabled)
        - 'ci_high' : float - Upper bound of confidence interval (if bootstrap enabled)
        
    Notes
    -----
    Processing Pipeline:
    1. **Input Validation**: Ensures binary input through boolean casting
    2. **Padding**: Applies edge effect mitigation if pad_factor is specified
    3. **Box Counting**: Performs optimized box counting analysis
    4. **Dimension Calculation**: Computes dimension using robust statistical methods
    
    The function automatically handles:
    - Binary conversion of non-binary inputs
    - Optimal padding for edge effect reduction
    - Selection of appropriate optimization algorithms
    - Robust statistical fitting with optional bootstrap confidence intervals
    
    For D0 (capacity dimension), the fractal dimension is computed as the negative
    slope of log₁₀(N(ε)) vs log₁₀(ε), where N(ε) is the number of boxes of size ε
    containing structure.
    
    For D1 (information dimension), the dimension is computed as the negative slope
    of H(ε) vs log₂(ε), where H(ε) is the Shannon entropy of the box probability
    distribution.
    
    Quality Indicators:
    - R² > 0.99: Excellent linear scaling
    - R² > 0.95: Good linear scaling
    - R² < 0.95: Poor scaling, consider adjusting size range
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple fractal-like pattern
    >>> array = np.random.choice([0, 1], size=(256, 256), p=[0.7, 0.3])
    >>> 
    >>> # Basic dimension measurement
    >>> result = measure_dimension(array, mode='D0')
    >>> print(f"Fractal dimension: {result['D']:.3f}")
    >>> print(f"Fit quality (R²): {result['R2']:.4f}")
    >>> 
    >>> # Advanced analysis with confidence intervals
    >>> result = measure_dimension(array, mode='D0', use_bootstrap_ci=True,
    ...                          bootstrap_method='residual', n_bootstrap=1000)
    >>> print(f"D0 = {result['D']:.3f} [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
    
    See Also
    --------
    boxcount : Low-level box counting interface
    compute_dimension : Statistical dimension computation
    dynamic_boxcount : Advanced analysis with optimal range detection
    portfolio_plot : Visualization and analysis interface
    """
    
    if not np.array_equal(input_array/np.max(input_array), input_array.astype(bool)):
        input_array = input_array.astype(bool).astype(np.uint8) # makes sure input_array is binary
    
    if pad_factor is None:
        padded_array = input_array
    else:
        padded_array = pad_image_for_boxcounting(input_array, max_size, pad_factor=pad_factor)
        
    sizes, counts = boxcount(padded_array, mode=mode, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed)
    valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(sizes, counts, mode=mode, use_weighted_fit=use_weighted_fit, use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
    
    
    return {'D': d_value, 'valid_sizes': valid_sizes, 'valid_counts': valid_counts, 'fit': fit, 'R2': r2, 'ci_low': ci_low, 'ci_high': ci_high}


def portfolio_plot(input_array = None,
                min_size=16, 
                max_size=None, 
                num_sizes=100, 
                num_offsets=100,
                pad_factor=1.5,
                figsize=(21, 7),
                save_dir = None,
                f_name = None,
                target_D0 = None,
                D0_threshold = 1,
                R2_D0_threshold = 0,
                D0_D1_threshold = 1,
                show_plot = True,
                use_optimization=True,
                sparse_threshold=0.01,
                use_dynamic=False,
                dynamic_params=None,
                compute_dimensions='both',
                use_min_count=False,
                use_weighted_fit=True,
                seed=None,
                use_bootstrap_ci=False,
                bootstrap_method='residual',
                n_bootstrap=1000,
                bootstrap_seed=None,
                include_slope_analysis=False,
                vertical_lines=None,
                custom_sizes=None):
    """
    Create comprehensive visualization and analysis of fractal dimensions with advanced plotting features.
    
    This function provides a complete fractal analysis workflow, combining dimension
    computation, visualization, and quality assessment in a single interface. It can
    compute both capacity (D0) and information (D1) dimensions, with optional dynamic
    range optimization, bootstrap confidence intervals, and advanced visualization
    features including slope analysis and custom vertical line annotations.
    
    Parameters
    ----------
    input_array : np.ndarray
        2D binary image array to analyze. Must contain only 0s and 1s.
    min_size : int, default 16
        Minimum box size in pixels for the analysis
    max_size : int, optional
        Maximum box size in pixels. Defaults to min(array.shape)//4 if not specified.
    num_sizes : int, default 100
        Number of box sizes to test, distributed geometrically between min and max
    num_offsets : int, default 100
        Number of grid offset positions to test for each box size
    pad_factor : float, default 1.5
        Padding factor for edge effect mitigation. Array is padded to
        pad_factor * max_size to reduce boundary artifacts.
    figsize : tuple, default (21, 7)
        Figure size for the portfolio plot (width, height) in inches
    save_dir : str, optional
        Directory to save the generated plots. If None, plots are not saved.
    f_name : str, optional
        Filename for saving plots. Required if save_dir is provided.
    target_D0 : float, optional
        Target D0 value for quality control. If None, uses computed D0 value.
    D0_threshold : float, default 1
        Maximum allowed deviation from target_D0 for quality control
    R2_D0_threshold : float, default 0
        Minimum R² threshold for D0 fit quality control
    D0_D1_threshold : float, default 1
        Maximum allowed deviation between D0 and target for quality control
    show_plot : bool, default True
        Whether to display the generated plots
    use_optimization : bool, default True
        Whether to use optimized box counting algorithms
    sparse_threshold : float, default 0.01
        Sparsity threshold for sparse optimization
    use_dynamic : bool, default False
        Whether to use dynamic box counting for optimal scaling range detection.
        When True, automatically finds the best scaling range rather than using
        the full size range.
    dynamic_params : dict, optional
        Parameters for dynamic box counting. If None, uses sensible defaults.
        See dynamic_boxcount() for available parameters.
    compute_dimensions : str, default 'both'
        Which dimensions to compute and display:
        - 'D0': Only capacity dimension
        - 'D1': Only information dimension  
        - 'both': Both dimensions (default)
    use_min_count : bool, default False
        For D0 mode: whether to use minimum count across offsets (True) or
        average count across offsets (False, recommended)
    use_weighted_fit : bool, default True
        Whether to use weighted least squares instead of ordinary least squares
    seed : int, optional
        Random seed for reproducible grid offset generation
    use_bootstrap_ci : bool, default False
        Whether to compute bootstrap confidence intervals for dimension estimates
    bootstrap_method : str, default 'residual'
        Bootstrap method for confidence intervals:
        - 'residual': Residual bootstrap (recommended)
        - 'standard': Standard bootstrap resampling
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility
    include_slope_analysis : bool, default False
        Whether to generate additional slope analysis plots showing pairwise
        slopes and second-order derivatives for detailed scaling behavior analysis
    vertical_lines : array-like, optional
        Array of box sizes where dashed vertical lines should be drawn on the
        scaling plots. Useful for marking specific scales of interest or
        theoretical predictions.
    custom_sizes : array-like, optional
        Array of custom box sizes to use instead of automatic geometric generation.
        When provided, disables dynamic mode and uses these exact sizes.
        
    Returns
    -------
    dict or None
        Dictionary containing comprehensive analysis results, or None if quality
        control thresholds are not met. The dictionary includes:
        
        Core Analysis Results:
        - 'D0' : float - Capacity dimension (if computed)
        - 'D0_R2' : float - R² value for D0 fit (if computed)
        - 'D1' : float - Information dimension (if computed)
        - 'D1_R2' : float - R² value for D1 fit (if computed)
        
        Confidence Intervals (if bootstrap enabled):
        - 'D0_ci_low', 'D0_ci_high' : float - D0 confidence bounds
        - 'D1_ci_low', 'D1_ci_high' : float - D1 confidence bounds
        - 'D0_confidence_interval' : tuple - D0 confidence interval
        - 'D1_confidence_interval' : tuple - D1 confidence interval
        - 'ci_type' : str - Type of confidence interval used
        
        Analysis Parameters:
        - 'min_box_size', 'max_box_size' : int - Size range used
        - 'num_sizes', 'num_offsets' : int - Sampling parameters
        - 'image_width', 'image_height' : int - Image dimensions
        - 'use_dynamic' : bool - Whether dynamic analysis was used
        - 'compute_dimensions' : str - Which dimensions were computed
        - 'custom_sizes' : bool - Whether custom sizes were used
        - 'actual_sizes' : list - List of actual box sizes used (if custom_sizes=True)
        
        Dynamic Analysis Results (if use_dynamic=True):
        - 'D0_optimal_range' : tuple - Optimal size range for D0
        - 'D0_decade_span' : float - Decade span of D0 optimal range
        - 'D0_num_candidates' : int - Number of ranges tested for D0
        - 'D1_optimal_range' : tuple - Optimal size range for D1
        - 'D1_decade_span' : float - Decade span of D1 optimal range
        - 'D1_num_candidates' : int - Number of ranges tested for D1
        
        Slope Analysis Results (if include_slope_analysis=True):
        - 'slopes' : np.ndarray - Pairwise slopes between consecutive points
        - 'second_order_slopes' : np.ndarray - Second-order slope derivatives
        
    Notes
    -----
    Enhanced Features:
    
    This updated version includes several advanced features:
    
    1. **Custom Size Support**: Allows specification of exact box sizes rather than
       automatic geometric generation
    2. **Vertical Line Annotations**: Adds dashed vertical lines at specified scales
       for marking important features or theoretical predictions
    3. **Slope Analysis**: Optional detailed analysis of pairwise slopes and
       second-order derivatives for deeper scaling behavior insights
    4. **Improved Weighted Fitting**: Enhanced support for weighted least squares
       with proper weight calculation for both D0 and D1 modes
    
    Quality Control System:
    
    The function implements a sophisticated quality control system that filters
    out analyses that don't meet specified criteria:
    
    1. **D0 Deviation Check**: |D0 - target_D0| ≤ D0_threshold
    2. **R² Quality Check**: R² ≥ R2_D0_threshold  
    3. **Target Consistency Check**: |D0 - target_D0| ≤ D0_D1_threshold
    
    If any check fails, the function prints a warning and returns None.
    
    Visualization Modes:
    
    The function creates different plot layouts based on compute_dimensions:
    - 'D0': 2-panel layout (image + D0 scaling plot)
    - 'D1': 2-panel layout (image + D1 scaling plot)
    - 'both': 3-panel layout (image + D0 scaling + D1 scaling)
    
    Additional slope analysis plots are generated if include_slope_analysis=True.
    
    Dynamic vs Standard Analysis:
    
    - **Standard Mode**: Uses the full specified size range for analysis
    - **Dynamic Mode**: Automatically finds the optimal scaling range with
      highest R² or largest range meeting quality criteria
    - **Custom Size Mode**: Uses exact user-specified sizes (disables dynamic mode)
    
    Dynamic mode is recommended for research applications where optimal
    scaling range identification is critical.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a fractal-like binary pattern
    >>> array = np.random.choice([0, 1], size=(512, 512), p=[0.7, 0.3])
    >>> 
    >>> # Basic portfolio analysis
    >>> result = portfolio_plot(array, compute_dimensions='both', show_plot=True)
    >>> if result is not None:
    ...     print(f"D0 = {result['D0']:.3f}, D1 = {result['D1']:.3f}")
    >>> 
    >>> # Advanced analysis with custom sizes and vertical lines
    >>> custom_sizes = [4, 8, 16, 32, 64, 128]
    >>> vertical_lines = [16, 64]  # Mark specific scales
    >>> result = portfolio_plot(array, custom_sizes=custom_sizes, 
    ...                        vertical_lines=vertical_lines,
    ...                        include_slope_analysis=True)
    >>> 
    >>> # Dynamic analysis with slope analysis
    >>> result = portfolio_plot(array, use_dynamic=True, 
    ...                        include_slope_analysis=True,
    ...                        vertical_lines=[8, 32, 128])
    >>> if result is not None:
    ...     print(f"Slopes: {result['slopes'][:5]}...")  # First 5 slopes
    >>> 
    >>> # Quality-controlled batch processing
    >>> result = portfolio_plot(array, target_D0=1.8, D0_threshold=0.1,
    ...                        R2_D0_threshold=0.99, show_plot=False)
    >>> if result is None:
    ...     print("Analysis failed quality control")
    
    See Also
    --------
    measure_dimension : Core dimension measurement function
    dynamic_boxcount : Advanced optimal range detection
    compute_dimension : Statistical dimension computation with bootstrap
    plot_pairwise_slopes : Detailed slope analysis visualization
    get_pairwise_slopes : Slope calculation function
    """
    
    assert np.array_equal(input_array/np.max(input_array), input_array.astype(bool)), "Input array must be binary (contain only 0s and 1s)"

    if save_dir is not None:
        assert f_name is not None, "f_name must be provided if save_dir is provided"
        save_path = os.path.join(save_dir, f_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Handle custom sizes - disable dynamic mode if custom sizes provided
    if custom_sizes is not None:
        custom_sizes = np.array(custom_sizes, dtype=int)
        use_dynamic = False  # Force standard mode with custom sizes
        # Set max_size to largest custom size for padding
        max_size = np.max(custom_sizes)
        print(f"Using custom sizes: {len(custom_sizes)} sizes from {custom_sizes.min()} to {custom_sizes.max()}")
    else:
        # Set default max_size if None before padding
        if max_size is None:
            max_size = min(input_array.shape) // 4
    
    padded_input_array = pad_image_for_boxcounting(input_array, max_size, pad_factor=pad_factor)

    # Validate compute_dimensions parameter
    if compute_dimensions not in ['D0', 'D1', 'both']:
        raise ValueError("compute_dimensions must be 'D0', 'D1', or 'both'")
    
    # Initialize variables
    d_value_d0 = d_value_d1 = None
    r2_d0 = r2_d1 = None
    valid_sizes_d0 = valid_sizes_d1 = None
    valid_counts = valid_entropies = None
    fit_d0 = fit_d1 = None
    global_sizes_d0 = global_sizes_d1 = None
    global_counts = global_entropies = None
    
    if use_dynamic:
        # Import dynamic_boxcount for optimal range finding
        from .boxcount import dynamic_boxcount
        
        # Set up default dynamic parameters if not provided
        if dynamic_params is None:
            dynamic_params = {
                'global_min_size': min_size,
                'global_max_size': max_size,
                'min_decade_span': 1.0,
                'min_points': 10,
                'num_sizes': num_sizes,
                'num_offsets': num_offsets,
                'show_progress': False,
                'stretch': False,
                'min_R2': 0.9985,
                'use_plateau_detection': False,
                'plateau_window': 4,
                'plateau_tol': 0.01,
                'plateau_method': 'pairwise_first',
                'pairwise_tol': 0.01,
                'use_occupancy_filter': False,
                'occ_low': 0.05,
                'occ_high': 0.95,
                'verbose': False,
                'plateau_fail_early': False
            }
        
        # Dynamic D0 analysis (if requested)
        if compute_dimensions in ['D0', 'both']:
            dynamic_result_d0 = dynamic_boxcount(padded_input_array, mode='D0', use_min_count=use_min_count, seed=seed, 
                                                use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, 
                                                n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed, **dynamic_params)
            
            ci_type = dynamic_result_d0['ci_type']
            ci_low = dynamic_result_d0['ci_low']
            ci_high = dynamic_result_d0['ci_high']
            
            # Extract optimal D0 data
            valid_sizes_d0 = dynamic_result_d0['optimal_sizes']
            valid_counts = dynamic_result_d0['optimal_measures']
            d_value_d0 = dynamic_result_d0['D_value']
            r2_d0 = dynamic_result_d0['R2']
            
            # Get global D0 data for plotting context
            global_sizes_d0 = dynamic_result_d0['global_sizes']
            global_counts = dynamic_result_d0['global_measures']
            
            # Calculate linear fits for plotting
            if use_weighted_fit:
                fit_d0 = np.polyfit(np.log10(valid_sizes_d0), np.log10(valid_counts), 1, w=np.sqrt(valid_counts))
            else:
                fit_d0 = np.polyfit(np.log10(valid_sizes_d0), np.log10(valid_counts), 1)
        
        # Dynamic D1 analysis (if requested)
        if compute_dimensions in ['D1', 'both']:
            dynamic_result_d1 = dynamic_boxcount(padded_input_array, mode='D1', seed=seed, 
                                                use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, 
                                                n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed, **dynamic_params)
            
            ci_type = dynamic_result_d1['ci_type']
            ci_low = dynamic_result_d1['ci_low']
            ci_high = dynamic_result_d1['ci_high']
            
            # Extract optimal D1 data
            valid_sizes_d1 = dynamic_result_d1['optimal_sizes']
            valid_entropies = dynamic_result_d1['optimal_measures']
            d_value_d1 = dynamic_result_d1['D_value']
            r2_d1 = dynamic_result_d1['R2']
            
            # Get global D1 data for plotting context
            global_sizes_d1 = dynamic_result_d1['global_sizes']
            global_entropies = dynamic_result_d1['global_measures']
            
            # Calculate linear fits for plotting
            if use_weighted_fit:
                fit_d1 = np.polyfit(np.log2(valid_sizes_d1), valid_entropies, 1, w=np.sqrt(valid_entropies))
            else:
                fit_d1 = np.polyfit(np.log2(valid_sizes_d1), valid_entropies, 1)
        
    else:
        # Standard box counting approach
        if custom_sizes is not None:
            # Direct box counting with custom sizes using numba functions
            from .boxcount import generate_random_offsets, numba_d0_optimized, numba_d1_optimized, numba_d0, numba_d1, numba_d0_sparse
            
            # Prepare array for numba functions
            array_numba = np.ascontiguousarray(padded_input_array.astype(np.float32))
            
            # Pre-generate random offsets for custom sizes
            offsets = generate_random_offsets(custom_sizes, num_offsets, seed=seed)
            
            if compute_dimensions in ['D0', 'both']:
                if use_optimization:
                    # Calculate sparsity to choose optimization strategy
                    total_pixels = array_numba.size
                    non_zero_pixels = np.count_nonzero(array_numba)
                    sparsity = non_zero_pixels / total_pixels
                    
                    # Use sparse optimization for very sparse arrays (D0 only)
                    if sparsity <= sparse_threshold:
                        counts = numba_d0_sparse(array_numba, custom_sizes, offsets, sparse_threshold, use_min_count)
                    else:
                        counts = numba_d0_optimized(array_numba, custom_sizes, offsets, use_min_count)
                else:
                    counts = numba_d0(array_numba, custom_sizes, offsets, use_min_count)
                
                sizes = custom_sizes
                valid_sizes_d0, valid_counts, d_value_d0, fit_d0, r2_d0, ci_low_d0, ci_high_d0 = compute_dimension(sizes, counts, mode='D0', use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
                ci_type = bootstrap_method
                ci_low = ci_low_d0
                ci_high = ci_high_d0
                
            if compute_dimensions in ['D1', 'both']:
                if use_optimization:
                    counts = numba_d1_optimized(array_numba, custom_sizes, offsets)
                else:
                    counts = numba_d1(array_numba, custom_sizes, offsets)
                
                sizes = custom_sizes
                valid_sizes_d1, valid_entropies, d_value_d1, fit_d1, r2_d1, ci_low_d1, ci_high_d1 = compute_dimension(sizes, counts, mode='D1', use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
                ci_type = bootstrap_method
                # Only update ci_low/ci_high if D1 is the only dimension being computed
                if compute_dimensions == 'D1':
                    ci_low = ci_low_d1
                    ci_high = ci_high_d1
        else:
            # Standard box counting approach with automatic size generation
            if compute_dimensions in ['D0', 'both']:
                sizes, counts = boxcount(padded_input_array, mode='D0', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed)
                valid_sizes_d0, valid_counts, d_value_d0, fit_d0, r2_d0, ci_low_d0, ci_high_d0 = compute_dimension(sizes, counts, mode='D0', use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
                ci_type = bootstrap_method
                ci_low = ci_low_d0
                ci_high = ci_high_d0
                
            if compute_dimensions in ['D1', 'both']:
                sizes, counts = boxcount(padded_input_array, mode='D1', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed)
                valid_sizes_d1, valid_entropies, d_value_d1, fit_d1, r2_d1, ci_low_d1, ci_high_d1 = compute_dimension(sizes, counts, mode='D1', use_bootstrap_ci=use_bootstrap_ci, bootstrap_method=bootstrap_method, n_bootstrap=n_bootstrap, random_seed=bootstrap_seed)
                ci_type = bootstrap_method
                ci_low = ci_low_d1
                ci_high = ci_high_d1
            
    # Quality checks (only if D0 was computed)
    if d_value_d0 is not None:
        if target_D0 is None:
            target_D0 = d_value_d0
        
        D0_D1_check = np.abs(d_value_d0 - target_D0) > D0_D1_threshold
        R2_check = r2_d0 < R2_D0_threshold  
        D0_check = np.abs(d_value_d0 - target_D0) > D0_threshold

        if D0_D1_check or R2_check or D0_check:
            d1_str = f", D1 = {d_value_d1:.3f}" if d_value_d1 is not None else ""
            print(f"Skipping, D0 = {d_value_d0:.3f}{d1_str}, r2 = {r2_d0:.6f}")
            return None
    
    # Determine number of subplots based on computed dimensions
    if compute_dimensions == 'D0':
        num_plots = 2
        figsize_adjusted = (figsize[0] * 2/3, figsize[1])  # Adjust figure width
    elif compute_dimensions == 'D1':
        num_plots = 2
        figsize_adjusted = (figsize[0] * 2/3, figsize[1])  # Adjust figure width
    else:  # 'both'
        num_plots = 3
        figsize_adjusted = figsize
    
    # Always create the figure (whether we show it or not)
    fig, axes = plt.subplots(1, num_plots, figsize=figsize_adjusted)
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always a list
    
    # First subplot is always the image
    axes[0].imshow(input_array, cmap='gray')
    axes[0].set_title('Fractal Binary Image', fontsize=22)
    axes[0].axis('off')
    
    plot_idx = 1  # Start from second subplot
    
    # D0 plot with dynamic range highlighting (if computed)
    if d_value_d0 is not None:
        ax = axes[plot_idx]
        
        if use_dynamic:
            # Plot global data lightly
            ax.scatter(np.log10(global_sizes_d0), np.log10(global_counts), 
                       color='lightgray', alpha=0.5, s=20, label='Global data')
            # Plot optimal range prominently
            ax.scatter(np.log10(valid_sizes_d0), np.log10(valid_counts), 
                       color='black', s=40, label='Optimal range')
            ax.plot(np.log10(valid_sizes_d0), fit_d0[0] * np.log10(valid_sizes_d0) + fit_d0[1], 
                    color='red', linewidth=2)
            ax.legend(fontsize=12)
            title_suffix = f" (Dynamic: {dynamic_result_d0['optimal_range'][0]:.0f}-{dynamic_result_d0['optimal_range'][1]:.0f})"
        else:
            ax.scatter(np.log10(valid_sizes_d0), np.log10(valid_counts), color='black')
            ax.plot(np.log10(valid_sizes_d0), fit_d0[0] * np.log10(valid_sizes_d0) + fit_d0[1], color='red')
            title_suffix = ""
            
        ax.text(0.7, 0.95, fr'$D_0$ = {d_value_d0:.3f}' '\n' fr'$R^2$ = {r2_d0:.5f}', 
                transform=ax.transAxes, fontsize=18, verticalalignment='top', 
                bbox=dict(boxstyle="round", alpha=0.3))
        
        # Store current axis limits before adding vertical lines
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        
        # Add vertical lines if specified
        if vertical_lines is not None and len(vertical_lines) > 0:
            for vline in vertical_lines:
                ax.axvline(x=np.log10(vline), color='red', linestyle='--', alpha=0.8, linewidth=2)
            # Restore original axis limits to prevent vertical lines from changing them
            ax.set_xlim(xlim_before)
            ax.set_ylim(ylim_before)
        else:
            # Only show grid if no vertical lines are present
            ax.grid(True)
        
        ax.set_title(f'Fractal Dimension{title_suffix}', fontsize=22)
        ax.set_xlabel(r'Log($\epsilon$)', fontsize=18)
        ax.set_ylabel(r'Log($N(\epsilon)$)', fontsize=18)
        
        plot_idx += 1

    # D1 plot with dynamic range highlighting (if computed)
    if d_value_d1 is not None:
        ax = axes[plot_idx]
        
        if use_dynamic:
            # Plot global data lightly
            ax.scatter(np.log2(global_sizes_d1), global_entropies, 
                       color='lightgray', alpha=0.5, s=20, label='Global data')
            # Plot optimal range prominently
            ax.scatter(np.log2(valid_sizes_d1), valid_entropies, 
                       color='black', s=40, label='Optimal range')
            ax.plot(np.log2(valid_sizes_d1), fit_d1[0] * np.log2(valid_sizes_d1) + fit_d1[1], 
                    color='red', linewidth=2)
            ax.legend(fontsize=12)
            title_suffix = f" (Dynamic: {dynamic_result_d1['optimal_range'][0]:.0f}-{dynamic_result_d1['optimal_range'][1]:.0f})"
        else:
            ax.scatter(np.log2(valid_sizes_d1), valid_entropies, color='black')
            ax.plot(np.log2(valid_sizes_d1), fit_d1[0] * np.log2(valid_sizes_d1) + fit_d1[1], color='red')
            title_suffix = ""
            
        ax.text(0.7, 0.95, fr'$D_1$ = {d_value_d1:.3f}' '\n' fr'$R^2$ = {r2_d1:.5f}', 
                transform=ax.transAxes, fontsize=18, verticalalignment='top', 
                bbox=dict(boxstyle="round", alpha=0.3))
        
        # Store current axis limits before adding vertical lines
        xlim_before = ax.get_xlim()
        ylim_before = ax.get_ylim()
        
        # Add vertical lines if specified
        if vertical_lines is not None and len(vertical_lines) > 0:
            for vline in vertical_lines:
                ax.axvline(x=np.log2(vline), color='red', linestyle='--', alpha=0.8, linewidth=2)
            # Restore original axis limits to prevent vertical lines from changing them
            ax.set_xlim(xlim_before)
            ax.set_ylim(ylim_before)
        else:
            # Only show grid if no vertical lines are present
            ax.grid(True)
        
        ax.set_title(f'Information Dimension{title_suffix}', fontsize=22)
        ax.set_xlabel(r'Log$_2$($\epsilon$)', fontsize=18)
        ax.set_ylabel(r'$H(\epsilon)$', fontsize=18)

    plt.tight_layout()
    
    # Only show the plot if requested
    if show_plot:
        plt.show()
    
    # Save the figure if save directory is provided
    if save_dir is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)

    # Prepare return dictionary (only include computed dimensions)
    image_info_dict = {
        'min_box_size': min_size if custom_sizes is None else custom_sizes.min(),
        'max_box_size': max_size if custom_sizes is None else custom_sizes.max(),
        'num_sizes': num_sizes if custom_sizes is None else len(custom_sizes),
        'num_offsets': num_offsets,
        'image_width': input_array.shape[1],
        'image_height': input_array.shape[0],
        'use_dynamic': use_dynamic,
        'compute_dimensions': compute_dimensions,
        'ci_type': ci_type,
        'custom_sizes': custom_sizes is not None,
        'actual_sizes': custom_sizes.tolist() if custom_sizes is not None else None
    }
    
    # Add D0 metrics if computed
    if d_value_d0 is not None:
        image_info_dict.update({
            'D0': d_value_d0,
            'D0_R2': r2_d0
        })
        
        # Add D0 confidence intervals if not using dynamic mode

        image_info_dict.update({
            'D0_ci_low': ci_low,
            'D0_ci_high': ci_high,
            'D0_confidence_interval': (ci_low, ci_high)
        })
        
        # Add D0 dynamic-specific information if applicable
        if use_dynamic:
            image_info_dict.update({
                'D0_optimal_range': dynamic_result_d0['optimal_range'],
                'D0_decade_span': dynamic_result_d0['decade_span'],
                'D0_confidence_interval': dynamic_result_d0['confidence_interval'],
                'D0_num_candidates': dynamic_result_d0['num_candidates_tested']
            })
    
    # Add D1 metrics if computed
    if d_value_d1 is not None:
        image_info_dict.update({
            'D1': d_value_d1,
            'D1_R2': r2_d1
        })
        
        image_info_dict.update({
            'D1_ci_low': ci_low,
            'D1_ci_high': ci_high,
            'D1_confidence_interval': (ci_low, ci_high)
        })
        
        # Add D1 dynamic-specific information if applicable
        if use_dynamic:
            image_info_dict.update({
                'D1_optimal_range': dynamic_result_d1['optimal_range'],
                'D1_decade_span': dynamic_result_d1['decade_span'],
                'D1_confidence_interval': dynamic_result_d1['confidence_interval'],
                'D1_num_candidates': dynamic_result_d1['num_candidates_tested']
            })


    if include_slope_analysis:
        if vertical_lines is not None:
            slopes, second_order_slopes = plot_pairwise_slopes(input_array, mode=r'$D_0$', plot_second_order=True, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed, figsize=figsize, vertical_lines=vertical_lines, save_path=save_path)
        else:
            slopes, second_order_slopes = plot_pairwise_slopes(input_array, mode=r'$D_0$', plot_second_order=True, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed, figsize=figsize, save_path=save_path)
        
        image_info_dict.update({
            'slopes': slopes,
            'second_order_slopes': second_order_slopes
        })

    return image_info_dict


def plot_pairwise_slopes(input_array, 
                         mode='D0', 
                         plot_second_order=False, 
                         min_size=16, 
                         max_size=506, 
                         num_sizes=100, 
                         num_offsets=10, 
                         use_optimization=True, 
                         sparse_threshold=0.01,
                         variation_band_threshold = 0.05,
                         use_min_count=False, 
                         seed=None, 
                         figsize=(11, 11),
                         vertical_lines=None,
                         save_path=None):
    """
    Create detailed visualization of pairwise slopes and second-order derivatives for scaling analysis.
    
    This function generates comprehensive plots showing the local scaling behavior of
    fractal systems by visualizing pairwise slopes between consecutive points in
    log-log scaling data. It can also plot second-order derivatives to reveal
    changes in scaling behavior and identify transitions between different regimes.
    
    Parameters
    ----------
    input_array : np.ndarray
        2D binary image array to analyze
    mode : str, default 'D0'
        Type of fractal dimension analysis:
        - 'D0': Capacity dimension (box counting)
        - 'D1': Information dimension (entropy-based)
    plot_second_order : bool, default False
        Whether to generate additional plots showing second-order slope derivatives
    min_size : int, default 16
        Minimum box size in pixels for box counting analysis
    max_size : int, default 506
        Maximum box size in pixels for box counting analysis
    num_sizes : int, default 100
        Number of box sizes to test, distributed geometrically
    num_offsets : int, default 10
        Number of grid offset positions to test for each box size
    use_optimization : bool, default True
        Whether to use optimized box counting algorithms
    sparse_threshold : float, default 0.01
        Sparsity threshold for sparse optimization
    variation_band_threshold : float, default 0.05
        Threshold for variation bands around median slopes, expressed as
        fraction of median slope magnitude
    use_min_count : bool, default False
        For D0 mode: whether to use minimum count across offsets (True) or
        average count across offsets (False, recommended)
    seed : int, optional
        Random seed for reproducible grid offset generation
    figsize : tuple, default (11, 11)
        Figure size for each plot (width, height) in inches
    vertical_lines : array-like, optional
        Array of box sizes where dashed vertical lines should be drawn
        to mark specific scales of interest
    save_path : str, optional
        Path to save the plots. If None, plots are displayed but not saved.
        
    Returns
    -------
    np.ndarray or tuple
        If plot_second_order=False:
            slopes : np.ndarray - Pairwise slopes between consecutive points
        If plot_second_order=True:
            (slopes, second_order_slopes) : tuple - Both first and second-order slopes
            
    Notes
    -----
    Visualization Features:
    
    The function creates detailed plots with the following features:
    
    1. **Pairwise Slopes Plot**:
       - Black line with markers showing local slopes
       - Green dashed line indicating median slope
       - Red dashed lines showing variation bands (±variation_band_threshold)
       - Optional vertical lines marking specific scales
       - Grid for easy reading
    
    2. **Second-Order Slopes Plot** (if plot_second_order=True):
       - Red line with square markers showing slope derivatives
       - Similar median and variation band indicators
       - Reveals acceleration/deceleration in scaling behavior
    
    Mathematical Background:
    
    Pairwise slopes are computed as:
    slope[i] = (log(N[i+1]) - log(N[i])) / (log(ε[i+1]) - log(ε[i]))
    
    Second-order slopes (derivatives) are:
    d²slope[i] = (slope[i+1] - slope[i]) / (log(ε[i+2]) - log(ε[i]))
    
    Applications:
    
    - **Scaling Quality Assessment**: Consistent slopes indicate good fractal behavior
    - **Plateau Detection**: Regions of stable slopes are optimal for dimension estimation
    - **Transition Identification**: Abrupt slope changes reveal scaling transitions
    - **Crossover Analysis**: Second-order derivatives help identify crossover points
    - **Method Validation**: Visual confirmation of automatic plateau detection
    
    Interpretation Guidelines:
    
    - **Flat, consistent slopes**: Good fractal scaling
    - **Scattered, variable slopes**: Poor scaling or insufficient data
    - **Systematic trends**: Possible finite-size effects or crossovers
    - **Abrupt changes**: Transitions between scaling regimes
    - **Second-order near zero**: Consistent scaling behavior
    - **Second-order fluctuations**: Variable scaling or noise
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create a binary fractal-like image
    >>> array = np.random.choice([0, 1], size=(512, 512), p=[0.8, 0.2])
    >>> 
    >>> # Basic slope analysis
    >>> slopes = plot_pairwise_slopes(array, mode='D0', min_size=8, max_size=128)
    >>> print(f"Median slope: {np.median(slopes):.3f}")
    >>> 
    >>> # Detailed analysis with second-order derivatives
    >>> slopes, second_order = plot_pairwise_slopes(array, mode='D0', 
    ...                                            plot_second_order=True,
    ...                                            vertical_lines=[16, 32, 64])
    >>> 
    >>> # Save plots for publication
    >>> slopes = plot_pairwise_slopes(array, mode='D0', 
    ...                              figsize=(8, 6), 
    ...                              save_path='./slope_analysis.png')
    
    See Also
    --------
    get_pairwise_slopes : Function that computes the slopes
    detect_scaling_plateau : Automatic plateau detection using slopes
    detect_plateau_pairwise : Plateau detection using slope differences
    portfolio_plot : Main analysis function that can include slope analysis
    """
    
    sizes, measures = boxcount(input_array, mode=mode, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_optimization=use_optimization, sparse_threshold=sparse_threshold, use_min_count=use_min_count, seed=seed)
    
    result = get_pairwise_slopes(sizes, measures, return_second_order=plot_second_order)
    
    # Calculate log sizes for x-axis
    log_sizes = np.log10(sizes)
    
    if plot_second_order:
        # Unpack the tuple when second order is requested
        slopes, second_order_slopes = result
        variation_band_vals = variation_band_threshold*np.median(slopes)
        
        
        # X-axis for first order slopes: midpoints between consecutive log sizes
        x_first_order = (log_sizes[:-1] + log_sizes[1:]) / 2
        
        # X-axis for second order slopes: midpoints of the three consecutive sizes involved
        # second_order_slopes[i] involves sizes[i], sizes[i+1], sizes[i+2]
        x_second_order = (log_sizes[:-2] + log_sizes[2:]) / 2
        
        # Plot first order slopes
        plt.figure(figsize=figsize)
        plt.plot(x_first_order, slopes, color='black', marker='o', markersize=3)
        plt.title('Pairwise Slopes (First Order)')
        plt.xlabel('Log₁₀(Box Size)')
        plt.ylabel('Slope')
        plt.grid(True, alpha=0.3)
        
        plt.axhline(y=np.median(slopes), color='green', linestyle='--', alpha=0.8, linewidth=2)
        plt.axhline(y=np.median(slopes) + variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=np.median(slopes) - variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        if vertical_lines is not None:
            for vline in vertical_lines:
                plt.axvline(x=np.log10(vline), color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Plot second order slopes
        plt.figure(figsize=figsize)
        plt.plot(x_second_order, second_order_slopes, color='red', marker='s', markersize=3)
        plt.title('Second Order Slopes')
        plt.xlabel('Log₁₀(Box Size)')
        plt.ylabel('Second Order Slope')
        plt.grid(True, alpha=0.3)
        
        plt.axhline(y=np.median(second_order_slopes), color='green', linestyle='--', alpha=0.8, linewidth=2)
        plt.axhline(y=np.median(second_order_slopes) + variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=np.median(second_order_slopes) - variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        if vertical_lines is not None:
            for vline in vertical_lines:
                plt.axvline(x=np.log10(vline), color='red', linestyle='--', alpha=0.8, linewidth=2)
                
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return slopes, second_order_slopes
    else:
        # result is just the slopes array
        slopes = result
        variation_band_vals = variation_band_threshold*np.median(slopes)
        # X-axis for first order slopes: midpoints between consecutive log sizes
        x_first_order = (log_sizes[:-1] + log_sizes[1:]) / 2
        
        plt.figure(figsize=figsize)
        plt.plot(x_first_order, slopes, color='black', marker='o', markersize=3)
        plt.title('Pairwise Slopes')
        plt.xlabel('Log₁₀(Box Size)')
        plt.ylabel('Slope')
        plt.grid(True, alpha=0.3)

        plt.axhline(y=np.median(slopes), color='green', linestyle='--', alpha=0.8, linewidth=2)
        plt.axhline(y=np.median(slopes) + variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=np.median(slopes) - variation_band_vals, color='red', linestyle='--', alpha=0.5, linewidth=2)
        
        if vertical_lines is not None:
            for vline in vertical_lines:
                plt.axvline(x=np.log10(vline), color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return slopes


def analyze_image(input_array = None,
                image_path = None,
                save_path = None,
                save_output=False,
                mode='D0', 
                invert=False, 
                threshold=150,
                pad_factor=None,
                min_size=16, 
                max_size=None, 
                num_sizes=100, 
                num_offsets=10,
                show_image=True,
                plot_objects=False,
                image_info=True,
                criteria=None,
                use_min_count=False,
                seed=None,
                use_weighted_fit=True,
                use_bootstrap_ci=False,
                bootstrap_method='residual',
                n_bootstrap=1000,
                bootstrap_seed=None):
    """
    Analyze a single image to calculate fractal dimension with comprehensive reporting and visualization.
    
    This function provides a complete single-image analysis workflow including image loading,
    preprocessing, fractal dimension calculation, visualization, and results export. It is
    designed for interactive analysis and detailed reporting of individual images.
    
    **Note**: This function contains legacy code patterns and may have redundant visualizations.
    For production use, consider using `measure_dimension()` or `portfolio_plot()` instead.
    
    Parameters
    ----------
    input_array : np.ndarray, optional
        2D binary image array to analyze. If None, image_path must be provided.
    image_path : str, optional
        Path to image file to load and analyze. If None, input_array must be provided.
    save_path : str, optional
        Custom path for saving results. If None and save_output=True, uses image_path directory.
    save_output : bool, default False
        Whether to save analysis results, plots, and processed images to files
    mode : str, default 'D0'
        Type of fractal dimension to compute. Currently only 'D0' (capacity dimension) is supported.
    invert : bool, default False
        Whether to invert the binary image after thresholding
    threshold : int, default 150
        Threshold value for image binarization (0-255 range)
    pad_factor : float, optional
        Padding factor for edge effect mitigation. If None, no padding is applied.
    min_size : int, default 16
        Minimum box size in pixels for box counting analysis
    max_size : int, optional
        Maximum box size in pixels. Defaults to min_size * 10 if not specified.
    num_sizes : int, default 100
        Number of box sizes to test, distributed geometrically between min and max
    num_offsets : int, default 10
        Number of grid offset positions to test for each box size
    show_image : bool, default True
        Whether to display plots and visualizations during analysis
    plot_objects : bool, default False
        Whether to analyze and plot object outlines for size characterization
    image_info : bool, default True
        Whether to display detailed information about the analysis results
    criteria : str, optional
        Additional criteria string appended to save paths for organization
    use_min_count : bool, default False
        For D0 mode: whether to use minimum count across offsets (True) or
        average count across offsets (False, recommended)
    seed : int, optional
        Random seed for reproducible grid offset generation
    use_weighted_fit : bool, default True
        Whether to use weighted least squares instead of ordinary least squares
    use_bootstrap_ci : bool, default False
        Whether to compute bootstrap confidence intervals for dimension estimates
    bootstrap_method : str, default 'residual'
        Bootstrap method for confidence intervals ('residual' or 'standard')
    n_bootstrap : int, default 1000
        Number of bootstrap resamples for confidence interval estimation
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing comprehensive analysis results with columns:
        - 'filename' : str - Image filename or 'input_array'
        - 'd_value' : float - Computed fractal dimension
        - 'r2' : float - R-squared value of the linear fit
        - 'pattern_width' : int - Image width in pixels
        - 'pattern_height' : int - Image height in pixels
        - 'smallest_diameter' : float - Diameter of smallest object (if plot_objects=True)
        - 'largest_diameter' : float - Diameter of largest object (if plot_objects=True)
        - 'min_box_size' : int - Minimum box size used
        - 'max_box_size' : int - Maximum box size used
        - 'threshold' : int - Threshold value used for binarization
        - 'num_sizes' : int - Number of box sizes tested
        - 'num_offsets' : int - Number of grid positions tested
        
    Notes
    -----
    Legacy Function Characteristics:
    
    This function exhibits several legacy code patterns:
    - Multiple redundant image displays (shows image twice)
    - Mixed parameter validation approaches
    - Hardcoded visualization settings
    - Limited error handling
    - Only supports D0 dimension calculation
    
    The function performs the following workflow:
    1. **Input Validation**: Handles image loading or array input
    2. **Preprocessing**: Applies thresholding and optional inversion
    3. **Object Analysis**: Optionally analyzes object sizes and plots outlines
    4. **Dimension Calculation**: Performs box counting analysis
    5. **Visualization**: Creates scaling plots and displays results
    6. **Export**: Saves results, plots, and processed images if requested
    
    File Outputs (if save_output=True):
    - `image_properties.txt`: Text file with analysis summary
    - `{filename}_thresholded.tif`: Processed binary image
    - Various plot files generated by visualization functions
    
    Limitations:
    - Only supports capacity dimension (D0)
    - No dynamic range optimization
    - Limited bootstrap support
    - Redundant visualizations
    - Legacy parameter handling
    
    Examples
    --------
    >>> # Analyze image from file
    >>> result_df = analyze_image(image_path='fractal_image.png', 
    ...                          threshold=128, min_size=8, max_size=64)
    >>> print(f"Fractal dimension: {result_df['d_value'].iloc[0]:.3f}")
    >>> 
    >>> # Analyze array with object detection
    >>> import numpy as np
    >>> array = np.random.choice([0, 1], size=(256, 256), p=[0.8, 0.2])
    >>> result_df = analyze_image(input_array=array, plot_objects=True,
    ...                          save_output=True, save_path='./analysis_results')
    >>> 
    >>> # Batch-compatible analysis
    >>> result_df = analyze_image(image_path='test.png', show_image=False, 
    ...                          image_info=False, save_output=False)
    
    See Also
    --------
    measure_dimension : Modern dimension measurement interface
    portfolio_plot : Advanced visualization and analysis
    analyze_images : Batch processing of multiple images
    """
    if image_path is not None:
        f_name = os.path.basename(image_path)
        input_array = process_image_to_array(image_path, threshold=threshold, invert=invert).astype(np.uint8)
    elif input_array is not None and save_path is not None:
        f_name = os.path.basename(save_path)
    elif input_array is not None and save_path is None:
        f_name = 'input_array'
    elif input_array is None and save_path is None and image_path is None:  
        raise ValueError("Either image_path or input_array must be provided. If input_array is provided, provide save_path as well.")
    elif input_array is not None and save_path is None and image_path is None and save_output is True:
        raise ValueError("If input_array is provided, provide save_path as well.")

    if plot_objects:
        largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image = find_largest_smallest_objects(input_array)
        plot_object_outlines(labeled_image, largest_object, smallest_object)

    if max_size is None:
        max_size = min_size * 10

    if criteria is None:
        criteria = ''
    
    if save_output and save_path is None:
        save_dir = os.path.dirname(image_path)
        save_path = os.path.join(save_dir, os.path.splitext(f_name)[0] + f'_{criteria}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif save_output and save_path is not None:
        save_path = save_path
    else:
        save_path = None

    plt.imshow(input_array, cmap='gray')
    plt.axis('off')
    plt.show()

    fit_info_dict = measure_dimension(input_array, mode=mode, 
                                      min_size=min_size, 
                                      max_size=max_size, 
                                      num_sizes=num_sizes, 
                                      num_offsets=num_offsets, 
                                      pad_factor=pad_factor, 
                                      use_min_count=use_min_count, 
                                      seed=seed, 
                                      use_weighted_fit=use_weighted_fit,
                                      use_bootstrap_ci=use_bootstrap_ci,
                                      bootstrap_method=bootstrap_method,
                                      n_bootstrap=n_bootstrap,
                                      bootstrap_seed=bootstrap_seed)
    
    plot_scaling_results(f_name, invert = invert, input_array=input_array, 
                         sizes=fit_info_dict['valid_sizes'], measures=fit_info_dict['valid_counts'], 
                         d_value=fit_info_dict['D'], fit=fit_info_dict['fit'], r2=fit_info_dict['R2'], 
                         save=save_output, save_path=save_path, show_image=show_image)
    
    if image_info:
        show_image_info(fname=f_name, invert=invert, d_value=fit_info_dict['D'], input_array=input_array, sizes=fit_info_dict['valid_sizes'], save=save_output, save_path=save_path)
    
        print('')
        print(f"D-value for {f_name}: {fit_info_dict['D']:.3f}")
        print(f"R^2 value for fit: {fit_info_dict['R2']:.6f}\n")
        print(f"Total pattern width: {input_array.shape[1]}")
        print(f"Total pattern height: {input_array.shape[0]}\n")
        if plot_objects == True:
            print(f"Smallest object diameter: {smallest_diameter:.1f}")
            print(f"Largest object diameter: {largest_diameter:.1f}\n")
        print(f"min box width: {min_size:.1f}")
        print(f"max box width: {max_size:.1f}")

    if save_output:
        txt_save_path = os.path.join(save_path, 'image_properties.txt') 
        with open(txt_save_path, mode='w') as file:
            file.write(f"D-value for {f_name}: {fit_info_dict['D']:.3f}\n")
            file.write(f"R^2 value for fit {fit_info_dict['R2']}\n\n")
            file.write(f"Total pattern width: {np.max(input_array.shape[1])}\n")
            file.write(f"Total pattern height: {np.max(input_array.shape[0])}\n\n")
            if plot_objects == True:
                file.write(f"Smallest object diameter: {smallest_diameter:.1f}\n")
                file.write(f"Largest object diameter: {largest_diameter:.1f}\n\n")
            file.write(f"min box width: {min_size:.1f}\n")
            file.write(f"max box width: {max_size:.1f}\n\n")

        # Convert boolean array to int before normalization
        input_array_int = input_array.astype(np.uint8)
        norm_array = ((input_array_int - input_array_int.min()) / np.ptp(input_array_int) * 255).astype(np.uint8)
        if norm_array.ndim == 3 and norm_array.shape[2] == 1:
            norm_array = np.squeeze(norm_array, axis=2)
        img = Image.fromarray(norm_array, mode='L')
        tiff_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_thresholded.tif")
        img.save(tiff_file, format='TIFF')

    plt.imshow(input_array, cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Create results DataFrame
    results_dict = {
        'filename': f_name,
        'd_value': fit_info_dict['D'],
        'r2': fit_info_dict['R2'],
        'pattern_width': input_array.shape[1],
        'pattern_height': input_array.shape[0],
        'smallest_diameter': smallest_diameter if plot_objects == True else np.nan,
        'largest_diameter': largest_diameter if plot_objects == True else np.nan,
        'min_box_size': min_size,
        'max_box_size': max_size,
        'threshold': threshold,
        'num_sizes': num_sizes,
        'num_offsets': num_offsets
    }

    results_df = pd.DataFrame([results_dict])

    return results_df


def analyze_images(base_path, save=False, invert=False, threshold=150, min_size=16, max_size=None, num_sizes=100, num_offsets=10, criteria = None, use_min_count=False, seed=None, use_weighted_fit=True):
    """
    Batch process multiple images in a directory to calculate fractal dimensions with comprehensive reporting.
    
    This function provides automated batch processing of multiple images for fractal dimension
    analysis. It processes all supported image formats in a directory, applies consistent
    analysis parameters, and generates both individual and summary reports.
    
    **Note**: This function contains legacy code patterns and always shows visualizations.
    For production batch processing, consider using `measure_dimension()` or `portfolio_plot()`
    in a custom loop with better control over visualization and error handling.
    
    Parameters
    ----------
    base_path : str
        Path to directory containing images to analyze. Function will process all
        supported image formats (.tiff, .tif, .jpeg, .png) in this directory.
    save : bool, default False
        Whether to save individual analysis results and visualizations to files.
        Creates subdirectories for each image if True.
    invert : bool, default False
        Whether to invert binary images after thresholding. Applied to all images.
    threshold : int, default 150
        Threshold value for image binarization (0-255 range). Applied to all images.
    min_size : int, default 16
        Minimum box size in pixels for box counting analysis
    max_size : int, optional
        Maximum box size in pixels. Defaults to min_size * 100 if not specified.
        Note: This default is different from analyze_image (min_size * 10).
    num_sizes : int, default 100
        Number of box sizes to test, distributed geometrically between min and max
    num_offsets : int, default 10
        Number of grid offset positions to test for each box size
    criteria : str, optional
        Additional criteria string appended to save paths and CSV filename for
        organization and identification of different analysis runs
    use_min_count : bool, default False
        For D0 mode: whether to use minimum count across offsets (True) or
        average count across offsets (False, recommended)
    seed : int, optional
        Random seed for reproducible grid offset generation across all images
    use_weighted_fit : bool, default True
        Whether to use weighted least squares instead of ordinary least squares
        for all dimension calculations
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing batch analysis results with columns:
        - 'Image Name' : str - Image filename without extension
        - 'D Value' : float - Computed fractal dimension
        - 'Min Box Width' : int - Minimum box size used
        - 'Max Box Width' : int - Maximum box size used  
        - 'R2 Score' : float - R-squared value of the linear fit
        
    Notes
    -----
    Legacy Function Characteristics:
    
    This function exhibits several legacy code patterns:
    - Always displays visualizations (cannot be disabled)
    - Limited error handling for individual image failures
    - Hardcoded file format support
    - Fixed analysis workflow with no customization per image
    - Only supports D0 dimension calculation
    - No progress reporting beyond basic tqdm bar
    
    Processing Workflow:
    1. **Directory Scanning**: Identifies all supported image files
    2. **Image Processing**: Loads, thresholds, and optionally inverts each image
    3. **Object Analysis**: Analyzes object sizes for each image
    4. **Dimension Calculation**: Performs box counting analysis
    5. **Visualization**: Creates and displays plots for each image
    6. **Export**: Saves individual results if requested
    7. **Summary**: Generates and displays summary DataFrame
    
    File Outputs (if save=True):
    For each image:
    - `{filename}_{criteria}/`: Analysis directory
    - `image_properties.txt`: Individual analysis summary
    - `{filename}_thresholded.tif`: Processed binary image
    - Various plot files from visualization functions
    
    Summary output:
    - `D_values_{criteria}.csv`: CSV file with all results
    
    Supported Image Formats:
    - TIFF (.tiff, .tif)
    - JPEG (.jpeg)
    - PNG (.png)
    
    Limitations:
    - Only supports capacity dimension (D0)
    - No per-image parameter customization
    - Always shows visualizations (not suitable for headless processing)
    - Limited error recovery for problematic images
    - No parallel processing support
    - Legacy parameter handling and validation
    
    Performance Considerations:
    - Processing time scales linearly with number of images
    - Memory usage depends on image sizes and visualization
    - Visualization display can be slow for large batches
    - File I/O overhead for saving individual results
    
    Examples
    --------
    >>> # Basic batch processing
    >>> results_df = analyze_images('/path/to/images', threshold=128, 
    ...                            min_size=8, max_size=64)
    >>> print(f"Processed {len(results_df)} images")
    >>> print(f"Mean dimension: {results_df['D Value'].mean():.3f}")
    >>> 
    >>> # Batch processing with results saving
    >>> results_df = analyze_images('/path/to/images', save=True, 
    ...                            criteria='experiment_1', 
    ...                            min_size=16, num_sizes=50)
    >>> 
    >>> # Analysis with specific parameters
    >>> results_df = analyze_images('/path/to/fractal_images', 
    ...                            invert=True, threshold=200,
    ...                            min_size=4, max_size=128, 
    ...                            num_offsets=20, seed=42)
    
    See Also
    --------
    analyze_image : Single image analysis with more options
    measure_dimension : Modern dimension measurement interface  
    portfolio_plot : Advanced visualization and analysis
    """
    D_value_list = []

    for f_name in tqdm(os.listdir(base_path), desc='Doing boxcounts...'):
        
        if os.path.isdir(os.path.join(base_path, f_name)):
            continue
        
        valid_extensions = ('.tiff', '.tif', '.jpeg', '.png')
        if not f_name.lower().endswith(valid_extensions):
            continue
        
        im_path = os.path.join(base_path, f_name)
        input_array = process_image_to_array(im_path, threshold=threshold, invert=invert).astype(np.uint8)
        
        largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image = find_largest_smallest_objects(input_array)

        if max_size is None:
            max_size = min_size * 100

        if criteria is None:
            criteria = ''
        
        if save:
            save_path = os.path.join(base_path, os.path.splitext(f_name)[0] + f'_{criteria}')

            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = None

        fit_info_dict = measure_dimension(input_array, mode='D0', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, use_min_count=use_min_count, seed=seed, use_weighted_fit=use_weighted_fit)
        
        plot_scaling_results(f_name, invert = invert, input_array=input_array, sizes=fit_info_dict['valid_sizes'], measures=fit_info_dict['valid_counts'], d_value=fit_info_dict['D'], fit=fit_info_dict['fit'], r2=fit_info_dict['R2'], save=save, save_path=save_path, show_image=True)
        show_image_info(fname=f_name, invert=invert, d_value=fit_info_dict['D'], input_array=input_array, sizes=fit_info_dict['valid_sizes'], save=save, save_path=save_path)
        
        print('')
        print(f"D-value for {f_name}: {fit_info_dict['D']:.3f}")
        print(f"R^2 value for fit: {fit_info_dict['R2']:.6f}\n")
        print(f"Total pattern width: {input_array.shape[1]}")
        print(f"Total pattern height: {input_array.shape[0]}\n")
        print(f"Smallest object diameter: {smallest_diameter:.1f}")
        print(f"Largest object diameter: {largest_diameter:.1f}\n")
        print(f"min box width: {min_size:.1f}")
        print(f"max box width: {max_size:.1f}")

        D_value_list.append((os.path.splitext(f_name)[0], fit_info_dict['D'], min_size, max_size, fit_info_dict['R2']))

        if save:
            txt_save_path = os.path.join(save_path, 'image_properties.txt') 
            with open(txt_save_path, mode='w') as file:
                file.write(f"D-value for {f_name}: {fit_info_dict['D']:.3f}\n")
                file.write(f"R^2 value for fit {fit_info_dict['R2']}\n\n")
                file.write(f"Total pattern width: {np.max(input_array.shape[1])}\n")
                file.write(f"Total pattern height: {np.max(input_array.shape[0])}\n\n")
                file.write(f"Smallest object diameter: {smallest_diameter:.1f}\n")
                file.write(f"Largest object diameter: {largest_diameter:.1f}\n\n")
                file.write(f"min box width: {min_size:.1f}\n")
                file.write(f"max box width: {max_size:.1f}\n\n")

            # Convert boolean array to int before normalization
            input_array_int = input_array.astype(np.uint8)
            norm_array = ((input_array_int - input_array_int.min()) / np.ptp(input_array_int) * 255).astype(np.uint8)
            if norm_array.ndim == 3 and norm_array.shape[2] == 1:
                norm_array = np.squeeze(norm_array, axis=2)
            img = Image.fromarray(norm_array, mode='L')
            tiff_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_thresholded.tif")

            img.save(tiff_file, format='TIFF')

    df = pd.DataFrame(D_value_list, columns=['Image Name', 'D Value', 'Min Box Width', 'Max Box Width', 'R2 Score'])

    if save:
        csv_save_path = os.path.join(base_path, f'D_values_{criteria}.csv')
        df.to_csv(csv_save_path, index=False)

    print('')
    print('___________________________________________________________')
    print()

    print(df.to_string(index=False, justify='center', formatters={
        'Image Name': lambda x: f'{x:^30}:',  
        'D Value': lambda x: f'{np.round(x, decimals=2):^10}'      
    }))
    
    return df


