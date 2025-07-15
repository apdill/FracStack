from .boxcount import (
    boxcount, 
    compute_dimension, 
    numba_d1, 
    numba_d0, 
    get_sizes,
    dynamic_boxcount,
    numba_d0_sparse,
    numba_d0_optimized,
    numba_d1_optimized,
    generate_random_offsets,
)
from .image_processing import (
    process_image_to_array,
    pad_image_for_boxcounting,
    find_largest_smallest_objects,
    find_largest_empty_spaces,
    create_mask_from_largest_object,
    invert_array,
    bounding_box_diameter,
    create_bounded_pattern
)
from .visualization import (
    plot_scaling_results,
    plot_object_outlines,
    show_largest_box_frame,
    create_boxcounting_animation,
    show_image_info,
    illustrate_boxcounting_regions,
    visualize_box_overlay,
    showim
)

from .core import (
    measure_dimension,
    analyze_image,
    analyze_images,
    portfolio_plot,
    plot_pairwise_slopes
)

__version__ = '0.1.1'
__author__ = 'DillyDilly'

__all__ = [
    # Core functionality
    'boxcount',
    'compute_dimension',
    'dynamic_boxcount',
    'numba_d0',
    'numba_d0_sparse',
    'numba_d0_optimized',
    'numba_d1_optimized',
    'generate_random_offsets',
    'measure_dimension',
    'analyze_image',
    'analyze_images',
    'portfolio_plot',
    'plot_pairwise_slopes',
    'get_sizes',

    # Image processing
    'process_image_to_array',
    'pad_image_for_boxcounting',
    'find_largest_smallest_objects',
    'find_largest_empty_spaces',
    'create_mask_from_largest_object',
    'invert_array',
    'bounding_box_diameter',
    'create_bounded_pattern',
    
    # Visualization
    'plot_scaling_results',
    'plot_object_outlines',
    'show_largest_box_frame',
    'create_boxcounting_animation',
    'show_image_info',
    'illustrate_boxcounting_regions',
    'visualize_box_overlay',
    'showim'
]