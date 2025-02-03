from .boxcount import boxcount, compute_dimension, numba_d1, numba_d0
from .image_processing import (
    process_image_to_array,
    pad_image_for_boxcounting,
    find_largest_smallest_objects,
    find_largest_empty_spaces,
    create_mask_from_largest_object,
    invert_array,
    calculate_diameter,
    bounding_box_diameter
)
from .visualization import (
    plot_scaling_results,
    plot_object_outlines,
    show_largest_box_frame,
    create_boxcounting_animation,
    show_image_info,
    illustrate_boxcounting_regions
)
from .core import (
    measure_dimension,
    analyze_image,
    analyze_images,
    portfolio_plot)

__version__ = '0.1.0'
__author__ = 'DillyDilly'

__all__ = [
    # Core functionality
    'boxcount',
    'numba_d1',
    'numba_d0',
    'compute_dimension',
    'measure_dimension',
    'analyze_image',
    'analyze_images',
    'portfolio_plot',
    
    # Image processing
    'process_image_to_array',
    'pad_image_for_boxcounting',
    'find_largest_smallest_objects',
    'find_largest_empty_spaces',
    'create_mask_from_largest_object',
    'invert_array',
    'calculate_diameter',
    'bounding_box_diameter',
    
    # Visualization
    'plot_scaling_results',
    'plot_object_outlines',
    'show_largest_box_frame',
    'create_boxcounting_animation',
    'show_image_info',
    'illustrate_boxcounting_regions'
]