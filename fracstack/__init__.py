from .boxcount import boxcount, compute_fractal_dimension, get_sizes, get_mincount
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
    measure_D,
    analyze_image,
    analyze_images
)

__version__ = '0.1.0'
__author__ = 'DillyDilly'

__all__ = [
    # Core functionality
    'boxcount',
    'compute_fractal_dimension',
    'get_sizes',
    'get_mincount',
    'measure_D',
    'analyze_image',
    'analyze_images',
    
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