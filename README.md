# FracStack

A Python package for fractal analysis and box counting with advanced visualization capabilities.

## Installation

### For Development (Recommended for Lab Members)

1. Clone the repository:
```bash
git clone https://github.com/apdill/FracStack.git
cd FracStack
```

2. Install in development mode:
```bash
pip install -e .
```

This installs the package in "editable" mode, so any changes to the code will be immediately available without reinstalling.

### Direct Installation from Git

```bash
pip install git+https://github.com/apdill/FracStack.git
```

## Dependencies

The package requires the following Python packages:
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scikit-image >= 0.18.0
- scipy >= 1.6.0
- tqdm >= 4.50.0
- scikit-learn >= 0.24.0
- pandas >= 2.2.0
- numba >= 0.50.0
- Pillow >= 8.0.0

These will be automatically installed when you install the package.

## Quick Start

```python
import fracstack
import numpy as np

# Create a simple binary pattern
pattern = np.random.choice([0, 1], size=(256, 256), p=[0.8, 0.2])

# Basic fractal dimension measurement
result = fracstack.measure_dimension(pattern, mode='D0')
print(f"Fractal dimension: {result['D']:.3f}")

# Advanced analysis with visualization
analysis = fracstack.portfolio_plot(pattern, compute_dimensions='both', 
                                   use_dynamic=True, show_plot=True)
```

## Main Functions

### Core Analysis
- `measure_dimension()`: Basic fractal dimension measurement
- `portfolio_plot()`: Advanced analysis with visualization
- `dynamic_boxcount()`: Optimal scaling range detection
- `boxcount()`: Low-level box counting interface

### Image Processing
- `process_image_to_array()`: Load and binarize images
- `pad_image_for_boxcounting()`: Edge effect mitigation
- `find_largest_smallest_objects()`: Object size analysis

### Visualization
- `visualize_box_overlay()`: Show box counting grids
- `plot_pairwise_slopes()`: Detailed scaling analysis
- `show_image_info()`: Display image properties

## Examples

See the examples directory for detailed usage examples and tutorials.

## Support

For questions or issues, please contact the development team or create an issue on GitHub. 