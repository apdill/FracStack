#!/usr/bin/env python3
"""
Example usage of 3D box counting functionality in FracStack.

This script demonstrates how to use the 3D box counting functions to analyze
3D fractal structures.
"""

import numpy as np
import matplotlib.pyplot as plt
from fracstack import boxcount_3d, measure_dimension_3d

def create_3d_fractal_example():
    """Create a simple 3D fractal-like structure for testing."""
    # Create a 3D array with some fractal-like structure
    size = 64
    array_3d = np.zeros((size, size, size), dtype=np.uint8)
    
    # Create a simple 3D fractal pattern (Menger sponge-like)
    def recursive_fill(x, y, z, s, level=0):
        if level > 2 or s < 3:
            return
        
        # Divide into 3x3x3 = 27 subcubes
        third = s // 3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Skip the center cube and face centers (simplified Menger sponge)
                    if not ((i == 1 and j == 1) or (i == 1 and k == 1) or (j == 1 and k == 1)):
                        new_x = x + i * third
                        new_y = y + j * third
                        new_z = z + k * third
                        
                        if level == 2:  # Fill at the final level
                            array_3d[new_x:new_x+third, new_y:new_y+third, new_z:new_z+third] = 1
                        else:
                            recursive_fill(new_x, new_y, new_z, third, level + 1)
    
    # Start the recursive pattern
    recursive_fill(0, 0, 0, size)
    
    return array_3d

def main():
    print("3D Box Counting Example")
    print("=" * 50)
    
    # Create a 3D fractal structure
    print("Creating 3D fractal structure...")
    array_3d = create_3d_fractal_example()
    
    print(f"3D Array shape: {array_3d.shape}")
    print(f"Non-zero voxels: {np.count_nonzero(array_3d)}")
    print(f"Sparsity: {np.count_nonzero(array_3d) / array_3d.size:.4f}")
    
    # Basic 3D box counting
    print("\nPerforming 3D box counting...")
    sizes, counts = boxcount_3d(array_3d, mode='D0', num_sizes=15, min_size=2, max_size=32, num_offsets=10)
    
    print(f"Tested {len(sizes)} box sizes from {min(sizes)} to {max(sizes)}")
    
    # Measure 3D fractal dimension
    print("\nMeasuring 3D fractal dimension...")
    result = measure_dimension_3d(array_3d, mode='D0', num_sizes=15, min_size=2, max_size=32, 
                                 num_offsets=20, use_bootstrap_ci=True, n_bootstrap=500)
    
    print(f"3D Fractal Dimension (D0): {result['D']:.4f}")
    print(f"R² value: {result['R2']:.6f}")
    print(f"95% Confidence Interval: [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    print(f"Valid data points used: {len(result['valid_sizes'])}")
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: 3D visualization (2D slices)
    plt.subplot(1, 2, 1)
    # Show middle slice
    middle_slice = array_3d.shape[0] // 2
    plt.imshow(array_3d[middle_slice, :, :], cmap='gray')
    plt.title(f'3D Structure (Middle Slice z={middle_slice})')
    plt.axis('off')
    
    # Plot 2: Scaling plot
    plt.subplot(1, 2, 2)
    plt.scatter(np.log10(result['valid_sizes']), np.log10(result['valid_counts']), 
                color='black', s=50, alpha=0.7)
    
    # Plot fit line
    log_sizes = np.log10(result['valid_sizes'])
    fit_line = result['fit'][0] * log_sizes + result['fit'][1]
    plt.plot(log_sizes, fit_line, color='red', linewidth=2)
    
    plt.xlabel('Log₁₀(Box Size)')
    plt.ylabel('Log₁₀(Box Count)')
    plt.title(f'3D Box Counting Scaling\nD₀ = {result["D"]:.4f}, R² = {result["R2"]:.4f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with different modes
    print("\nComparing D0 and D1 dimensions...")
    
    # D1 dimension
    result_d1 = measure_dimension_3d(array_3d, mode='D1', num_sizes=15, min_size=2, max_size=32, 
                                    num_offsets=20, use_bootstrap_ci=True, n_bootstrap=500)
    
    print(f"3D Information Dimension (D1): {result_d1['D']:.4f}")
    print(f"D1 R² value: {result_d1['R2']:.6f}")
    print(f"D1 95% Confidence Interval: [{result_d1['ci_low']:.4f}, {result_d1['ci_high']:.4f}]")
    
    # Performance comparison
    print("\nPerformance test...")
    import time
    
    start_time = time.time()
    for _ in range(5):
        boxcount_3d(array_3d, mode='D0', num_sizes=10, min_size=2, max_size=16, num_offsets=5)
    avg_time = (time.time() - start_time) / 5
    
    print(f"Average time per 3D box count: {avg_time:.3f} seconds")
    
    print("\n3D Box counting analysis complete!")

if __name__ == "__main__":
    main() 