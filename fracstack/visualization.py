import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from skimage.measure import find_contours # type: ignore
from tqdm.auto import tqdm
import os
from .image_processing import invert_array, pad_image_for_boxcounting
from numba import njit # type: ignore
from .boxcount import generate_random_offsets, numba_d0_optimized, numba_d1_optimized, numba_d2_optimized


def _format_dimension_suffix(d_value) -> str:
    try:
        d_float = float(d_value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(d_float):
        return ""
    formatted = f"{d_float:.3f}".rstrip('0').rstrip('.')
    return f"_D{formatted}" if formatted else ""


@njit(nogil=True, cache=True)
def numba_d0_visualize(array, size, offsets):
    """
    Numba-accelerated box counting for visualization purposes.
    Returns the minimum count and the corresponding grid offset.
    
    Args:
        array (np.ndarray): 2D binary array to analyze
        size (int): Box size to use
        offsets (np.ndarray): Pre-generated offsets array of shape (num_offsets, 2)
        
    Returns:
        tuple: (min_count, best_x_off, best_y_off) - the minimum count and the corresponding offsets
    """
    
    H, W = array.shape
    
    # Calculate actual centered offsets for this array size
    centered_x = (H % size) // 2
    centered_y = (W % size) // 2
    total_offsets = min(offsets.shape[0], size**2)
    
    min_count = np.inf
    best_x_off = centered_x
    best_y_off = centered_y
    
    for offset_idx in range(total_offsets):
        if offset_idx == 0:
            x_off = centered_x
            y_off = centered_y
        else:
            # Use pre-generated offsets
            x_off = offsets[offset_idx, 0] % size
            y_off = offsets[offset_idx, 1] % size
        
        # Box counting logic
        count = 0
        max_x = x_off + ((H - x_off) // size) * size
        max_y = y_off + ((W - y_off) // size) * size
        
        for x in range(x_off, max_x, size):
            for y in range(y_off, max_y, size):
                count += array[x:x+size, y:y+size].any()
        
        if count < min_count:
            min_count = count
            best_x_off = x_off
            best_y_off = y_off
    
    return min_count, best_x_off, best_y_off


def visualize_box_overlay(array, size, mode='D0', figsize=(10, 10), alpha=0.1, use_min_count=False, 
                          num_offsets=100, return_count=False, ax=None, gridline_cutoff=128,
                          pad_factor=1.5, use_optimization=True, sparse_threshold=0.01, seed=None):
    """
    Visualize the boxes used in box counting by overlaying them on the binary image.
    Updated to match the current implementation in portfolio_plot and dynamic_boxcount.
    
    Args:
        array (np.ndarray): 2D binary array to analyze
        size (int): Size of boxes to visualize
        mode (str): 'D0' for capacity dimension or 'D1' for information dimension
        figsize (tuple): Figure size for the plot
        alpha (float): Transparency of the box overlays
        use_min_count (bool): If True, use minimum count across offsets; if False, use average count (default: False)
        num_offsets (int): Number of offsets to try (default: 100)
        return_count (bool): If True, return the box count
        ax (matplotlib.axes.Axes, optional): If provided, plot on this axis instead of creating a new figure
        gridline_cutoff (int): Minimum box size to show grid lines (default: 128)
        pad_factor (float): Padding factor for box counting (default: 1.5)
        use_optimization (bool): Whether to use optimized box counting algorithms (default: True)
        sparse_threshold (float): Threshold for sparse optimization (default: 0.01)
        seed (int, optional): Random seed for reproducible results

    Returns:
        int or None: Box count if return_count is True, otherwise None
    """
    
    # Store original array for visualization
    original_array = array.copy()
    
    # Ensure array is contiguous for numba
    array = np.ascontiguousarray(array.astype(np.float32))
    
    # Apply padding if specified (matching portfolio_plot behavior)
    if pad_factor is not None and pad_factor > 1.0:
        array = pad_image_for_boxcounting(array, size, pad_factor=pad_factor)
    
    H, W = array.shape
    
    # Generate offsets using the same method as portfolio_plot
    offsets = generate_random_offsets([size], num_offsets, seed=seed)
    
    # Use the same optimized box counting functions as portfolio_plot
    if mode == 'D0':
        if use_optimization:
            # Calculate sparsity to choose optimization strategy
            total_pixels = array.size
            non_zero_pixels = np.count_nonzero(array)
            sparsity = non_zero_pixels / total_pixels
            
            # Use sparse optimization for very sparse arrays
            if sparsity <= sparse_threshold:
                from .boxcount import numba_d0_sparse
                count = numba_d0_sparse(array, np.array([size]), offsets, sparse_threshold, use_min_count)[0]
            else:
                count = numba_d0_optimized(array, np.array([size]), offsets, use_min_count)[0]
        else:
            from .boxcount import numba_d0
            count = numba_d0(array, np.array([size]), offsets, use_min_count)[0]
    elif mode == 'D1':
        if use_optimization:
            count = numba_d1_optimized(array, np.array([size]), offsets)[0]
        else:
            from .boxcount import numba_d1
            count = numba_d1(array, np.array([size]), offsets)[0]
    elif mode == 'D2':
        if use_optimization:
            count = numba_d2_optimized(array, np.array([size]), offsets)[0]
        else:
            from .boxcount import numba_d2
            count = numba_d2(array, np.array([size]), offsets)[0]
    else:
        raise ValueError("Invalid mode, use 'D0', 'D1', or 'D2'")
    
    # For visualization, we need to determine which offset was used
    # We'll use the centered offset for visualization consistency
    offset_x = (H % size) // 2
    offset_y = (W % size) // 2
    
    # Create figure and plot image (use original array for visualization)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(original_array, cmap='gray')
    
    # Draw grid lines if box size is large enough
    if size >= gridline_cutoff:
        # Create complete grid lines across the entire image (using original dimensions)
        orig_H, orig_W = original_array.shape
        x_grid = np.arange(offset_x, orig_H + size, size)  
        y_grid = np.arange(offset_y, orig_W + size, size)  
        
        ax.vlines(y_grid, 0, orig_H, colors='gray', linewidth=0.5, alpha=0.5)  
        ax.hlines(x_grid, 0, orig_W, colors='gray', linewidth=0.5, alpha=0.5)
    
    # Calculate the number of boxes in each dimension (using padded array dimensions)
    num_boxes_x = (H - offset_x + size - 1) // size
    num_boxes_y = (W - offset_y + size - 1) // size
    
    # Calculate offset adjustment if padding was applied
    orig_H, orig_W = original_array.shape
    pad_offset_x = (H - orig_H) // 2 if pad_factor is not None and pad_factor > 1.0 else 0
    pad_offset_y = (W - orig_W) // 2 if pad_factor is not None and pad_factor > 1.0 else 0
    
    # Highlight occupied boxes with rectangles
    for i in range(num_boxes_x):
        for j in range(num_boxes_y):
            x = offset_x + i * size
            y = offset_y + j * size
            
            # Calculate box dimensions, handling edge cases
            box_height = min(size, H - x)
            box_width = min(size, W - y)
            
            # Skip if box is completely outside the image
            if box_height <= 0 or box_width <= 0:
                continue
                
            # Get the portion of the array for this box
            box = array[x:x+box_height, y:y+box_width]
            
            # Check if box is occupied
            if mode == 'D0':
                is_occupied = box.any()
            elif mode in ('D1', 'D2'):
                is_occupied = box.sum() > 0
            else:
                is_occupied = False

            if is_occupied:
                # Adjust coordinates for visualization on original image
                vis_x = x - pad_offset_x
                vis_y = y - pad_offset_y
                vis_height = min(box_height, orig_H - vis_x) if vis_x >= 0 else box_height + vis_x
                vis_width = min(box_width, orig_W - vis_y) if vis_y >= 0 else box_width + vis_y
                
                # Only draw rectangle if it's within the original image bounds
                if vis_x < orig_H and vis_y < orig_W and vis_height > 0 and vis_width > 0:
                    vis_x = max(0, vis_x)
                    vis_y = max(0, vis_y)
                    
                    rect = Rectangle((vis_y, vis_x), vis_width, vis_height,
                              fill=True,
                              edgecolor='red',
                              facecolor='red',
                              alpha=alpha)
                    ax.add_patch(rect)
    
    # Add title with count information
    count_type = "min" if use_min_count else "avg"
    ax.set_title(f'{mode} Box Counting: size={size}, count={count:.1f} ({count_type})', fontsize=14)
    ax.axis('off')
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    if return_count:
        return count


def plot_scaling_results(f_name, 
                         input_array, 
                         sizes, 
                         measures, 
                         d_value, 
                         fit, 
                         r2, 
                         mode = 'D0', 
                         show_image = True, 
                         save=False, 
                         save_path=None, 
                         invert = False,
                         legend_info = None,
                         legend = True):
    
    d_suffix = _format_dimension_suffix(d_value)
    mode_suffix = f"_{mode}"  # Add mode suffix (D0, D1, D2) to distinguish from dimension value

    # Plot the original image
    if show_image == True:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
        
        # For visualization, show the original orientation (invert back if needed)
        # This ensures edge-detected images are shown in their original form for clarity
        display_array = input_array.copy()
        if invert:
            display_array = invert_array(display_array)
        
        ax1.imshow(display_array, cmap='gray')
        
        ax1.set_title(f"{os.path.splitext(f_name)[0]} || D value: {np.round(d_value, decimals=2)}", fontsize = 22)
        ax1.axis('off')  
    
        # Plot the scaling (log-log) plot
        if mode == 'D0':
            ax2.scatter(np.log10(sizes), np.log10(measures), color='black')
            ax2.plot(np.log10(sizes), fit[0] * np.log10(sizes) + fit[1], color='red')
            ax2.set_title(r'Scaling Plot: $Log_{10}(Counts)$ vs. $Log_{10}(Box Size)$', fontsize = 22)
            ax2.set_ylabel(r'$Log_{10}(N_L)$', fontsize = 22)
        elif mode == 'D1':
            ax2.scatter(np.log10(sizes), measures, color='black')
            ax2.plot(np.log10(sizes), fit[0] * np.log2(sizes) + fit[1], color='red')
            ax2.set_title(r'Shannon Entropy vs. $Log_{2}(Box Size)$', fontsize = 22)
            ax2.set_ylabel(r'$H(L)$', fontsize = 22)
        elif mode == 'D2':
            log_sizes = np.log10(sizes)
            log_measures = np.log10(measures)
            ax2.scatter(log_sizes, log_measures, color='black')
            ax2.plot(log_sizes, fit[0] * log_sizes + fit[1], color='red')
            ax2.set_title(r'Correlation Sum vs. $Log_{10}(Box Size)$', fontsize = 22)
            ax2.set_ylabel(r'$Log_{10}(C_L)$', fontsize = 22)

        if legend == True:
            bc_info_text = f"D Value: {np.round(d_value, decimals=2)}"
            bc_info_text += f"\n$R^2$ = {np.round(r2, decimals=5)}"
            if legend_info is not None:
                smallest_feature = legend_info.get('smallest_feature_width', None)
                if smallest_feature is not None:
                    bc_info_text += f"\nSmallest feature: {np.round(smallest_feature, decimals=1)}"
                min_box = legend_info.get('min_box_size', np.round(sizes.min()))
                max_box = legend_info.get('max_box_size', np.round(sizes.max()))
                bc_info_text += f"\nFine cut-off = {np.round(min_box)}"
                bc_info_text += f"\nCoarse cut-off = {np.round(max_box)}"
                img_w = legend_info.get('image_width', None)
                img_h = legend_info.get('image_height', None)
                if img_w is not None:
                    bc_info_text += f"\nImage width: {img_w}"
                if img_h is not None:
                    bc_info_text += f"\nImage height: {img_h}"
                magnificiation = legend_info.get('magnificiation_range', None)
                if magnificiation is not None:
                    bc_info_text += f"\nMagnification range: {magnificiation}"
            
            ax2.text(0.55, 0.95, bc_info_text, transform=ax2.transAxes, fontsize=22, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))

        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_xlabel(r'$Log_{10}(L)$', fontsize = 22)
        
        plt.tight_layout()
        # plt.show()

        if save == True:
            # This is the two-panel result (image + scaling plot)
            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_result{mode_suffix}{d_suffix}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)


    elif show_image == False:

        plt.figure(figsize=(11,11))

        if mode == 'D0':
            plt.scatter(np.log10(sizes), np.log10(measures), color='black')
            plt.plot(np.log10(sizes), fit[0] * np.log10(sizes) + fit[1], color='red')
        elif mode == 'D1':
            plt.scatter(np.log10(sizes), measures, color='black')
            plt.plot(np.log10(sizes), fit[0] * -np.log2(1/sizes) + fit[1], color='red')
        elif mode == 'D2':
            log_sizes = np.log10(sizes)
            log_measures = np.log10(measures)
            plt.scatter(log_sizes, log_measures, color='black')
            plt.plot(log_sizes, fit[0] * log_sizes + fit[1], color='red')

        plt.title(f"{os.path.splitext(f_name)[0]}", fontsize=22)

        # Build richer legend using legend_info when available
        lines = [
            f"D Value: {np.round(d_value, 2)}",
            f"$R^2$ = {np.round(r2, 5)}",
        ]
        if legend_info is not None:
            smallest_feature = legend_info.get('smallest_feature_width', None)
            if smallest_feature is not None:
                lines.append(f"Smallest feature: {np.round(smallest_feature, 1)}")
            min_box = legend_info.get('min_box_size', np.round(sizes.min()))
            max_box = legend_info.get('max_box_size', np.round(sizes.max()))
            lines.append(f"Fine cut-off = {np.round(min_box)}")
            lines.append(f"Coarse cut-off = {np.round(max_box)}")
            img_w = legend_info.get('image_width', None)
            img_h = legend_info.get('image_height', None)
            if img_w is not None and img_h is not None:
                lines.append(f"Image width: {img_w}")
                lines.append(f"Image height: {img_h}")
            mag = legend_info.get('magnificiation_range', None)
            if mag is not None:
                lines.append(f"Magnification range: {mag}")
        else:
            # Fallback to minimal info if legend_info is missing
            lines.append(f"Smallest box size (L) = {np.round(sizes.min())}")
            lines.append(f"Largest box size (L) = {np.round(sizes.max())}")

        bc_info_text = "\n".join(lines)

        plt.text(
            0.65,
            0.95,
            bc_info_text,
            fontsize=18,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle="round", facecolor="#e9f1f7", alpha=0.6, edgecolor="#cbd5e1"),
        )
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xlabel(r'$Log(L)$', fontsize=18)
        plt.ylabel(r'$Log(N_L)$', fontsize=18)
        plt.tight_layout()

        if save == True:
            # This is the scaling plot only
            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_scaling_plot{mode_suffix}{d_suffix}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)
        

def plot_lacunarity_curve(sizes,
                          lacunarity,
                          mean_mass=None,
                          variance=None,
                          window_counts=None,
                          show=True,
                          save_path=None,
                          title=None,
                          ax=None):
    """
    Plot gliding-box lacunarity following Plotnick et al. (1996).

    Parameters
    ----------
    sizes : array-like
        Box sizes evaluated in the lacunarity analysis.
    lacunarity : array-like
        Lacunarity values Λ(L) corresponding to each box size.
    mean_mass : array-like, optional
        Mean mass per gliding box for each size. Displayed in the legend when
        provided to aid interpretation.
    variance : array-like, optional
        Mass variance per gliding box. Used for legend metadata.
    window_counts : array-like, optional
        Number of gliding windows contributing to each scale. Included in legend.
    show : bool, default True
        Whether to display the plot interactively.
    save_path : str, optional
        Path to save the figure. Directories are created if needed.
    title : str, optional
        Title label for the plot (e.g., filename).
    ax : matplotlib.axes.Axes, optional
        Existing axis to draw on. When None, a new figure is created and returned.

    Returns
    -------
    tuple
        (fig, ax) when a new figure is created; otherwise returns (None, ax).
    """
    sizes = np.asarray(sizes)
    lacunarity = np.asarray(lacunarity)
    mask = (
        np.isfinite(sizes)
        & np.isfinite(lacunarity)
        & (sizes > 0)
        & (lacunarity > 0)
    )

    if not np.any(mask):
        return (None, ax)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = None

    x = sizes[mask]
    y = lacunarity[mask]
    ax.plot(x, y, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Box Size (pixels)')
    ax.set_ylabel('Lacunarity Λ(L)')
    ax.set_title(title if title else 'Lacunarity Curve')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    

    legend_lines = []
    if mean_mass is not None:
        mean_mass = np.asarray(mean_mass)
        if mean_mass.shape == sizes.shape:
            mean_vals = mean_mass[mask]
            legend_lines.append(f"Mean mass range: {mean_vals.min():.2f} - {mean_vals.max():.2f}")
    if variance is not None:
        variance = np.asarray(variance)
        if variance.shape == sizes.shape:
            var_vals = variance[mask]
            legend_lines.append(f"Variance range: {var_vals.min():.2f} - {var_vals.max():.2f}")
    if window_counts is not None:
        window_counts = np.asarray(window_counts)
        if window_counts.shape == sizes.shape:
            wc_vals = window_counts[mask]
            legend_lines.append(f"Windows per scale: min {wc_vals.min()}, max {wc_vals.max()}")

    if legend_lines:
        ax.legend(['\n'.join(legend_lines)], loc='best', frameon=False)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        target_fig = fig if fig is not None else ax.figure
        target_fig.savefig(save_path, dpi=300, bbox_inches='tight')

    if show and created_fig:
        plt.show()

    if created_fig and not show:
        plt.close(fig)

    return (fig, ax)


def plot_object_outlines(image, largest_object, smallest_object, invert=False, figsize=(8,8), f_name=None, save_dir=None, 
                          largest_diameter=None, smallest_diameter=None, min_feature_width=None):

    # Ensure we have a binary image for display
    if image.max() > 1:
        # If it's a labeled image, convert to binary
        display_image = (image > 0).astype(np.uint8)
    else:
        display_image = image.copy()
    
    if invert:
        if display_image.dtype == np.bool or display_image.max() <= 1:
            display_image = 1 - display_image  
        else:
            display_image = 255 - display_image  
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(display_image, cmap='gray', vmin=0, vmax=1)
    
    def plot_contours(obj, color, label):
        if obj is None:
            return

        obj_mask = obj.image
        if obj_mask.shape[0] < 1 or obj_mask.shape[1] < 1:
            print(f"Skipping {label}: object mask is too small ({obj_mask.shape}).")
            return
        
        contours = find_contours(obj_mask, level=0.5)
        
        for contour in contours:
            
            contour[:, 0] += obj.bbox[0]
            contour[:, 1] += obj.bbox[1]
            
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color=color)
        
        ax.plot([], [], color=color, linewidth=2, label=label)

    plot_contours(largest_object, 'red', 'Largest Object Outline')
    plot_contours(smallest_object, 'blue', 'Min Feature Width Object')
    
    #ax.legend(loc='upper right')
    ax.axis('off')  
    plt.tight_layout()

    if save_dir is not None:
        save_file = os.path.join(save_dir, f"{os.path.splitext(f_name)[0]}_object_outlines.png")
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def illustrate_boxcounting_regions(input_array, sizes, counts, invert = False):
    for size, count in zip(sizes, counts):
        fig, ax = plt.subplots(figsize=(10, 10))
        if invert == True:
            ax.imshow(invert_array(input_array), cmap='gray')
        else:
            ax.imshow(input_array, cmap='gray')
        ax.set_title(f"Box size: {size}, Count: {count}")

        # Highlight overlapping regions
        for i in range(0, input_array.shape[0], size):
            for j in range(0, input_array.shape[1], size):
                if np.any(input_array[i:i + size, j:j + size]):
                    rect = plt.Rectangle((j, i), size, size, fill=True, edgecolor='red', 
                                        facecolor='red', alpha=0.2)
                    ax.add_patch(rect)

        plt.show()


def create_boxcounting_animation(input_array, sizes, counts, fps = 10, invert=False, save_path=None):
    """
    Create an animation (GIF) of the box-counting process.

    Args:
        input_array (np.ndarray): Input binary array to perform box-counting on.
        sizes (list): List of box sizes.
        counts (list): List of box counts corresponding to the sizes.
        invert (bool): Whether to invert the input array for visualization. Default is False.
        save_path (str, optional): Path to save the resulting GIF. Default is None (no saving).

    Returns:
        None
    """

    sizes = sizes[::-1]
    counts = counts[::-1]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    if invert:
        display_array = invert_array(input_array)
    else:
        display_array = input_array

    height, width = input_array.shape
    center_x, center_y = width / 2, height / 2

    pbar = tqdm(total=len(sizes), desc="Creating Animation Frames", leave=True)

    def update(frame):
        # Clear the axes for the current frame
        ax.clear()
        size = sizes[frame]
        count = counts[frame]

        offset_x = int((center_x % size) - size / 2)
        offset_y = int((center_y % size) - size / 2)

        ax.imshow(display_array, cmap='gray')
        ax.set_title(f"Box size: {size}, Count: {count}")
        ax.axis('off')

        for i in range(offset_y, height, size):
            for j in range(offset_x, width, size):
                if 0 <= i < height and 0 <= j < width and np.any(input_array[i:i + size, j:j + size]):
                    rect = plt.Rectangle((j, i), size, size, fill=True, edgecolor='red', 
                                          facecolor='red', alpha=0.2)
                    ax.add_patch(rect)

        pbar.update(1)

    anim = FuncAnimation(fig, update, frames=len(sizes), repeat=True)

    # Save the animation as a GIF if a path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)  # Adjust fps for speed control
        print(f"Animation saved at {save_path}")

    plt.close(fig)

def show_largest_box_frame(input_array, sizes, counts, invert=False):
    """
    Display the frame with the largest boxes and count the number of boxes used.

    Args:
        input_array (np.ndarray): Input binary array to perform box-counting on.
        sizes (list): List of box sizes.
        counts (list): List of box counts corresponding to the sizes.
        invert (bool): Whether to invert the input array for visualization. Default is False.

    Returns:
        int: Number of boxes used for the largest box size.
    """
    # Find the largest box size and corresponding count
    largest_box_size = sizes[-1]
    largest_box_count = counts[-1]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    if invert:
        display_array = invert_array(input_array)
    else:
        display_array = input_array

    height, width = input_array.shape
    center_x, center_y = width / 2, height / 2

    # Calculate offsets for grid alignment
    offset_x = int((center_x % largest_box_size) - largest_box_size / 2)
    offset_y = int((center_y % largest_box_size) - largest_box_size / 2)

    ax.imshow(display_array, cmap='gray')
    ax.set_title(f"Largest Box size: {largest_box_size}, Count: {largest_box_count}")
    ax.axis('off')

    # Add rectangles to represent boxes
    for i in range(offset_y, height, largest_box_size):
        for j in range(offset_x, width, largest_box_size):
            if 0 <= i < height and 0 <= j < width and np.any(input_array[i:i + largest_box_size, j:j + largest_box_size]):
                rect = plt.Rectangle((j, i), largest_box_size, largest_box_size, fill=True, edgecolor='red',
                                      facecolor='red', alpha=0.2)
                ax.add_patch(rect)

    plt.show()

    return largest_box_count


def show_image_info(fname, 
                    d_value, 
                    input_array, 
                    sizes, 
                    invert = False, 
                    figsize = (11,11), 
                    save = False, 
                    save_path = None, 
                    largest_diameter = None, 
                    smallest_diameter = None,
                    min_feature_width = None,
                    r2 = None):

    plt.figure(figsize=figsize)
    # Display the input_array as-is since it's already been fully processed 
    # (including thresholding, inversion, and edge detection if enabled)
    plt.imshow(input_array, cmap='gray')
    plt.axis('off')  # This turns off the axes (ticks and borders)
    plt.title(f"{os.path.splitext(fname)[0]}", fontsize=22)  # Title above the image

    info_text = f"D Value: {np.round(d_value, decimals=2)} \n$R^2$ = {np.round(r2, decimals=5)} \nSmallest box size (L) = {np.round(sizes.min())} \nLargest box size (L) = {np.round(sizes.max())}"

    if largest_diameter is not None and min_feature_width is not None:
        info_text += f"\nSmallest feature: {np.round(min_feature_width, decimals=1)} \nLargest object: {np.round(largest_diameter, decimals=1)}"

    plt.text(0.5, -0.05, info_text, fontsize=22, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='center', bbox=dict(boxstyle="round", alpha=0.1))

    if save == True:
        if save_path is not None:
            # This is just the image with info text - not used by GUI
            save_file = os.path.join(save_path, f"image_info.png")
            plt.savefig(save_file, bbox_inches='tight')
        else: print('no save path for image info!')



def showim(im_array, figsize=(4, 4), show_hist=False, nbins=None, bin_width=None, cmap='gray', vmin=None, vmax=None, titles=None):
    
    if isinstance(im_array, (list, tuple)):
        n_images = len(im_array)
        fig_width, fig_height = figsize
        plt.figure(figsize=(fig_width * n_images, fig_height))
        
        for idx, img in enumerate(im_array):
            plt.subplot(1, n_images, idx + 1)
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, (list, tuple)) and len(titles) == n_images:
                plt.title(titles[idx])
            elif titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
        plt.tight_layout()
        
        plt.show()
    else:
        plt.figure(figsize=figsize)
        
        if show_hist:
            plt.subplot(1, 2, 1)
            plt.imshow(im_array, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
            plt.subplot(1, 2, 2)
            
            im_flattened = im_array.ravel()
            min_val = np.floor(im_flattened.min())
            max_val = np.ceil(im_flattened.max())
            
            if bin_width is not None:
                bins = np.arange(min_val, max_val + bin_width, bin_width)
            elif nbins is not None:
                bins = nbins
            else:
                bins = int(max_val - min_val)
            
            plt.hist(im_flattened, bins=bins, color='black')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.title('Image Intensity Histogram')
        
        else:
            plt.imshow(im_array, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
        plt.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from skimage.measure import find_contours # type: ignore
from tqdm.auto import tqdm
import os
from .image_processing import invert_array, pad_image_for_boxcounting
from numba import njit # type: ignore
from .boxcount import generate_random_offsets, numba_d0_optimized, numba_d1_optimized


@njit(nogil=True, cache=True)
def numba_d0_visualize(array, size, offsets):
    """
    Numba-accelerated box counting for visualization purposes.
    Returns the minimum count and the corresponding grid offset.
    
    Args:
        array (np.ndarray): 2D binary array to analyze
        size (int): Box size to use
        offsets (np.ndarray): Pre-generated offsets array of shape (num_offsets, 2)
        
    Returns:
        tuple: (min_count, best_x_off, best_y_off) - the minimum count and the corresponding offsets
    """
    
    H, W = array.shape
    
    # Calculate actual centered offsets for this array size
    centered_x = (H % size) // 2
    centered_y = (W % size) // 2
    total_offsets = min(offsets.shape[0], size**2)
    
    min_count = np.inf
    best_x_off = centered_x
    best_y_off = centered_y
    
    for offset_idx in range(total_offsets):
        if offset_idx == 0:
            x_off = centered_x
            y_off = centered_y
        else:
            # Use pre-generated offsets
            x_off = offsets[offset_idx, 0] % size
            y_off = offsets[offset_idx, 1] % size
        
        # Box counting logic
        count = 0
        max_x = x_off + ((H - x_off) // size) * size
        max_y = y_off + ((W - y_off) // size) * size
        
        for x in range(x_off, max_x, size):
            for y in range(y_off, max_y, size):
                count += array[x:x+size, y:y+size].any()
        
        if count < min_count:
            min_count = count
            best_x_off = x_off
            best_y_off = y_off
    
    return min_count, best_x_off, best_y_off


def visualize_box_overlay(array, size, mode='D0', figsize=(10, 10), alpha=0.1, use_min_count=False, 
                          num_offsets=100, return_count=False, ax=None, gridline_cutoff=128,
                          pad_factor=1.5, use_optimization=True, sparse_threshold=0.01, seed=None):
    """
    Visualize the boxes used in box counting by overlaying them on the binary image.
    Updated to match the current implementation in portfolio_plot and dynamic_boxcount.
    
    Args:
        array (np.ndarray): 2D binary array to analyze
        size (int): Size of boxes to visualize
        mode (str): 'D0' for capacity dimension or 'D1' for information dimension
        figsize (tuple): Figure size for the plot
        alpha (float): Transparency of the box overlays
        use_min_count (bool): If True, use minimum count across offsets; if False, use average count (default: False)
        num_offsets (int): Number of offsets to try (default: 100)
        return_count (bool): If True, return the box count
        ax (matplotlib.axes.Axes, optional): If provided, plot on this axis instead of creating a new figure
        gridline_cutoff (int): Minimum box size to show grid lines (default: 128)
        pad_factor (float): Padding factor for box counting (default: 1.5)
        use_optimization (bool): Whether to use optimized box counting algorithms (default: True)
        sparse_threshold (float): Threshold for sparse optimization (default: 0.01)
        seed (int, optional): Random seed for reproducible results

    Returns:
        int or None: Box count if return_count is True, otherwise None
    """
    
    # Store original array for visualization
    original_array = array.copy()
    
    # Ensure array is contiguous for numba
    array = np.ascontiguousarray(array.astype(np.float32))
    
    # Apply padding if specified (matching portfolio_plot behavior)
    if pad_factor is not None and pad_factor > 1.0:
        array = pad_image_for_boxcounting(array, size, pad_factor=pad_factor)
    
    H, W = array.shape
    
    # Generate offsets using the same method as portfolio_plot
    offsets = generate_random_offsets([size], num_offsets, seed=seed)
    
    # Use the same optimized box counting functions as portfolio_plot
    if mode == 'D0':
        if use_optimization:
            # Calculate sparsity to choose optimization strategy
            total_pixels = array.size
            non_zero_pixels = np.count_nonzero(array)
            sparsity = non_zero_pixels / total_pixels
            
            # Use sparse optimization for very sparse arrays
            if sparsity <= sparse_threshold:
                from .boxcount import numba_d0_sparse
                count = numba_d0_sparse(array, np.array([size]), offsets, sparse_threshold, use_min_count)[0]
            else:
                count = numba_d0_optimized(array, np.array([size]), offsets, use_min_count)[0]
        else:
            from .boxcount import numba_d0
            count = numba_d0(array, np.array([size]), offsets, use_min_count)[0]
    elif mode == 'D1':
        if use_optimization:
            count = numba_d1_optimized(array, np.array([size]), offsets)[0]
        else:
            from .boxcount import numba_d1
            count = numba_d1(array, np.array([size]), offsets)[0]
    else:
        raise ValueError("Invalid mode, use 'D0' or 'D1'")
    
    # For visualization, we need to determine which offset was used
    # We'll use the centered offset for visualization consistency
    offset_x = (H % size) // 2
    offset_y = (W % size) // 2
    
    # Create figure and plot image (use original array for visualization)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    ax.imshow(original_array, cmap='gray')
    
    # Draw grid lines if box size is large enough
    if size >= gridline_cutoff:
        # Create complete grid lines across the entire image (using original dimensions)
        orig_H, orig_W = original_array.shape
        x_grid = np.arange(offset_x, orig_H + size, size)  
        y_grid = np.arange(offset_y, orig_W + size, size)  
        
        ax.vlines(y_grid, 0, orig_H, colors='gray', linewidth=0.5, alpha=0.5)  
        ax.hlines(x_grid, 0, orig_W, colors='gray', linewidth=0.5, alpha=0.5)
    
    # Calculate the number of boxes in each dimension (using padded array dimensions)
    num_boxes_x = (H - offset_x + size - 1) // size
    num_boxes_y = (W - offset_y + size - 1) // size
    
    # Calculate offset adjustment if padding was applied
    orig_H, orig_W = original_array.shape
    pad_offset_x = (H - orig_H) // 2 if pad_factor is not None and pad_factor > 1.0 else 0
    pad_offset_y = (W - orig_W) // 2 if pad_factor is not None and pad_factor > 1.0 else 0
    
    # Highlight occupied boxes with rectangles
    for i in range(num_boxes_x):
        for j in range(num_boxes_y):
            x = offset_x + i * size
            y = offset_y + j * size
            
            # Calculate box dimensions, handling edge cases
            box_height = min(size, H - x)
            box_width = min(size, W - y)
            
            # Skip if box is completely outside the image
            if box_height <= 0 or box_width <= 0:
                continue
                
            # Get the portion of the array for this box
            box = array[x:x+box_height, y:y+box_width]
            
            # Check if box is occupied
            if mode == 'D0':
                is_occupied = box.any()
            elif mode == 'D1':
                is_occupied = box.sum() > 0
            
            if is_occupied:
                # Adjust coordinates for visualization on original image
                vis_x = x - pad_offset_x
                vis_y = y - pad_offset_y
                vis_height = min(box_height, orig_H - vis_x) if vis_x >= 0 else box_height + vis_x
                vis_width = min(box_width, orig_W - vis_y) if vis_y >= 0 else box_width + vis_y
                
                # Only draw rectangle if it's within the original image bounds
                if vis_x < orig_H and vis_y < orig_W and vis_height > 0 and vis_width > 0:
                    vis_x = max(0, vis_x)
                    vis_y = max(0, vis_y)
                    
                    rect = Rectangle((vis_y, vis_x), vis_width, vis_height,
                              fill=True,
                              edgecolor='red',
                              facecolor='red',
                              alpha=alpha)
                    ax.add_patch(rect)
    
    # Add title with count information
    count_type = "min" if use_min_count else "avg"
    ax.set_title(f'{mode} Box Counting: size={size}, count={count:.1f} ({count_type})', fontsize=14)
    ax.axis('off')
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    if return_count:
        return count
def illustrate_boxcounting_regions(input_array, sizes, counts, invert = False):
    for size, count in zip(sizes, counts):
        fig, ax = plt.subplots(figsize=(10, 10))
        if invert == True:
            ax.imshow(invert_array(input_array), cmap='gray')
        else:
            ax.imshow(input_array, cmap='gray')
        ax.set_title(f"Box size: {size}, Count: {count}")

        # Highlight overlapping regions
        for i in range(0, input_array.shape[0], size):
            for j in range(0, input_array.shape[1], size):
                if np.any(input_array[i:i + size, j:j + size]):
                    rect = plt.Rectangle((j, i), size, size, fill=True, edgecolor='red', 
                                        facecolor='red', alpha=0.2)
                    ax.add_patch(rect)

        plt.show()


def create_boxcounting_animation(input_array, sizes, counts, fps = 10, invert=False, save_path=None):
    """
    Create an animation (GIF) of the box-counting process.

    Args:
        input_array (np.ndarray): Input binary array to perform box-counting on.
        sizes (list): List of box sizes.
        counts (list): List of box counts corresponding to the sizes.
        invert (bool): Whether to invert the input array for visualization. Default is False.
        save_path (str, optional): Path to save the resulting GIF. Default is None (no saving).

    Returns:
        None
    """

    sizes = sizes[::-1]
    counts = counts[::-1]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    if invert:
        display_array = invert_array(input_array)
    else:
        display_array = input_array

    height, width = input_array.shape
    center_x, center_y = width / 2, height / 2

    pbar = tqdm(total=len(sizes), desc="Creating Animation Frames", leave=True)

    def update(frame):
        # Clear the axes for the current frame
        ax.clear()
        size = sizes[frame]
        count = counts[frame]

        offset_x = int((center_x % size) - size / 2)
        offset_y = int((center_y % size) - size / 2)

        ax.imshow(display_array, cmap='gray')
        ax.set_title(f"Box size: {size}, Count: {count}")
        ax.axis('off')

        for i in range(offset_y, height, size):
            for j in range(offset_x, width, size):
                if 0 <= i < height and 0 <= j < width and np.any(input_array[i:i + size, j:j + size]):
                    rect = plt.Rectangle((j, i), size, size, fill=True, edgecolor='red', 
                                          facecolor='red', alpha=0.2)
                    ax.add_patch(rect)

        pbar.update(1)

    anim = FuncAnimation(fig, update, frames=len(sizes), repeat=True)

    # Save the animation as a GIF if a path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)  # Adjust fps for speed control
        print(f"Animation saved at {save_path}")

    plt.close(fig)

def show_largest_box_frame(input_array, sizes, counts, invert=False):
    """
    Display the frame with the largest boxes and count the number of boxes used.

    Args:
        input_array (np.ndarray): Input binary array to perform box-counting on.
        sizes (list): List of box sizes.
        counts (list): List of box counts corresponding to the sizes.
        invert (bool): Whether to invert the input array for visualization. Default is False.

    Returns:
        int: Number of boxes used for the largest box size.
    """
    # Find the largest box size and corresponding count
    largest_box_size = sizes[-1]
    largest_box_count = counts[-1]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    if invert:
        display_array = invert_array(input_array)
    else:
        display_array = input_array

    height, width = input_array.shape
    center_x, center_y = width / 2, height / 2

    # Calculate offsets for grid alignment
    offset_x = int((center_x % largest_box_size) - largest_box_size / 2)
    offset_y = int((center_y % largest_box_size) - largest_box_size / 2)

    ax.imshow(display_array, cmap='gray')
    ax.set_title(f"Largest Box size: {largest_box_size}, Count: {largest_box_count}")
    ax.axis('off')

    # Add rectangles to represent boxes
    for i in range(offset_y, height, largest_box_size):
        for j in range(offset_x, width, largest_box_size):
            if 0 <= i < height and 0 <= j < width and np.any(input_array[i:i + largest_box_size, j:j + largest_box_size]):
                rect = plt.Rectangle((j, i), largest_box_size, largest_box_size, fill=True, edgecolor='red',
                                      facecolor='red', alpha=0.2)
                ax.add_patch(rect)

    plt.show()

    return largest_box_count


def showim(im_array, figsize=(4, 4), show_hist=False, nbins=None, bin_width=None, cmap='gray', vmin=None, vmax=None, titles=None):
    
    if isinstance(im_array, (list, tuple)):
        n_images = len(im_array)
        fig_width, fig_height = figsize
        plt.figure(figsize=(fig_width * n_images, fig_height))
        
        for idx, img in enumerate(im_array):
            plt.subplot(1, n_images, idx + 1)
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, (list, tuple)) and len(titles) == n_images:
                plt.title(titles[idx])
            elif titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
        plt.tight_layout()
        
        plt.show()
    else:
        plt.figure(figsize=figsize)
        
        if show_hist:
            plt.subplot(1, 2, 1)
            plt.imshow(im_array, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
            plt.subplot(1, 2, 2)
            
            im_flattened = im_array.ravel()
            min_val = np.floor(im_flattened.min())
            max_val = np.ceil(im_flattened.max())
            
            if bin_width is not None:
                bins = np.arange(min_val, max_val + bin_width, bin_width)
            elif nbins is not None:
                bins = nbins
            else:
                bins = int(max_val - min_val)
            
            plt.hist(im_flattened, bins=bins, color='black')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.title('Image Intensity Histogram')
        
        else:
            plt.imshow(im_array, cmap=cmap, vmin=vmin, vmax=vmax)
            
            if titles and isinstance(titles, str):
                plt.title(titles)
            
            plt.axis('off')
        plt.tight_layout()
        plt.show()

