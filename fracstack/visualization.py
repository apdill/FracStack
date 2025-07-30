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
                         invert = False):
    
    # Plot the original image
    if show_image == True:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
        
        if invert == True:
            ax1.imshow(invert_array(input_array), cmap='gray')
        else:
            ax1.imshow(input_array, cmap='gray')
        
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

        bc_info_text = f"D Value: {np.round(d_value, decimals=2)} \nSmallest box size (L) = {np.round(sizes.min())} \nLargest box size (L) = {np.round(sizes.max())}"
        
        ax2.text(0.55, 0.95, bc_info_text, transform=ax2.transAxes, fontsize=22,
                verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_xlabel(r'$Log_{10}(L)$', fontsize = 22)
        

        if save == True:

            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_{d_value:.3f}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)
        
        plt.tight_layout()
        plt.show()


    elif show_image == False:

        plt.figure(figsize=(11,11))

        if mode == 'D0':
            plt.scatter(np.log10(sizes), np.log10(measures), color='black', )
            plt.plot(np.log10(sizes), fit[0] * np.log10(sizes) + fit[1], color='red')
        elif mode == 'D1':
            plt.scatter(np.log10(sizes), measures, color='black')
            plt.plot(np.log10(sizes), fit[0] * -np.log2(1/sizes) + fit[1], color='red')

        plt.title(f"{os.path.splitext(f_name)[0]}", fontsize = 22)
        bc_info_text = f"D Value: {np.round(d_value, decimals=2)} \nSmallest box size (L) = {np.round(sizes.min())} \nLargest box size (L) = {np.round(sizes.max())}"
        plt.text(0.5, 0.95, bc_info_text, fontsize=22, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlabel(r'$Log(L)$', fontsize = 22)
        plt.ylabel(r'$Log(N_L)$', fontsize = 22)
        
        if save == True:

            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_scaling_plot.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)
        
        plt.tight_layout()
        plt.show()

def plot_object_outlines(image, largest_object, smallest_object, invert=False, figsize=(8,8)):

    if invert:
    
        if image.dtype == np.bool or image.max() <= 1:
            image = 1 - image  
        else:
            image = 255 - image  
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    
    def plot_contours(obj, color, label):

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
    plot_contours(smallest_object, 'blue', 'Smallest Object Outline')
    
    #ax.legend(loc='upper right')
    ax.axis('off')  
    plt.tight_layout()
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


def show_image_info(fname, d_value, input_array, sizes, invert = False, figsize = (11,11), save = False, save_path = None):

    plt.figure(figsize=figsize)
    if invert is True:
        plt.imshow(invert_array(input_array), cmap='gray')
    else:
        plt.imshow(input_array, cmap='gray')
    plt.axis('off')  # This turns off the axes (ticks and borders)
    plt.title(f"{os.path.splitext(fname)[0]}", fontsize=22)  # Title above the image
    plt.text(0.5, -0.2, 
             f" D Value: {np.round(d_value, decimals=2):}\nSmallest box size = {np.round(sizes.min())} pixels\nLargest box size = {np.round(sizes.max())} pixels", 
             fontsize=18, ha='center', transform=plt.gca().transAxes)
    if save == True:
        if save_path is not None:
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


def plot_scaling_results(f_name = 'fractal image', 
                         input_array = None, 
                         sizes = None, 
                         measures = None, 
                         d_value = None, 
                         fit = None, 
                         r2 = None, 
                         mode = 'D0', 
                         show_image = True, 
                         save=False, 
                         save_path=None, 
                         invert = False):
    
    # Plot the original image
    if show_image == True:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
        
        if invert == True:
            ax1.imshow(invert_array(input_array), cmap='gray')
        else:
            ax1.imshow(input_array, cmap='gray')
        
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

        bc_info_text = f"D Value: {np.round(d_value, decimals=2)} \nSmallest box size (L) = {np.round(sizes.min())} \nLargest box size (L) = {np.round(sizes.max())} \n$R^2$ = {np.round(r2, decimals=4)}"
        
        ax2.text(0.55, 0.95, bc_info_text, transform=ax2.transAxes, fontsize=22,
                verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=18)
        ax2.set_xlabel(r'$Log_{10}(L)$', fontsize = 22)
        plt.tight_layout()
        

        if save == True:

            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_{d_value:.3f}.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)



    elif show_image == False:

        plt.figure(figsize=(11,11))

        if mode == 'D0':
            plt.scatter(np.log10(sizes), np.log10(measures), color='black', )
            plt.plot(np.log10(sizes), fit[0] * np.log10(sizes) + fit[1], color='red')
        elif mode == 'D1':
            plt.scatter(np.log10(sizes), measures, color='black')
            plt.plot(np.log10(sizes), fit[0] * -np.log2(1/sizes) + fit[1], color='red')

        plt.title(f"{os.path.splitext(f_name)[0]}", fontsize = 22)
        bc_info_text = f"D Value: {np.round(d_value, decimals=2)} \nSmallest box size (L) = {np.round(sizes.min())} \nLargest box size (L) = {np.round(sizes.max())} \n$R^2$ = {np.round(r2, decimals=4)}"
        plt.text(0.5, 0.95, bc_info_text, fontsize=22, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.1))
        plt.grid(True)
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlabel(r'$Log(L)$', fontsize = 22)
        plt.ylabel(r'$Log(N_L)$', fontsize = 22)
        plt.tight_layout()
        
        if save == True:

            save_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_scaling_plot.png")
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            plt.savefig(save_file)
        

def plot_object_outlines(image, largest_object, smallest_object, invert=False, figsize=(8,8)):

    if invert:
    
        if image.dtype == np.bool or image.max() <= 1:
            image = 1 - image  
        else:
            image = 255 - image  
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')
    
    def plot_contours(obj, color, label):

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
    plot_contours(smallest_object, 'blue', 'Smallest Object Outline')
    
    #ax.legend(loc='upper right')
    ax.axis('off')  
    plt.tight_layout()
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


def show_image_info(fname, d_value, input_array, sizes, invert = False, figsize = (11,11), save = False, save_path = None):

    plt.figure(figsize=figsize)
    if invert is True:
        plt.imshow(invert_array(input_array), cmap='gray')
    else:
        plt.imshow(input_array, cmap='gray')
    plt.axis('off')  # This turns off the axes (ticks and borders)
    plt.title(f"{os.path.splitext(fname)[0]}", fontsize=22)  # Title above the image
    plt.text(0.5, -0.2, 
             f" D Value: {np.round(d_value, decimals=2):}\nSmallest box size = {np.round(sizes.min())} pixels\nLargest box size = {np.round(sizes.max())} pixels", 
             fontsize=18, ha='center', transform=plt.gca().transAxes)
    if save == True:
        if save_path is not None:
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

