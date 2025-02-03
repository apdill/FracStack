import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.measure import find_contours
from tqdm.auto import tqdm
import os
from .image_processing import invert_array

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

