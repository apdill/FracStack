import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage.io as io

from .boxcount import boxcount, compute_fractal_dimension
from .image_processing import process_image_to_array, pad_image_for_boxcounting, find_largest_smallest_objects
from .visualization import plot_scaling_results, show_image_info, plot_object_outlines


def measure_D(array, num_sizes=10, min_size=None, max_size=None, num_pos=1, invert=False, mask=None):
    sizes, counts = boxcount(array, num_sizes, min_size, max_size, num_pos, invert, mask)
    valid_sizes, valid_counts, d_value, fit, r2 = compute_fractal_dimension(sizes, counts)
    return d_value, valid_sizes, valid_counts, fit, r2

def analyze_image(image_path, 
                  save=False, 
                  invert=True, 
                  threshold=150, 
                  min_size=16, 
                  max_size=None, 
                  num_sizes=100, 
                  num_pos=10,
                  show_image=False,
                  plot_objects=False,
                  criteria=None):
    """
    Analyze a single image to calculate fractal dimension and other properties.
    
    Args:
        image_path (str): Path to the image file
        save (bool): Whether to save results to files
        invert (bool): Whether to invert the image
        threshold (int): Threshold value for image binarization
        min_size (int): Minimum box size for box counting
        max_size (int): Maximum box size for box counting (defaults to min_size*100)
        num_sizes (int): Number of box sizes to use
        num_pos (int): Number of grid positions to test
    
    Returns:
        tuple: (d_value, min_size, max_size, r2) containing the fractal dimension and analysis parameters
    """
    f_name = os.path.basename(image_path)
    input_array = process_image_to_array(image_path, threshold=threshold, invert=invert).astype(np.uint8)
    
    im = io.imread(image_path)
    print(np.max(np.unique(im)))

    plt.imshow(input_array, cmap='gray')
    plt.show()

    largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image = find_largest_smallest_objects(input_array)

    if plot_objects:
        plot_object_outlines(labeled_image, largest_object, smallest_object)

    if max_size is None:
        max_size = min_size * 100

    if criteria is None:
        criteria = ''
    
    if save:
        save_dir = os.path.dirname(image_path)
        save_path = os.path.join(save_dir, os.path.splitext(f_name)[0] + f'_{criteria}')

        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = None

    padded_input_array = pad_image_for_boxcounting(input_array, max_size, pad_factor=1)
    sizes, counts = boxcount(padded_input_array, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_pos=num_pos)
    valid_sizes, valid_counts, d_value, fit, r2 = compute_fractal_dimension(sizes, counts)
    
    plot_scaling_results(f_name, padded_input_array, valid_sizes, valid_counts, d_value, fit, r2, save=save, save_path=save_path, show_image=show_image)
    show_image_info(fname=f_name, d_value=d_value, input_array=input_array, sizes=valid_sizes, save=save, save_path=save_path)
    
    print('')
    print(f"D-value for {f_name}: {d_value:.3f}")
    print(f"R^2 value for fit: {r2:.6f}\n")
    print(f"Total pattern width: {input_array.shape[1]}")
    print(f"Total pattern height: {input_array.shape[0]}\n")
    print(f"Smallest object diameter: {smallest_diameter:.1f}")
    print(f"Largest object diameter: {largest_diameter:.1f}\n")
    print(f"min box width: {min_size:.1f}")
    print(f"max box width: {max_size:.1f}")

    if save:
        txt_save_path = os.path.join(save_path, 'image_properties.txt') 
        with open(txt_save_path, mode='w') as file:
            file.write(f"D-value for {f_name}: {d_value:.3f}\n")
            file.write(f"R^2 value for fit {r2}\n\n")
            file.write(f"Total pattern width: {np.max(input_array.shape[1])}\n")
            file.write(f"Total pattern height: {np.max(input_array.shape[0])}\n\n")
            file.write(f"Smallest object diameter: {smallest_diameter:.1f}\n")
            file.write(f"Largest object diameter: {largest_diameter:.1f}\n\n")
            file.write(f"min box width: {min_size:.1f}\n")
            file.write(f"max box width: {max_size:.1f}\n\n")

        norm_array = ((input_array - input_array.min()) / np.ptp(input_array) * 255).astype(np.uint8)
        if norm_array.ndim == 3 and norm_array.shape[2] == 1:
            norm_array = np.squeeze(norm_array, axis=2)
        img = Image.fromarray(norm_array, mode='L')
        tiff_file = os.path.join(save_path, f"{os.path.splitext(f_name)[0]}_thresholded.tif")

        img.save(tiff_file, format='TIFF')

    return d_value, min_size, max_size, r2

def analyze_images(base_path, save=False, invert=True, threshold=150, min_size=16, max_size=None, num_sizes=100, num_pos=10):
    """
    Analyze images in a directory to calculate fractal dimensions and other properties.
    
    Args:
        base_path (str): Path to directory containing images
        save (bool): Whether to save results to files
        invert (bool): Whether to invert the images
        threshold (int): Threshold value for image binarization
        min_size (int): Minimum box size for box counting
        max_size (int): Maximum box size for box counting (defaults to min_size*100)
        num_sizes (int): Number of box sizes to use
        num_pos (int): Number of grid positions to test
    
    Returns:
        pd.DataFrame: DataFrame containing image names and D values
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
        
        im = io.imread(im_path)
        print(np.max(np.unique(im)))

        plt.imshow(input_array, cmap='gray')
        plt.show()

        largest_object, smallest_object, largest_diameter, smallest_diameter, labeled_image = find_largest_smallest_objects(input_array)

        if max_size is None:
            max_size = min_size * 100

        criteria = f'threshold={threshold}'
        
        if save:
            save_path = os.path.join(base_path, os.path.splitext(f_name)[0] + f'_{criteria}')

            if not os.path.exists(save_path):
                os.makedirs(save_path)
        else:
            save_path = None

        padded_input_array = pad_image_for_boxcounting(input_array, max_size, pad_factor=1)
        sizes, counts = boxcount(padded_input_array, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_pos=num_pos)
        valid_sizes, valid_counts, d_value, fit, r2 = compute_fractal_dimension(sizes, counts)
        
        plot_scaling_results(f_name, padded_input_array, valid_sizes, valid_counts, d_value, fit, r2, save=save, save_path=save_path, show_image=False)
        show_image_info(fname=f_name, d_value=d_value, input_array=input_array, sizes=valid_sizes, save=save, save_path=save_path)
        
        print('')
        print(f"D-value for {f_name}: {d_value:.3f}")
        print(f"R^2 value for fit: {r2:.6f}\n")
        print(f"Total pattern width: {input_array.shape[1]}")
        print(f"Total pattern height: {input_array.shape[0]}\n")
        print(f"Smallest object diameter: {smallest_diameter:.1f}")
        print(f"Largest object diameter: {largest_diameter:.1f}\n")
        print(f"min box width: {min_size:.1f}")
        print(f"max box width: {max_size:.1f}")

        D_value_list.append((os.path.splitext(f_name)[0], d_value, min_size, max_size, r2))

        if save:
            txt_save_path = os.path.join(save_path, 'image_properties.txt') 
            with open(txt_save_path, mode='w') as file:
                file.write(f"D-value for {f_name}: {d_value:.3f}\n")
                file.write(f"R^2 value for fit {r2}\n\n")
                file.write(f"Total pattern width: {np.max(input_array.shape[1])}\n")
                file.write(f"Total pattern height: {np.max(input_array.shape[0])}\n\n")
                file.write(f"Smallest object diameter: {smallest_diameter:.1f}\n")
                file.write(f"Largest object diameter: {largest_diameter:.1f}\n\n")
                file.write(f"min box width: {min_size:.1f}\n")
                file.write(f"max box width: {max_size:.1f}\n\n")

            norm_array = ((input_array - input_array.min()) / np.ptp(input_array) * 255).astype(np.uint8)
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
