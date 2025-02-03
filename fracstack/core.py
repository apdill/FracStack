import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm # type: ignore
import matplotlib.pyplot as plt
import skimage.io as io # type: ignore

from .boxcount import boxcount, compute_dimension
from .image_processing import process_image_to_array, pad_image_for_boxcounting, find_largest_smallest_objects
from .visualization import plot_scaling_results, show_image_info, plot_object_outlines


def measure_dimension(array, mode = 'D0', num_sizes=10, min_size=None, max_size=None, num_offsets=1, pad_factor=1.5):
    if pad_factor is None:
        padded_array = array
    else:
        padded_array = pad_image_for_boxcounting(array, max_size, pad_factor=pad_factor)
        
    sizes, counts = boxcount(padded_array, mode=mode, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets)
    valid_sizes, valid_counts, d_value, fit, r2, ci_low, ci_high = compute_dimension(sizes, counts)
    return {'D': d_value, 'valid_sizes': valid_sizes, 'valid_counts': valid_counts, 'fit': fit, 'R2': r2, 'ci_low': ci_low, 'ci_high': ci_high}

def analyze_image(input_array = None,
                image_path = None,
                save_path = None,
                save_output=False,
                mode='D0', #D0 for fractal dimension, D1 for information dimension, D2 mass fractal dimension
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
                criteria=None):
    """
    Analyze a single image to calculate fractal dimension and other properties.
    
    Args:
        image_path (str): Path to the image file
        input_array (ndarray, optional): Input image array if not loading from file
        save_path (str, optional): Path to save results
        save_output (bool): Whether to save results to files
        invert (bool): Whether to invert the image
        threshold (int): Threshold value for image binarization
        min_size (int): Minimum box size for box counting
        max_size (int): Maximum box size for box counting (defaults to min_size*10)
        num_sizes (int): Number of box sizes to use
        num_offsets (int): Number of grid positions to test
        show_image (bool): Whether to display plots
        plot_objects (bool): Whether to plot object outlines
        criteria (str, optional): Additional criteria string for save path
    
    Returns:
        tuple: (d_value, results_df) where:
            - d_value (float): The calculated fractal dimension
            - results_df (pd.DataFrame): DataFrame containing analysis parameters and results including:
                - d_value: Fractal dimension
                - r2: R-squared value of fit
                - pattern_width: Total pattern width
                - pattern_height: Total pattern height
                - smallest_diameter: Diameter of smallest object
                - largest_diameter: Diameter of largest object
                - min_box_size: Minimum box size used
                - max_box_size: Maximum box size used
                - threshold: Threshold value used
                - num_sizes: Number of box sizes used
                - num_offsetsitions: Number of grid positions tested
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

    fit_info_dict = measure_dimension(input_array, mode=mode, min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, pad_factor=pad_factor, multiprocessing=True)
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

def analyze_images(base_path, save=False, invert=False, threshold=150, min_size=16, max_size=None, num_sizes=100, num_offsets=10, criteria = None):
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
        num_offsets (int): Number of grid positions to test
    
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

        fit_info_dict = measure_dimension(input_array, mode='D0', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets, multiprocessing=True)
        
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

def portfolio_plot(input_array = None,
                min_size=16, 
                max_size=None, 
                num_sizes=100, 
                num_offsets=100,
                figsize=(21, 7),
                save_dir = None,
                f_name = None,
                target_D0 = None,
                D0_threshold = 1,
                R2_D0_threshold = 0,
                D0_D1_threshold = 1):
    
    assert np.array_equal(input_array/np.max(input_array), input_array.astype(bool)), "Input array must be binary (contain only 0s and 1s)"

    if save_dir is not None:
        assert f_name is not None, "f_name must be provided if save_dir is provided"
        save_path = os.path.join(save_dir, f_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    padded_input_array = pad_image_for_boxcounting(input_array, max_size, pad_factor=1.5)

    sizes, counts = boxcount(padded_input_array, mode='D0', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets)
    valid_sizes_d0, valid_counts, d_value_d0, fit_d0, r2_d0, ci_low_d0, ci_high_d0 = compute_dimension(sizes, counts, mode='D0')
    
    sizes, counts = boxcount(padded_input_array, mode='D1', min_size=min_size, max_size=max_size, num_sizes=num_sizes, num_offsets=num_offsets)
    valid_sizes_d1, valid_entropies, d_value_d1, fit_d1, r2_d1, ci_low_d1, ci_high_d1 = compute_dimension(sizes, counts, mode='D1')
    
    D0_D1_check = np.abs(d_value_d0 - target_D0) > D0_D1_threshold
    R2_check = r2_d0 < R2_D0_threshold  
    D0_check = np.abs(d_value_d0 - target_D0) > D0_threshold

    if D0_D1_check or R2_check or D0_check:
        print(f"Skipping, D0 = {d_value_d0:.3f}, D1 = {d_value_d1:.3f}, r2 = {r2_d0:.6f}")
        return None
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    ax1.imshow(input_array, cmap='gray')
    ax1.set_title('Fractal Binary Image', fontsize=22)
    ax1.axis('off')
    
    ax2.scatter(np.log10(valid_sizes_d0), np.log10(valid_counts), color='black')
    ax2.plot(np.log10(valid_sizes_d0), fit_d0[0] * np.log10(valid_sizes_d0) + fit_d0[1], color='red')
    ax2.text(0.7, 0.95, fr'$D_0$ = {d_value_d0:.3f}' '\n' fr'$R^2$ = {r2_d0:.5f}', transform=ax2.transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.3))
    ax2.grid(True)
    ax2.set_title('Fractal Dimension', fontsize=22)
    ax2.set_xlabel(r'Log($\epsilon$)', fontsize=18)
    ax2.set_ylabel(r'Log($N(\epsilon)$)', fontsize=18)
   
    ax3.scatter(np.log10(valid_sizes_d1), valid_entropies, color='black')
    ax3.plot(np.log10(valid_sizes_d1), fit_d1[0] * np.log2(valid_sizes_d1) + fit_d1[1], color='red')
    ax3.grid(True)
    ax3.text(0.7, 0.95, fr'$D_1$ = {d_value_d1:.3f}' '\n' fr'$R^2$ = {r2_d1:.5f}', transform=ax3.transAxes, fontsize=18, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.3))
    ax3.set_title('Information Dimension', fontsize=22)
    ax3.set_xlabel(r'Log($\epsilon$)', fontsize=18)
    ax3.set_ylabel(r'$H(\epsilon)$', fontsize=18)

    plt.tight_layout()

    image_info_dict = {
        'D0': d_value_d0,
        'D1': d_value_d1,
        'R2_D0': r2_d0,
        'R2_D1': r2_d1,
        'min_box_size': min_size,
        'max_box_size': max_size,
        'num_sizes': num_sizes,
        'num_offsets': num_offsets,
        'image_width': input_array.shape[1],
        'image_height': input_array.shape[0],
    }

    if save_dir is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return image_info_dict
