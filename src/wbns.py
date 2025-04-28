import numpy as np
from pywt import wavedecn, waverecn
from scipy.ndimage import gaussian_filter
from joblib import Parallel, delayed
import multiprocessing

def _wavelet_based_BG_subtraction(image, num_levels, noise_lvl):
    """
    Perform wavelet-based background and noise estimation on a single 2D image slice.
    
    Parameters:
        image (np.ndarray): A 2D image slice.
        num_levels (int): Number of decomposition levels.
        noise_lvl (int): Noise level parameter.
        
    Returns:
        tuple: (Background, Noise, BG_unfiltered) as 2D arrays.
    """
    coeffs = wavedecn(image, 'db1', level=None)
    coeffs2 = coeffs.copy()
    
    # Zero out the detail coefficients for levels 1 to num_levels-1 to estimate background.
    for BGlvl in range(1, num_levels):
        coeffs[-BGlvl] = {k: np.zeros_like(v) for k, v in coeffs[-BGlvl].items()}
    
    Background = waverecn(coeffs, 'db1')
    BG_unfiltered = Background
    Background = gaussian_filter(Background, sigma=2**num_levels)
    
    # Estimate noise: preserve the approximation and first few details
    coeffs2[0] = np.ones_like(coeffs2[0])
    for lvl in range(1, len(coeffs2) - noise_lvl):
        coeffs2[lvl] = {k: np.zeros_like(v) for k, v in coeffs2[lvl].items()}
    Noise = waverecn(coeffs2, 'db1')
    
    return Background, Noise, BG_unfiltered

def substract_background(image: np.ndarray, resolution_px: int, noise_lvl: int) -> np.ndarray:
    """
    Subtract the estimated background and noise from an image.
    
    The function first performs a wavelet-based background and noise estimation on each slice 
    of the image (adding a singleton axis if a 2D image is provided). It then subtracts the 
    estimated background and a noise correction from the original image.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array. Can be 2D or 3D (stack of 2D slices).
        resolution_px (int): Resolution in pixels (FWHM of the PSF) used to determine the number 
                             of wavelet decomposition levels.
        noise_lvl (int): Noise level parameter; if resolution_px > 6, a noise_lvl of 2 may be better.
    
    Returns:
        np.ndarray: The background- and noise-corrected image with the same data type as the input.
    """
    # Store the original data type
    orig_dtype = image.dtype
    
    # Convert image to float32 for processing
    image = np.array(image, dtype='float32')
    
    # If a 2D image is provided, add a new axis to treat it as a single slice
    if image.ndim == 2:
        image = image[np.newaxis, ...]
    
    # Pad image if needed so that spatial dimensions are even.
    pad_1 = False
    pad_2 = False
    shape = image.shape
    if shape[1] % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 1), (0, 0)), mode='edge')
        pad_1 = True
    if shape[2] % 2 != 0:
        image = np.pad(image, ((0, 0), (0, 0), (0, 1)), mode='edge')
        pad_2 = True
    
    # Determine number of wavelet decomposition levels from resolution_px
    num_levels = np.uint16(np.ceil(np.log2(resolution_px)))
    
    # Process each slice in parallel
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores, max_nbytes=None)(
        delayed(_wavelet_based_BG_subtraction)(image[i], num_levels, noise_lvl) for i in range(image.shape[0])
    )
    Background, Noise, _ = zip(*results)
    Background = np.asarray(Background, dtype='float32')
    Noise = np.asarray(Noise, dtype='float32')
    
    # Remove padding if it was added
    if pad_1:
        image = image[:, :-1, :]
        Background = Background[:, :-1, :]
        Noise = Noise[:, :-1, :]
    if pad_2:
        image = image[:, :, :-1]
        Background = Background[:, :, :-1]
        Noise = Noise[:, :, :-1]
    
    # Subtract the estimated background from the image (first pass)
    temp_result = image - Background
    temp_result[temp_result < 0] = 0  # enforce positivity
    
    # Correct noise: apply positivity constraint and thresholding
    Noise[Noise < 0] = 0
    noise_threshold = np.mean(Noise) + 2 * np.std(Noise)
    Noise[Noise > noise_threshold] = noise_threshold
    
    # Final result: subtract both background and noise, then enforce positivity
    result = image - Background - Noise
    result[result < 0] = 0
    
    # If the input was 2D, return a 2D array
    if result.shape[0] == 1:
        result = result[0]
    
    return result.astype(orig_dtype)
