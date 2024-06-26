import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut
from tqdm import tqdm
from astropy.nddata import block_reduce
import astropy.convolution as ac
import scipy.ndimage as ndimage
from scipy.ndimage import zoom

# Importing data
stokes_original = fits.open('qs_ph_disk_center_synth.fits')[0].data[:512, :512, :, :]

# Modeling PSF
llambda_0 = 630E-9  # m
D = 1  # m
pixel_size = 57 

theta_0 = 1.22 * llambda_0 / D * 206265  # Angular resolution in arcseconds
theta_km = theta_0 * 720 # Convert arcseconds to kilometers
theta_pixel = theta_km / pixel_size  # Size in pixels

# Creating Airy Disk 2D Kernel
psf = ac.AiryDisk2DKernel(theta_pixel)
psf_array = psf.array
psf_array /= psf_array.sum()  # Normalize the PSF array

ut.writeFits('psf.fits', psf_array)

########################################################################################################################

# Binning factor
bin_factor = 57 / 16 

# Target shape for the spatial dimensions
target_shape_x = int(stokes_original.shape[0] / bin_factor)
target_shape_y = int(stokes_original.shape[1] / bin_factor)

# Initializing new shape
new_shape = (target_shape_x, target_shape_y, stokes_original.shape[2], stokes_original.shape[3])
binned_data = np.zeros(new_shape)

# Calculation of precise zoom factors to match the target shape
zoom_factors = (target_shape_x / stokes_original.shape[0], target_shape_y / stokes_original.shape[1])

if __name__ == "__main__":
    nthreads = 16  # Adapt this number to the number of cores that are available in your machine
    
    # Applying zoom for each Stokes parameter and wavelength
    for i in tqdm(range(stokes_original.shape[3])):
        for j in range(stokes_original.shape[2]):
            # Zoom the data
            binned_data[:, :, j, i] = zoom(stokes_original[:, :, j, i], zoom_factors, order=0)  # order=0 for nearest neighbor

    # ut.writeFits('binned.fits', binned_data)

    # Convolution
    convolved_data = np.zeros((binned_data.shape))
    for i in tqdm(range(binned_data.shape[3])): 
        for j in range(binned_data.shape[2]): 
            result = ac.convolve(binned_data[:, :, j, i], psf_array, boundary='wrap')  # Check if 'wrap' is appropriate
            convolved_data[:, :, j, i] = result

    # Normalize convolved data
    mean_continuum = np.mean(convolved_data[:, :, 0, -10:])
    convolved_data /= mean_continuum

    ut.writeFits('convolved_data.fits', convolved_data)

