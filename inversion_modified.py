# Importing packages

import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut
from tqdm import tqdm

convolved = fits.open('binned_convolved_data.fits')[0].data[0:128,0:128, :,:]
convolved_part = convolved

hdu = fits.PrimaryHDU(convolved_part)
hdul = fits.HDUList([hdu])
hdul.writeto('convolved_part.fits', overwrite=True)

# Importing data

def waveGrid(nw):

    wav = np.arange(nw) * 0.01 + 6301.0
    return wav

def loadData(clip_threshold = 0.99):

    obs = ut.readFits('convolved_part.fits')
    wav = waveGrid(obs.shape[-1])

    psf = ut.readFits('psf.fits')
    sig = np.zeros([4,len(wav)])
    sig[:,:] = 1e-3
    sig[1:4,:] /= 5.0

    return [[wav, None]], [[obs, sig, psf, clip_threshold]]

if __name__ == "__main__":

    nthreads = 16 # adapt this number to the number of cores that are available in your machine

    # Load data
    region, sregion = loadData()
# Init ME inverter
    me = ME.MilneEddington(region, [6301, 6302], nthreads=nthreads)

    # generate initial model
    nx, ny = sregion[0][0].shape[0:2]
    Ipar = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    m = me.repeat_model(Ipar, nx, ny)

    # Invert pixel by pixel
    mpix, syn, chi2 = me.invert(m, sregion[0][0], sregion[0][1], nRandom=50, nIter=20, chi2_thres=1.0, mu=0.96)
    ut.writeFits("modelout_pix.fits", mpix)

mpix = ut.readFits('modelout_pix.fits')

import gc

# Define the alpha values for the first and second inversion steps
alpha_values_m2 = [50, 75, 100, 125, 150, 175, 200, 225]
alpha_values_m4 = [1, 5, 10, 25, 35]

# Create an empty list to store the results
results = []

# Loop through the alpha values for the first inversion
for alpha_m2 in alpha_values_m2:

    m1 = ut.smoothModel(mpix, 4)

    # Perform the first inversion
    m2, chi = me.invert_spatially_coupled(
        m1, sregion, mu=1.0, nIter=5, alpha=alpha_m2,
        alphas=np.float64([1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
        init_lambda=10
    )

    # Smooth the model with a very narrow PSF
    m3 = ut.smoothModel(m2, 2)

    # Loop through the alpha values for the second inversion
    for alpha_m4 in alpha_values_m4:
        # Perform the second inversion
        m4, chi = me.invert_spatially_coupled(
m3, sregion, mu=1.0, nIter=10, alpha=alpha_m4,
            alphas=np.float64([2, 2, 2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
            init_lambda=1.0
        )

        # Write the result to a FITS file
        filename = f"modelout_sc_{alpha_m2}_{alpha_m4}.fits"
        ut.writeFits(filename, m4)

        # Store the result in the results list
        results.append((alpha_m2, alpha_m4, filename))

        # Free up memory
     #   del m1, m2, m3, m4
      #  gc.collect()

# Print the results
for alpha_m2, alpha_m4, filename in results:
    print(f"Alpha_m2: {alpha_m2}, Alpha_m4: {alpha_m4}, Output File: {filename}")
