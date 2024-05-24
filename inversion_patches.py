# INVERSION

import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut

b_c = ut.readFits('convolved_data.fits')
b_c_mean = np.mean(b_c[:,:,0,-10:])
b_c /= b_c_mean
psf = ut.readFits('psf.fits')

def waveGrid(nw):
    wav = np.arange(nw) * 0.01 + 6301.0
    return wav

def loadData(clip_threshold=0.99):
    obs = b_c
    wav = waveGrid(obs.shape[-1])

    sig = np.zeros([4, len(wav)])
    sig[:, :] = 1e-3
    sig[1:4, :] /= 10.0

    return [[wav, None]], [[obs, sig, psf, clip_threshold]]

if __name__ == "__main__":
    nthreads = 16 
    patch_size = 128
    overlap = 10

    # Load data
    region, sregion = loadData()

    # Init ME inverter
    me = ME.MilneEddington(region, [6301, 6302], nthreads=nthreads)

    # Generate initial model
    ny, nx = sregion[0][0].shape[0:2]
    Ipar = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    m = me.repeat_model(Ipar, ny, nx)

    patch_size_with_overlap = patch_size + 2 * overlap

    for y in range(0, ny, patch_size - overlap):
        for x in range(0, nx, patch_size - overlap):
            # Define the patch limits considering the overlap
            patch_y_start = y
            patch_y_end = min(y + patch_size_with_overlap, ny)
            patch_x_start = x
            patch_x_end = min(x + patch_size_with_overlap, nx)

            # Perform inversion on the patch
            patch_obs = sregion[0][0][patch_y_start:patch_y_end, patch_x_start:patch_x_end]
            patch_sig = sregion[0][1][:, patch_x_start:patch_x_end]
            mpix, syn, chi2 = me.invert(m, patch_obs, patch_sig, nRandom=8, nIter=10, chi2_thres=1.0, mu=0.96)

            # Save the inversion results
            np.save(f'inversion_{patch_y_start}_{patch_x_start}.npy', mpix)

            # Smooth model
            m = ut.smoothModel(mpix, 4)

            # Invert spatially-coupled with initial guess from pixel-to-pixel (less iterations)
            m1, chi = me.invert_spatially_coupled(mpix, sregion, mu=0.96, nIter=10, alpha=100., \
                                                   alphas=np.float64([1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), \
                                                   init_lambda=10)

            # Smooth model with very narrow PSF and restart with less regularization (lower alpha)
            m = ut.smoothModel(m1, 2)
            
            # Invert spatially-coupled 
            m1, chi = me.invert_spatially_coupled(m, sregion, mu=0.96, nIter=20, alpha=10, \
                                                   alphas=np.float64([2, 2, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]), \
                                                   init_lambda=1.0)

            # Write the results to a FITS file
            ut.writeFits("modelout_sc.fits", m1)

            # Free up memory
            del m1, chi, m, mpix, syn, chi2

            # Reinitialize the model for the next patch
            m = me.repeat_model(Ipar, ny, nx)
