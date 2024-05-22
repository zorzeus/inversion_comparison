# INVERSION

import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut

# Importing data

def waveGrid(nw):
    
    wav = np.arange(nw) * 0.01 + 6301.0
    return wav
    
def loadData(clip_threshold = 0.99):

    b_c = ut.readFits('binned_and_convolved_stokes.fits')
    print("info::input file has size: ", b_c.shape)
    mean_continuum = np.mean(b_c[:,:,0,-10:])
    b_c /= mean_continuum

    obs = b_c[:64,:64] 
    wav = waveGrid(obs.shape[-1])
    print (wav)
 
    sig = np.zeros([4,len(wav)])
    sig[:,:] = 1e-3
    sig[1:4,:] /= 10.0

    psf = ut.readFits('psf_1m_binned.fits')
    
    return [[wav, None]], [[obs, sig, psf/psf.sum(), clip_threshold]]

if __name__ == "__main__":

    nthreads = 12 # adapt this number to the number of cores that are available in your machine
    
    # Load data
    region, sregion = loadData()

    # Init ME inverter
    me = ME.MilneEddington(region, [6301, 6302], nthreads=nthreads)
    
    # generate initial model
    nx, ny = sregion[0][0].shape[0:2]
    print("nx, ny", nx, ny)
    Ipar = np.float64([500., 0.1, 0.1, 0.0, 0.04, 100, 0.5, 0.1, 1.0])
    m = me.repeat_model(Ipar, nx, ny)
    
    # Invert pixel by pixel
    mpix, syn, chi2 = me.invert(m, sregion[0][0], sregion[0][1], nRandom=20, nIter=50, chi2_thres=0.1, mu=0.96)
    ut.writeFits("modelout_pixel-to-pixel.fits", mpix)
    # smooth model
    m = ut.smoothModel(mpix, 4)


    # invert spatially-coupled with initial guess from pixel-to-pixel (less iterations)
    m1, chi = me.invert_spatially_coupled(m, sregion, mu=0.96, nIter=5, alpha=100., \
                                    alphas = np.float64([1,1,1,0.01,0.01,0.01,0.01,0.01,0.01]),\
                                    init_lambda=10.0)

    

    # smooth model with very narrow PSF and restart with less regularization (lower alpha)
    #m = ut.smoothModel(m1, 2)

    
    # invert spatially-coupled 
    #m1, chi = me.invert_spatially_coupled(m, sregion, mu=0.96, nIter=20, alpha=10., \
    #                                     alphas = np.float64([2,2,2,0.01,0.01,0.01,0.01,0.01,0.01]),\
    #                                     init_lambda=1.0)
    
    ut.writeFits("modelout_spatially_coupled.fits", m1)