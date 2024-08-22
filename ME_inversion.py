import MilneEddington as ME
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
import utils as ut
from tqdm import tqdm
import gc
from scipy.ndimage import gaussian_filter

x = fits.open('hinode_data.fits')[0].data[124:, 256:644, :,:] # FOV 388x388
mean_continuum = np.mean(x[:, :, 0, -10:])
x = x.copy().astype(np.float64)
x /=  mean_continuum

# Masking based on Stokes I
#mean_stokes_I = np.mean(x[:, :, 0, :], axis=-1)
#mask_I = np.mean(x[:, :, 0, :], axis=-1) < 0.2 * mean_stokes_I
#nan_mask = np.isnan(x[:,:,0,0])
#x[mask_I, :, :] = np.nan

# Gaussian filtering
#sigma = 1
#for i in range(x.shape[2]):
#    for j in range(x.shape[3]):
#        x[:, :, i, j] = gaussian_filter(x[:, :, i, j], sigma=sigma)

hdu = fits.PrimaryHDU(x)
hdul = fits.HDUList([hdu])
hdul.writeto('to_invert.fits', overwrite=True)

def doubleGrid(obs):
    ny, nx, ns, nw = obs.shape
    obs1 = np.zeros((ny, nx, ns, nw*2))
    obs1[:,:,:,0::2] = obs

    wav = (np.arange(nw*2, dtype='float64')-nw)*0.010765 + 6302.08

    return wav, obs1

def loadData(clip_threshold = 0.99):

    wav, obs = doubleGrid(ut.readFits('to_invert.fits'))
    tr = np.float64([0.00240208, 0.00390950, 0.0230995, 0.123889, 0.198799,0.116474,0.0201897,0.00704875,0.00277027]) 
    psf = ut.readFits('hinode_psf_0.16.fits')

    sig = np.zeros((4, 112*2)) + 1.e32
    sig[:,0::2] = 1.e-3
    sig[1:3] /= 4.0
    sig[3] /= 3.0

    return [[wav, tr/tr.sum()]], [[obs, sig, psf/psf.sum(), clip_threshold]]

def divide_fov_with_overlap(mpix, overlap=4, region_size=132):

    regions = []
    nx, ny = mpix.shape[0:2]

    regions.append(((0, 132), (0, 132), mpix[0:132, 0:132, :]))
    regions.append(((128, 260), (0, 132), mpix[128:260, 0:132, :]))
    regions.append(((256, 388), (0, 132), mpix[256:388, 0:132, :]))

    regions.append(((0, 132), (128, 260), mpix[0:132, 128:260, :]))
    regions.append(((128, 260), (128, 260), mpix[128:260, 128:260, :]))
    regions.append(((256, 388), (128, 260), mpix[256:388, 128:260, :]))

    regions.append(((0, 132), (256, 388), mpix[0:132, 256:388, :]))
    regions.append(((128, 260), (256, 388), mpix[128:260, 256:388, :]))
    regions.append(((256, 388), (256, 388), mpix[256:388, 256:388, :]))

    return regions 
    
if __name__ == "__main__":
    nthreads = 16 

region, sregion = loadData()
obs, sig, psf, clip_threshold = sregion[0]

me = ME.MilneEddington(region, [6301, 6302], nthreads=nthreads)
nx, ny = sregion[0][0].shape[0:2]
Ipar = np.float64([1000, 1,1,0.01,0.02,20.,0.1, 0.2,0.7])
m = me.repeat_model(Ipar, nx, ny)
    
mpix, syn, chi2 = me.invert(m, sregion[0][0], sregion[0][1], nRandom=15, nIter=20, chi2_thres=1.0, mu=0.96)

ut.writeFits('piksel_part_0.fits', mpix)
ut.writeFits('chi2_0.fits', chi2)

for idx, (x_range, y_range, region) in enumerate(divide_fov_with_overlap(mpix)):
        
    m1 = ut.smoothModel(region, 4)
    m2, chi2 = me.invert_spatially_coupled(m1, sregion,
            mu=0.96, nIter=1, alpha=100,
            alphas=np.float64([1, 1, 1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
            init_lambda=10)
    
    m3 = ut.smoothModel(m2, 2)
    m4, chi2 = me.invert_spatially_coupled(m3, sregion,
            mu=0.96, nIter=2, alpha=10,
            alphas=np.float64([2, 2, 2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]),
            init_lambda=1)

    ut.writeFits(f'spatial_part_{idx}.fits', m4)
    ut.writeFits(f'spatial_chi2_{idx}.fits', chi2)

    del m1, m2, m3, m4, chi2, region
    gc.collect()
    
print("Inversion completed for all regions with overlap.")