# MAKE PSF

import numpy as np
from astropy.io import fits 
import matplotlib.pyplot as plt 
import astropy.convolution as ac

pixel_size = 32.0

D = 1.0 # m 
llambda = 630E-9 # m 

diff_limit = 1.22 * llambda / D * 206265

diff_limit_km = diff_limit * 725

diff_limit_px = diff_limit_km / pixel_size

print("info::diffraction limit in pixels is: ", diff_limit_px)

psf = ac.AiryDisk2DKernel(diff_limit_px)

file = fits.PrimaryHDU(psf)
file.writeto("psf_1m_binned.fits")
