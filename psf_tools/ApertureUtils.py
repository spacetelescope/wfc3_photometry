import glob
import matplotlib.pyplot as  plt
import numpy as np

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.wcs import WCS
from bisect import bisect_left
from skimage.draw import polygon

# APERTURE UTILS

# Drizzle Image
# Investigate struture- pam or focus?
# Do photometry on either only 10 pix, or first 5 then 10
# Use EE table to correct to infinity
# Get zpt from imphttab
