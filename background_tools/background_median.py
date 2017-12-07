import numpy as np
from astropy.table import Table

def aperture_stats_tbl(data, apertures, method='exact'):
    """Computes mean, median, and mode statistics inside Photutils 
    apertures. This is primarily intended for estimating 
    backgrounds via annulus apertures.  The intent is that this 
    falls easily into other code to provide background measurements.
    
    Parameters
    ----------
    data : array
        The data for the image to be measured.
    apertures : photutils.aperture.core.PixelAperture object (or subclass)
        The phoutils aperture object to measure the stats in.
        i.e. the object returned via CirularAperture, CircularAnnulus, 
        or RectangularAperture etc.
    method: str
        The method by which to handle the pixel overlap.  Defaults to 
        computing the exact area.
        NOTE: Currently, this will actually fully include a pixel where 
        the aperture has ANY overlap,
        as a median is also being performed.  If the method is set to 
        'center' the pixels will only be included if the pixel's center 
        falls within the aperture.
        
    Returns
    -------
    stats_tbl : astropy.table.Table
        An astropy Table with the colums X, Y, aperture_mean, 
        aperture_median, aperture_mode,
        and a row for each of the positions of the apertures.
    
    """
    
    # Get the masks that will be used to identify our desired pixels.
    masks = apertures.to_mask(method=method) 
    # Compute the stats of pixels within the masks
    aperture_stats = [calc_aperture_mmm(data, mask) for mask in masks]
    aperture_stats = np.array(aperture_stats)
    
    # Place the array of the x y positions alongside the stats
    stacked = np.hstack([apertures.positions, aperture_stats])
    # Name the columns
    names = ['X','Y','aperture_mean','aperture_median','aperture_mode']
    # Make the table
    stats_tbl = Table(data=stacked, names=names)
    return stats_tbl

def calc_aperture_mmm(data, mask):
    """Helper function to actually calculate the stats for pixels 
        falling within some Photutils aperture mask on some array
        of data.
    """
    cutout = mask.cutout(data, fill_value=np.nan)
    if cutout is None:
        return (np.nan, np.nan, np.nan)
    else:
        values = cutout * mask.data / mask.data
        mean = np.nanmean(values)
        median = np.nanmedian(values)
        mode = 3 * median - 2 * mean
        return (mean, median, mode)
