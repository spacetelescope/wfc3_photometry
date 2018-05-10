import numpy as np
import matplotlib.pyplot as plt

from photutils import centroid_com, centroid_1dg, centroid_2dg
from photutils.aperture import CircularAperture
from scipy.optimize import curve_fit

def fit_profile(distances, values, ax=None):
    """Fits 1d Moffat function to measured radial profile.

    Fits a moffat profile to the distance and values of the pixels.
    If the fit is successful, the full width at half max is returned.
    If not, np.nan is returned.  If an axes object is passed in, it will
    plot the fitted model.  Further development may expose the model
    to the user and possibly allow user defined models.

    Parameters
    ----------
    distances : array
        The distances from each pixel center to the centroid if source
    values : array
        The values of the the corresponding pixels.
    ax: matplotlib.axes.Axes
        An axes object to plot fit on.  If not set, no plot is made.

    Returns
    -------
    fwhm : float
        The full width at half maximum of the fit model.
    """
    try:
        best_vals, covar = curve_fit(profile_model,
                                    distances,
                                    values,
                                    p0 = [np.amax(values), 1.5, 1.5, 0],
                                    bounds = ([0., .3, .5, 0],
                                              [np.inf, 10., 10., np.inf]))
    except:
        # print('Fit Failed')
        return np.nan

    hwhm = best_vals[1] * np.sqrt(2. ** (1./best_vals[2]) - 1.)
    fwhm = 2 * hwhm
    if ax is not None:
        tmp_r = np.arange(0,np.ceil(np.amax(distances)),.1)
        label = r'$\gamma$= {}, $\alpha$ = {}'.format(round(best_vals[1],2), round(best_vals[2],2))
        label += '\nFWHM = {}'.format(round(2. * hwhm, 2))
        ax.plot(tmp_r, profile_model(tmp_r, *(best_vals)), label=label)
        ax.legend(loc=1)

    return fwhm

def profile_model(r, amp, gamma, alpha, bias):
    """Returns 1D Moffat profile evaluated at r values.

    This function takes radius values and parameters in a simple 1D
    moffat profiles and returns the values of the profile at those
    radius values.  The model is defined as:
    model = amp * (1. + (r / gamma) ** 2.) ** (-1. * alpha) + bias

    Parameters
    ----------
    r : array
        The distances at which to sample the model
    amp : float
        The amplitude of the of the model
    gamma: float
        The width of the profile.
    alpha: float
        The decay of the profile.
    bias: float
        The bias level (piston term) of the data.  This is like a background
        value.

    Returns
    -------
    model : array
        The values of the model sampled at the r values.
    """
    model = amp * (1. + (r / gamma) ** 2.) ** (-1. * alpha) + bias
    return model

def radial_profile(x, y, data, r=5, fit=False, show=False, ax=None, recenter=False):
    """Main function to calulate radial profiles

    Computes a radial profile of a source in an array.  This function
    leverages some of the tools in photutils to cutout the small region
    around the source.  This function can first recenter the source
    via a 2d Gaussian fit (radial profiles are sensitive to centroids)
    and then fit a 1D Moffat profile to the values.  The profile
    is calculated by computing the distance from the center of each
    pixel within a box of size r to the centroid of the source in the
    box.  Additionally, the profile and the fit can be plotted.
    If fit is set to True, then the profile is fit with a 1D Moffat.
    If show is set to True, then profile (and/or fit) is plotted.
    If an axes object is provided, the plot(s) will be on that object.
    NOTE: THE POSITIONS ARE 0 INDEXED (bottom left corner pixel
    center is set to (0,0)).

    This may be re written into a class to be more flexible in the future.


    Parameters
    ----------
    x : float
        The x position of the centroid of the source. ZERO INDEXED
    y : float
        The y position of the centroid of the source. ZERO INDEXED
    data : array
        A 2D array containing the full image data.  A small box
        is cut out of this array for the radial profile
    r : float
        The size of the box used to cut out the source pixels.
        The box is typically square with side length ~ 2*r + 1.
    fit:  bool
        Fit a 1D Moffat profile?  Default false
    show : bool
        Plot the profile?  Default false.  See ax parameter for info.
    ax : matplotlib.axes.Axes
        Axes object to make the plots on.  If None and show is True,
        a axes object will be created.  If None and show is False,
        no plot is made.
    recenter : bool
        If true, a new centroid is computed by fitting a 2D Gaussian.

    Returns
    -------
    fwhm : float
        The full width at half max of profile, returned if fit=True.
        Otherwise None is returned
    x : float
        If recenter is True, return the new x position
    y : float
        If recenter is True, return the new y position
    """
    # Create the axes object/cutout etc
    cutout, sx, sy = setup_cutout(x, y, data, r)

    if cutout is None:
        return np.nan

    if recenter:
        cutout, x, y, sx, sy = recenter_source(cutout, x, y, sx,
                                               sy, data,r)
        if cutout is None: # Just in case?
            return np.nan, np.nan, np.nan


    # Flip order for array convention
    iY, iX = np.mgrid[sy, sx]
    # extent = [sx.start, sx.stop-1, sy.start, sy.stop-1]

    distances = np.sqrt((iX - x) ** 2. + (iY - y) ** 2. ).flatten()
    values = cutout.flatten()


    if show:
        ax = show_profile(distances, values, ax)

    # If show is True and fit is True, fit will also be overplotted
    if fit:
        fwhm = fit_profile(distances, values, ax)
        if recenter:
            return fwhm, x, y
        return fwhm

    return

def recenter_source(cutout, x, y, sx, sy, data, r):
    """Recenters source position in cutout and returns new position"""

    xg1, yg1 = centroid_2dg(cutout)
    dx = xg1 + sx.start - x
    dy = yg1 + sy.start - y
    dr = (dx ** 2. + dy ** 2.) ** .5
    if dr > 2.:
        print('Large shift of {},{} computed.'.format(dx, dy))
        print('Rejecting and using original x, y coordinates')

    else:
        # This is bad DRY practice.
        x, y = xg1 + sx.start, yg1 + sy.start
        cutout, sx, sy = setup_cutout(x, y, data, r)
    return cutout, x, y, sx, sy

def setup_cutout(x, y, data, r):
    """Cuts out the aperture and returns slice objects

    General setup procedure.  Mostly an internal function.
    """
    ap = CircularAperture((x,y), r=r)
    mask = ap.to_mask()[0]
    sy = mask.slices[0]
    sx = mask.slices[1]

    cutout = mask.cutout(data, fill_value=np.nan)
    return cutout, sx, sy

def show_profile(distances, values, ax=None):
    """Makes plot of radial profile

    Plots the radial profile, that is pixel distance vs
    pixel value.  Can plot on an existing axes object if
    the an axes object is passed in via the ax parameter.
    The function attempts to set sensible axes limits, specifically
    half the minimum of the values array, unless that minumum value
    is negative (not allowed as plot uses logarithmic y scale).  In
    that case, the lower y limit is set to 10^-3.  The axes object is
    returned by this, so that can be set by the user later.

    Parameters
    ----------
    distances : array
        The distances from each pixel center to the centroid if source
    values : array
        The values of the the corresponding pixels.
    ax: matplotlib.axes.Axes
        An axes object to plot the radial profile on (for integrating)
        the plot into other figures.  If not set, the script will create
        an axes object.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the radial profile plot/
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.scatter(distances, values, alpha=.5)
    min_y = np.amin(values)/2.
    if min_y <= 0:
        min_y = np.amin(values[values >0.])/2.
    ax.set_ylim(min_y, np.amax(values)*2.)
    ax.set_yscale('log')
    return ax
