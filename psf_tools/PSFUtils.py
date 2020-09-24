import numpy as np
import os

from astropy.io import fits
from astropy.wcs import WCS
from drizzlepac import astrodrizzle, tweakback
from itertools import product
from photutils.psf.models import GriddedPSFModel

from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import NDData


# from .PyFitting import make_models, get_subtrahend
from .PSFPhot import get_standard_psf, _get_exec_path
from .CatalogUtils import get_pam_func


def _generate_input_coordinates(wcs_naxis, spacing=150.25, offset=100):
    xmax, ymax = wcs_naxis
    input_grid = product(np.arange(offset, xmax, spacing),
                         np.arange(offset, ymax, spacing))
    input_grid = [grid_point for grid_point in input_grid]
    return np.array(input_grid)

def _prefilter_coordinates(input_skycoords, xmin, xmax, ymin, ymax, flt_wcs):
    # this probably will explode if the image contains a celestial pole?
    inp_ra, inp_dec = input_skycoords.T

    corners = [p for p in product([xmin, xmax], [ymin, ymax])]
    corner_ra, corner_dec = flt_wcs.all_pix2world(corners, 1).T

    mask_ra = (inp_ra>np.amin(corner_ra)) & (inp_ra < np.max(corner_ra))
    mask_dec = (inp_dec>np.amin(corner_dec)) & (inp_dec<np.amax(corner_dec))
    mask = mask_ra & mask_dec
    return input_skycoords[mask]


def _transform_points(input_skycoords, flt_wcs, padding=9):

    xmin = ymin = -1 * padding
    xmax, ymax = np.array(flt_wcs._naxis) + padding

    filtered_coords = _prefilter_coordinates(input_skycoords, xmin, xmax,
                                             ymin, ymax, flt_wcs)

    xc, yc = flt_wcs.all_world2pix(filtered_coords, 1).T
    mask = (xc > xmin) & (xc < xmax) & (yc > ymin) & (yc < ymax)

    return np.array([xc[mask], yc[mask]])

def make_model_star_image(drz, input_images=None, models_only=True,
                        input_coordinates=None, psf_file=None):
    """
    Main function for making false star drz/drc

    This function takes in a user supplied drizzled image, and makes an ouput
    image that contains a grid of PSFs as they would be projected in the
    drizzled frame.  The corresponding PSF/input images are determined by
    looking at the header of the drizzled image automatically.

    Parameters
    ----------
    drz : str
        Path to drizzled image to derive frame parameters from.
    input_images : list, optional
        List of the input flt/flc images.  If none, determined from header, but
        the flts/flcs MUST be in the same directory as the drz/drc.  In
        addition, it is critical that the images are well aligned, or the
        output image is not representative of the actual PSF.
    models_only : bool, optional
        Make the output drizzled image only contain the PSF models?  Default
        True (CURRENTLY DOES NOTHING- Ignore)
    input_coordinates : `numpy.ndarray`, optional
        Grid of pixel positions at which to insert the model stars (in drizzle
        frame pixel coordinates).  If none, a grid is automatically generated.
        See _generate_input_coordinates().
    psf_file : str, optional
        Path to file containing PSF models.  If none, automatically determined
        and downloaded using header information of `drz`.

    Returns
    -------
    output_files : list
        List containing the paths to the false star flts/flcs.
    """
    # Should probably roll a lot of this into other function
    drz_hdr0 = fits.getheader(drz, 0)
    if drz_hdr0['INSTRUME'] != 'WFC3' and psf_file is None:
        raise ValueError('Image is not a WFC3 Image, and thus not supported')

    if input_images is None:
        input_images = tweakback.extract_input_filenames(drz)
        drz_dir = os.path.split(drz)[0]
        input_images = [os.path.join(drz_dir, im) for im in input_images]

    if psf_file == None:
        path = os.path.dirname(_get_exec_path())
        filt = drz_hdr0['FILTER']
        all_psf_filts = ['F105W', 'F110W', 'F125W', 'F127M', 'F139M',
                        'F140W', 'F160W', 'F225W', 'F275W', 'F336W',
                        'F390W', 'F438W', 'F467M', 'F555W', 'F606W',
                        'F775W', 'F814W', 'F850LP']
        if filt not in all_psf_filts:
            raise ValueError('No PSF to download for {}'.format(filt))
        psf_file = get_standard_psf(path, filt)

    mods = make_models(psf_file)

    if fits.getdata(drz, 0) is not None:
        drz_wcs = WCS(fits.getheader(drz, 0))
    else:
        drz_wcs = WCS(fits.getheader(drz, 1))

    if input_coordinates is None:
        input_coordinates = _generate_input_coordinates(drz_wcs._naxis)
    input_skycoords = drz_wcs.all_pix2world(input_coordinates, 1)

    output_files = []
    for im in input_images:
        print('Making false star image for {}'.format(im))
        complete_image = insert_in_exposure(im, input_skycoords, mods)
        output_files.append(complete_image)

    dumb_drizzle(drz, output_files)
    return output_files


def insert_in_exposure(flt, input_skycoords, psf_models):
    # TODO: Make this handle subarrays

    # handle both flt and flc case
    new_name = flt.replace('.fits', '_star.fits')


    hdul = fits.open(flt) # DO NOT USE MODE='UPDATE'
    det = hdul[0].header['DETECTOR'].lower()

    sci_count  = 0

    # zero out dat arrays
    for hdu in hdul:
        if hdu.name in ['SCI', 'ERR']:
            hdu.data *= 0. # floating point data
            if hdu.name == 'SCI':
                sci_count += 1
        elif hdu.name =='DQ':
            hdu.data *= 0 # int data

    # Get the false stars in!
    for i in range(sci_count):
        ext = hdul['SCI',i+1]
        ext_wcs = WCS(ext.header, hdul)
        flt_positions = np.array(_transform_points(input_skycoords, ext_wcs))

        if det == 'ir':
            pam_func = get_pam_func(det)
            mod_ind = 0
        elif det == 'uvis':
            chip = ext.header['CCDCHIP']
            mod_ind = 2 - chip # 0 if UVIS2, 1 if UVIS1
            pam_func = get_pam_func(det + str(chip))

        pam_values = pam_func(*flt_positions - 1.)
        fluxes = hdul[0].header['EXPTIME']/pam_values

        false_image = get_subtrahend(*flt_positions, fluxes,
                                      psf_models[mod_ind], ext.data.shape)
        ext.data += false_image

    hdul.writeto(new_name, overwrite=True)
    return new_name

def dumb_drizzle(drz, false_images):
    drz_hdr0 = fits.getheader(drz)
    out = drz.replace('_drz', '_star_drz')
    out = drz.replace('_drc', '_star_drc')
    print('OUTPUT NAME SHOULD BE {}'.format(out))
    astrodrizzle.AstroDrizzle(false_images, build=True, clean=True,
                              in_memory=True, final_refimage=drz,
                              final_wcs=True, skysub=False,
                              median=False, blot=False,
                              driz_cr=False, driz_separate=False, static=False,
                              final_pixfrac=drz_hdr0['D001PIXF'],
                              preserve=False, context=False, output=out)



#---------------------------SUBTRACTION--------------------------------

def compute_cutout(x, y, flux, mod, shape):
    """Gets cutout of evaluated model, and indices to place cutout"""

    # Get integer pixel positions of centers, with appropriate
    # edge convention, i.e. edges of pixels are integers, 0 indexed
    x_cen = int(x-.5)
    y_cen = int(y-.5)

    # Handle cases where model spills over edge of data array
    x1 = max(0, x_cen - 10)
    y1 = max(0, y_cen - 10)

    # upper bound is exclusive, so add 1 more
    x2 = min(x_cen + 11, shape[1])
    y2 = min(y_cen + 11, shape[0])

    y_grid, x_grid = np.mgrid[y1:y2, x1:x2]
    cutout = mod.evaluate(x_grid, y_grid, flux, x-1., y-1.)
    return cutout, x_grid, y_grid

def get_subtrahend(xs, ys, fluxes, mod, shape):
    """Make the image to be subtracted"""

    # Initialize the array to be subtracted
    subtrahend = np.zeros(shape, dtype=float)

    for x, y, flux in zip(xs, ys, fluxes):
        if flux == np.nan: # Skip ones without good fluxes
            continue
        cutout, x_grid, y_grid = compute_cutout(x, y, flux, mod, shape)

        # Important: use += to account for overlapping cutouts
        subtrahend[y_grid, x_grid] += cutout

    return subtrahend

def subtract_psfs(data, cat, mod):
    """Subtracts the fitted PSF from the positions in catalog"""

    shape = data.shape

    fluxes = np.power(10, cat['m']/-2.5).data # Convert from mags to fluxes
    xs = cat['x'].data
    ys = cat['y'].data

    # Evaluate the PSF at each x, y, flux, and place it in subtrhend
    subtrahend = get_subtrahend(xs, ys, fluxes, mod, shape)


    # Subtact the image!
    difference = data - subtrahend
    return difference

#--------------------------Focus PSF-------------------------------------


def make_models(psf_file):
    """
    Reads in PSF model, splits it for multichip data, and makes model objects.

    This function takes a fits file containing a spatially dependent PSF
    (single PSF models at each point in a grid across the detector) and reads
    them into GriddedPSFModel objects that can be used for fitting.  Since the
    PSF model files contain models for both chips (in the case of UVIS) put
    together, this file splits them into the appropriate sections for each
    chip.

    Parameters
    ----------
    psf_file : str
        Name of fits file containing the PSFs

    Returns
    -------
    mod1 : `photutils.psf.models.GriddedPSFModel`
        The gridded PSF model object for the 'SCI, 1' data.
    mod1 : `photutils.psf.models.GriddedPSFModel` or None
        The gridded PSF model object for the 'SCI, 2' data.
        If there is no 'SCI, 2' for that detector (i.e. WFC3/IR), then is None.
    """
    # Probably only need to add support for subarray into this part.
    hdu = fits.open(psf_file)
    psf_data = hdu[0].data
    hdr = hdu[0].header

    xlocs = [hdr['IPSFX'+str(i).zfill(2)] for i in range(1,11)]
    xlocs = np.array([xloc for xloc in xlocs if xloc != 9999]) -1

    ylocs = [hdr['JPSFY'+str(i).zfill(2)] for i in range(1,11)]
    ylocs = np.array([yloc for yloc in ylocs if yloc != 9999]) -1



    if len(ylocs) > 4:   # 2 chips/UVIS data
        ylocs1 = ylocs[:4]
        ylocs2 = ylocs[4:]-2048

        g_xypos1 = [p[::-1] for p in product(ylocs1, xlocs)]
        g_xypos2 = [p[::-1] for p in product(ylocs2, xlocs)]

        if len(psf_data.shape) == 3:
            ndd1 = NDData(data=psf_data[:28],
                          meta={'grid_xypos':g_xypos1, 'oversampling':4})
            mod1 = GriddedPSFModel(ndd1)

            ndd2 = NDData(data=psf_data[28:],
                          meta={'grid_xypos':g_xypos2, 'oversampling':4})
            mod2 = GriddedPSFModel(ndd2)

        elif len(psf_data.shape) == 4:
            print('using focus psf')
            ndd1 = NDData(data=psf_data[:,:28, :, :],
                          meta={'grid_xypos':g_xypos1, 'oversampling':4})
            mod1 = SlowGriddedFocusPSFModel(ndd1)

            ndd2 = NDData(data=psf_data[:,:28, :, :],
                          meta={'grid_xypos':g_xypos2, 'oversampling':4})
            mod2 = SlowGriddedFocusPSFModel(ndd2)

        return(mod1, mod2)

    else:
        # IR Case
        g_xypos = [p[::-1] for p in product(ylocs, xlocs)]
        ndd1 = NDData(data=psf_data,
                      meta={'grid_xypos':g_xypos, 'oversampling':4})
        mod1 = GriddedPSFModel(ndd1)
        return(mod1, None)

class SlowGriddedFocusPSFModel(GriddedPSFModel):
    flux = Parameter(description='Intensity scaling factor for the PSF '
                     'model.', default=1.0)
    x_0 = Parameter(description='x position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    y_0 = Parameter(description='y position in the output coordinate grid '
                    'where the model is evaluated.', default=0.0)
    def __init__(self, data, flux=flux.default, x_0=x_0.default,
                 y_0=y_0.default, fill_value=0.0):
        if not isinstance(data, NDData):
            raise TypeError('data must be an NDData instance.')
        if data.data.ndim != 4:
            raise ValueError('The NDData data attribute must be a 4D numpy '
                             'ndarray')

        self.nfoc = data.data.shape[0]
        ndds = [NDData(data=data.data[i], meta=data.meta) for i in range(self.nfoc)]
        self.models = [GriddedPSFModel(ndd) for ndd in ndds]
        super().__init__(ndds[0], flux, x_0, y_0)
        self.data = data.data
        self.meta = data.meta
        self.interp_model = GriddedPSFModel(ndds[0])
        self._focus_level = 0.


    def interp_focus(self, focus):
        if not 0 <= focus <= self.nfoc-1:
            raise ValueError('Focus level {} not in range \
                             [0, {}]'.format(focus, self.nfoc))
        if focus != int(focus):
            left = np.floor(focus)
            right = np.ceil(focus)
            delta = right - focus
            weights = np.array([delta, 1 - delta])

            # Right bound exclusive
            interp_data = np.sum(self.data[int(left):int(right)+1] * weights[:,None,None,None],
                                 axis=0)
            foc_model = GriddedPSFModel(NDData(data = interp_data, meta=self.meta))
            # self.interp_model._data = interp_data

        else:
            # self.interp_model._data = self.data[int(focus)]
            foc_model = self.models[int(focus)]


        self.interp_model = foc_model
        self._focus_level = focus
        return self.interp_model

    def evaluate(self, x, y, flux, x_0, y_0, focus=None):
        if focus == None:
            focus = self._focus_level
        if focus != self._focus_level:
            self.interp_focus(focus)
        return self.interp_model.evaluate(x, y, flux, x_0, y_0)
