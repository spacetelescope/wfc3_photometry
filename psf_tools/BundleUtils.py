import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from astropy.io import fits
from ginga.util import zscale
from matplotlib.colors import LogNorm

sys.path.append('..')
from photometry_tools import RadialProfile
from photutils.centroids import centroid_2dg
from photutils.psf import FittableImageModel
from astropy.modeling import fitting
from scipy.optimize import curve_fit

class Bundle(object):

    def __init__(self, bundle_file):
        self.filename = bundle_file
        self.HDUList = fits.open(bundle_file)
        self.pri_hdr = self.HDUList[0].header
        self.nimages = int(self.pri_hdr['NIMAGES'])
        self._process_header_()
        exts = ['sci', 'err', 'dq', 'upix', 'vpix', 'uv2i', 'uv2j', 'psf']
        for i, ext in enumerate(exts):
            setattr(self, ext, self.HDUList[i+1].data)

        self.fitter = fitting.LevMarLSQFitter()



    def _process_header_(self):
        hdr = self.pri_hdr
        self.pixls = [hdr[key] for key in sorted(hdr['PIXL_*'].keys())]
        self.stems = [hdr[key] for key in sorted(hdr['STEM_*'].keys())]
        self.filts = [hdr[key] for key in sorted(hdr['FILT_*'].keys())]
        self.expts = [hdr[key] for key in sorted(hdr['EXPT_*'].keys())]
        self.rdats = [hdr[key] for key in sorted(hdr['RDAT_*'].keys())]
        self.icens = [hdr[key] for key in sorted(hdr['ICEN_*'].keys())]
        self.jcens = [hdr[key] for key in sorted(hdr['JCEN_*'].keys())]
        self.psfcs = [hdr[key] for key in sorted(hdr['PSFC_*'].keys())]

        all_comments = hdr['COMMENT*'].values()
        comment_lines = [ln for ln in all_comments][-1*self.nimages:]
        self.ucens, self.vcens = np.array([ln.split()[-2:] for ln in comment_lines], dtype=float).T

        return None

    def _make_flattened_psf_(self, fim):

        def psf_mod((x,y), flux, x_0, y_0):
            evaluated = fim.evaluate(x, y, flux, x_0, y_0)
            return np.ravel(evaluated)
        return psf_mod
    #     fim = FittableImageModel(bdl.psf[i], oversampling=4)

    def calculate_local_center(self, cutout):
        center_box_slices = (slice(6,15), slice(6,15))
        cen = centroid_2dg(cutout[center_box_slices])
        return cen + 6

    def fit_psf(self):
        fluxes, x_0s, y_0s, bgs = [], [], [], []
        residuals = []
        y, x = np.mgrid[0:21,0:21]
        radial_dists = np.sqrt((10.-x)**2. + (10.-y)**2.)
        circle_mask = radial_dists >=9.
        for i in range(self.nimages):
            stamp = self.sci[i]
            if np.all(stamp < -900.):
                fluxes.append(np.nan)
                x_0s.append(np.nan)
                y_0s.append(np.nan)
                bgs.append(np.nan)
                residuals.append(np.full(stamp.shape, np.nan))
                continue
            stamp_mask = stamp > -900

            bg_est = np.nanmedian(stamp[stamp_mask & circle_mask])
            center = stamp[8:12,8:12]

            flux_low = np.sum(center[center>-900] - bg_est)

            fim = FittableImageModel(self.psf[i], oversampling=4)
            fit_func = self._make_flattened_psf_(fim)
            bounds=((.5*flux_low, 8., 8.), (np.inf, 12., 12.))
            p0 = [flux_low, 10., 10.]
            popt, pcov = curve_fit(fit_func, (x,y), np.ravel(stamp-bg_est),
                                    bounds=bounds, p0=p0)
            flux, x_0, y_0 = popt
            fluxes.append(flux)
            x_0s.append(x_0)
            y_0s.append(y_0)
            bgs.append(bg_est)

            resid = stamp - fit_func((x,y), *popt).reshape(21,21)-bg_est
            residuals.append(resid)

            # print popt
        self.fluxes = np.array(fluxes)
        self.x_0s = np.array(x_0s)
        self.y_0s = np.array(y_0s)
        self.bgs = np.array(bgs)
        self.residual = np.array(residuals)

        # return fim.evaluate(x, y, flux, x_0, y_0)

    def make_radial_profiles(self, radius=9.):
        fwhms = []
        for i in range(self.nimages):
            if np.all(self.sci[i] < -900.):
                fwhms.append(np.nan)
                continue
            x, y = self.calculate_local_center(self.sci[i])
            tmp = self.sci[i].copy()
            dq_mask = self.dq[i] != 0
            tmp[dq_mask] = np.nan
            rp = RadialProfile(x, y, data=self.sci[i], r=radius, recenter=False)
            fwhms.append(rp.fwhm)
            plt.scatter(rp.distances, rp.values, s=1, alpha=.1, c='b')
        plt.yscale('log')
        plt.ylim(10.,1E5)
        self.fwhms = fwhms


    def show_bundle(self, ext='sci', compact=False, log_scale=True):
        ext_data = getattr(self, ext)
        for i in range(self.nimages):
            if (ext_data[i]<-900).all():
                continue
            else:
                z1, z2 = zscale.zscale(ext_data[i])
                break
            
        ncols = int(math.sqrt(self.nimages))
        nrows = int(math.ceil(float(self.nimages)/ncols))
        if ext == 'sci' or ext == 'psf':
            z2_mult = 50.
        else:
            z2_mult = 2.


        if log_scale:
            norm = LogNorm()
        else:
            norm=None

        if not compact:
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                                    sharex='col', sharey='row',
                                    gridspec_kw={'wspace':.01, 'hspace':.01},
                                    figsize=(ncols*.8, nrows*.8))
            for i in range(self.nimages):
                axs[divmod(i,ncols)].imshow(ext_data[i], origin='lower',
                                            vmin=z1, vmax=z2*10., cmap='magma',
                                            norm=norm)

            if nrows * ncols > self.nimages:
                for j in range(self.nimages % ncols, ncols):
                    fig.delaxes(axs[-1, j])

        elif compact:
            dims = (nrows*21, ncols*21)
            tmp = ext_data
            if nrows * ncols > self.nimages:
                n_extra = nrows * ncols - self.nimages
                tmp = np.vstack([tmp, np.zeros((n_extra, 21, 21))])
            tmp = tmp.reshape(nrows, ncols, 21, 21)
            tmp = np.vstack(([np.hstack(col) for col in tmp]))
            plt.imshow(tmp, origin='lower', vmin=z1, vmax=z2*z2_mult,
                       cmap='magma', norm=norm)
