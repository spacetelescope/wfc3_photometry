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
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

class Bundle(object):

    def __init__(self, bundle_file):
        self.filename = bundle_file
        self.HDUList = fits.open(bundle_file)
        self.pri_hdr = self.HDUList[0].header
        self.nimages = int(self.pri_hdr['NIMAGES'])
        self._process_header_()
        self.group_indices()
        exts = ['sci', 'err', 'dq', 'upix', 'vpix', 'uv2i', 'uv2j', 'psf']
        for i, ext in enumerate(exts):
            setattr(self, ext, self.HDUList[i+1].data)

        self.fitter = fitting.LevMarLSQFitter()



    def _process_header_(self):
        hdr = self.pri_hdr
        get_vals = lambda kw_root : [hdr[key] for key in sorted(hdr[kw_root].keys())]
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

    def clip_mean(self, vals):
        clip = sigmaclip(vals)[0]
        return np.mean(clip)

    def fit_psf(self):
        fluxes, x_0s, y_0s, bgs, qs = [], [], [], [], []
        residuals = []
        y, x = np.mgrid[0:21,0:21]
        radial_dists = np.sqrt((10.-x)**2. + (10.-y)**2.)
        circle_mask = radial_dists >=8.
        for i in range(self.nimages):
            stamp = self.sci[i]
            if np.all(stamp < -900.):
                fluxes.append(np.nan)
                x_0s.append(np.nan)
                y_0s.append(np.nan)
                bgs.append(np.nan)
                residuals.append(np.full(stamp.shape, np.nan))
                qs.append(np.nan)
                continue
            stamp_mask = stamp > -900

            bg_est = self.clip_mean(stamp[stamp_mask & circle_mask])
            center = stamp[8:13,8:13]

            flux_low = np.sum(center[center>-900] - bg_est)

            fim = FittableImageModel(self.psf[i], oversampling=4)
            fit_func = self._make_flattened_psf_(fim)
            # bounds=((.5*flux_low, 9., 9.), (np.inf, 11., 11.))
            bounds=((.5*flux_low, 9.4, 9.4), (np.inf, 10.6, 10.6))
            p0 = [flux_low, 10., 10.]
            yf, xf = np.mgrid[8:13,8:13]
            popt, pcov = curve_fit(fit_func, (xf,yf), np.ravel(center-bg_est),
                                    bounds=bounds, p0=p0)
            flux, x_0, y_0 = popt
            fluxes.append(flux)
            x_0s.append(x_0)
            y_0s.append(y_0)
            bgs.append(bg_est)


            small_resid = center - fit_func((xf,yf), *popt).reshape(5,5) - bg_est
            resid = stamp - fit_func((x,y), *popt).reshape(21,21)-bg_est
            q = np.sum(np.abs(small_resid))/flux
            residuals.append(resid)
            qs.append(q)

            # print popt
        self.fluxes = np.array(fluxes)
        self.x_0s = np.array(x_0s)
        self.y_0s = np.array(y_0s)
        self.bgs = np.array(bgs)
        self.residual = np.array(residuals)
        self.qs = np.array(qs)


    def group_indices(self):
        all_inds = np.arange(self.nimages)
        uniq = list(set([stem[:6] for stem in self.stems]))
        n_epochs = len(uniq)
        ep_inds = []
        for root in sorted(uniq):
            tmp = [j for j in all_inds if root in self.stems[j]]
            ep_inds.append(tmp)
        self.epoch_inds = np.array(ep_inds)


    def make_radial_profiles(self, radius=9., s=1, alpha=.1, **kwargs):
        fwhms = []
        if 'c' not in kwargs.keys() and 'color' not in kwargs.keys():
            kwargs['c'] = 'b'
        for i in range(self.nimages):
            if np.all(self.sci[i] < -900.):
                fwhms.append(np.nan)
                continue
            try:
                x = self.x_0s[i]
                y = self.y_0s[i]
            except AttributeError:
                x, y = self.calculate_local_center(self.sci[i])
            tmp = self.sci[i].copy()
            dq_mask = self.dq[i] != 0
            tmp[dq_mask] = np.nan
            rp = RadialProfile(x, y, data=self.sci[i], r=radius, recenter=False)
            fwhms.append(rp.fwhm)
            plt.scatter(rp.distances, rp.values, s=s, alpha=alpha, **kwargs)
        plt.yscale('log')
        plt.ylim(10.,1E5)
        plt.ylabel('Flux [electrons]')
        plt.xlabel('Radius [pixels]')

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

    def xy_to_uv(self):
        us, vs = [], []
        tmp = np.arange(21)
        for i in range(self.nimages):
            rbu = RectBivariateSpline(tmp, tmp, self.upix[i].T, kx=1, ky=1)
            rbv = RectBivariateSpline(tmp, tmp, self.vpix[i].T, kx=1, ky=1)
            us.append(rbu(self.x_0s[i], self.y_0s[i])[0,0])
            vs.append(rbv(self.x_0s[i], self.y_0s[i])[0,0])
        return np.array([us, vs]).T
