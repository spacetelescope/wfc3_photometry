import astropy.units as u
import numpy as np

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join
from bisect import bisect_left
from copy import copy

def make_id_list(coord_ints, x_digits=None, y_digits=None):
    """
    Turns pixel coordinates into IDs for each star.

    Each ID is just the X and Y integer pixel postions concatenated.
    For example, (3894, 215) becomes 38940215.  These IDs are then
    used in the matching process.  This will likely be rewritten or
    removed soon in favor of better matching methods.

    Parameters
    ----------
    coord_ints : ndarray
        Nx2 array of coordinates.  Must be integers
    x_digits : int, optional
        Number of digits in the x coordinates.  If none, will
        calculate from largest value.  Necessary for correct
        padding of of ID.
    y_digits : int, optional
        Number of digits in the y coordinates.  If none, will
        calculate from largest value.  Necessary for correct
        padding of of ID.

    Returns
    -------
    ids: list
        List of ids for the source positions from the input coords.
    """
    xs = coord_ints[0]
    ys = coord_ints[1]
    if not x_digits:
        x_digits = len(str(max(xs)))
    if not y_digits:
        y_digits = len(str(max(ys)))
    ids = []
    for i in range(len(xs)):
        coord_id = str(xs[i]).zfill(x_digits) + str(ys[i]).zfill(y_digits)
        ids.append(coord_id)
    return ids


def get_match_indices(master_ids, input_ids):
    """
    Matches ID from master list with input ids from input catalog

    Does search for each master_id in input_ids, and returns index
    of matched element in input_ids.  This whole matching system
    will likely be rewritten soon to use better/smarter matching
    methods, like table joins/clustering computations.

    Parameters
    ----------
    master_ids : list
        List of master_ids to search for within input_ids
    input_ids : list
        The list of ids to be searched

    Returns
    -------
    matched_indices : list
        List of indices for each master_id found in input_ids, in
        the same order as master_ids.  If no match was found, the
        value for that master_id in matched_indices is -1.
    """

    matched_indices = []
    input_sorted_inds = np.argsort(input_ids)
    input_ids = sorted(input_ids)
    for master_id in master_ids:
        ind = binary_search_index(input_ids, master_id)

        if ind >= 0:
            matched_indices.append(input_sorted_inds[ind])
        else:
            matched_indices.append(-1)
    print('N matched: {}'.format(len(matched_indices)-matched_indices.count(-1)))
    return matched_indices

def binary_search_index(a, x):
    """Binary searches array a for element x and returns index"""
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    return -1

def match_final_catalogs(cat1, cat2, max_distance=.05):
    """
    Matches two final catalogs so each row of the returned tables
    correspond to the same star.

    This function takes in two final catalogs and matches the sources
    listed in them.  This is done by finding closest point (in sky
    space) of in cat2 for every point in cat1.  Points closer than
    max_distance are considered to be the same source. The
    returned tables only contain the sources that were matched.
    Basically just inner joins the tables when matching rows are
    found.

    Parameters
    ----------
    cat1 : astropy.table.Table or str
        The first table (or filename) to be matched
    cat2 : astropy.table.Table or str
        The second table (or filename) to be matched
    max_distance : float, optional
        The threshold (in arcsec) which distances must be in for
        sources to be considered a match.

    Returns
    -------
    matched_cat1 : astropy.table.Table
        The matched version of cat1
    matched_cat2 : astropy.table.Table
        The matched version of cat2
    """

    if type(cat1) == str:
        cat1 = Table.read(cat1, format='ascii.commented_header')
    if type(cat2) == str:
        cat2 = Table.read(cat2, format='ascii.commented_header')

    cat1_skycoord = SkyCoord(cat1['rbar']*u.deg, cat1['dbar']*u.deg)
    cat2_skycoord = SkyCoord(cat2['rbar']*u.deg, cat2['dbar']*u.deg)

    idx, ang, wat = cat1_skycoord.match_to_catalog_sky(cat2_skycoord)
    distance_mask = ang.arcsec  < max_distance

    matched_cat1 = cat1[distance_mask]
    matched_cat2 = cat2[idx][distance_mask]
    return matched_cat1, matched_cat2

def match_to_master_catalog(master_cat, sci_cat, max_distance=.05):
    """
    Matches a single image extension catalog to the master catalog
    so that each row of the returned catalog is the same star

    This function takes in one final catalog and one catalog of a
    single science extension (sci_cat) and matches sources in them.
    This is done by finding closest point in master_cat (in sky space)
    to each point in sci_cat.  Points closer than max_distance are
    considered to be the same source. The returned table contains the
    same number of rows as master_cat, and has values for each of the
    stars successfully matched. Basically just left joins the tables
    when matching rows are found.

    Parameters
    ----------
    master_cat : astropy.table.Table or str
        The master table (or filename) to be matched to
    sci_cat : astropy.table.Table or str
        The single science extension table (or filename) to be matched
    max_distance : float, optional
        The threshold (in arcsec) which distances must be in for
        sources to be considered a match.

    Returns
    -------
    joined : astropy.table.Table
        The matched (left-joined) table.
    """

    if type(master_cat) == str:
        master_cat = Table.read(master_cat, format='ascii.commented_header')
    if type(sci_cat) == str:
        sci_cat = Table.read(sci_cat, format='ascii.commented_header')

    master_skycoord = SkyCoord(master_cat['rbar']*u.deg,
                               master_cat['dbar']*u.deg)
    sci_skycoord = SkyCoord(sci_cat['r']*u.deg, sci_cat['d']*u.deg)

    idx, ang, wat = sci_skycoord.match_to_catalog_sky(master_skycoord)
    distance_mask = ang.arcsec  < max_distance
    master_cat['id'] = range(len(master_cat))
    idx[ang.arcsec>.05] = -1
    sci_copy= copy(sci_cat)
    sci_copy['id'] = idx
    joined = join(master_cat, sci_copy, keys='id', join_type='left')
    return joined
