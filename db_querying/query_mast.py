import sys
import os
import time
import re
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib
except ImportError:  # Python 2.x
    import httplib

from astropy.table import Table
from matplotlib.patches import Polygon


import pprint
pp = pprint.PrettyPrinter(indent=4)

def mastQuery(request):
    """Perform a MAST query.

        Parameters
        ----------
        request (dictionary): The MAST request json object

        Returns head,content where head is the response HTTP headers, and content is the returned data"""

    server='mast.stsci.edu'

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)

    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content

def parse_footprint(s_region):
    reg = s_region.split()[1:]
    if len(reg) % 2:
        reg = reg[1:]
    reg = np.array(reg).astype(float).reshape(len(reg)/2,2)
    reg[:,0] = reg[:,0]%360.
    return reg

def plot_footprints(s_regions,ax=None):
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    colors = cm.rainbow(np.linspace(0, 1, len(s_regions)))
    for i, s_reg in enumerate(s_regions):
        reg = parse_footprint(s_reg)
        rect = Polygon(reg, alpha=.1, closed=True, fill=False, color=colors[i])
        ax.add_patch(rect)
    plt.show()

if __name__ == '__main__':
    objectOfInterest = 'GD153'

    resolverRequest = {'service':'Mast.Name.Lookup',
                         'params':{'input':objectOfInterest,
                                   'format':'json'},
                         }

    headers,resolvedObjectString = mastQuery(resolverRequest)

    resolvedObject = json.loads(resolvedObjectString)


    objRa = resolvedObject['resolvedCoordinate'][0]['ra']
    objDec = resolvedObject['resolvedCoordinate'][0]['decl']

    mashupRequest = {
            "service":"Mast.Caom.Filtered.Position",
            "format":"json",
            "params":{
                "columns":"*",
                "filters":[
                    {"paramName":"dataproduct_type",
                     "values":["image"]},
                    {"paramName":"project",
                      "values":["HST"]},
                    {"paramName":"instrument_name",
                      "values":["WFC3/UVIS"]}
                    ],
                "obstype":"all",
                "position":"{}, {}, 0.3".format(objRa, objDec)
            }}


    headers,mastDataString = mastQuery(mashupRequest)

    mastData = json.loads(mastDataString)

    # print(mastData.keys())
    print("Query status:",mastData['status'])

    # pp.pprint(mastData['fields'])
    data = mastData['data']
    # pp.pprint(data[0])
    print 'N: {}'.format(len(mastData['data']))
    df = pd.read_json(json.dumps(data))
    # print df['proposal_id']
    df = df.loc[(df['target_name'] != 'NONE') & (df['calib_level'] > 2)]
    # print set(df.loc[df['target_name'] == 'NONE']['obs_title'])
    print len(df)
    ax = df.plot('s_ra','s_dec',kind='scatter',alpha=.3)


    plt.xlim(195,194)
    plt.ylim(21.8,22.2)
    plot_footprints(df['s_region'].values, ax)
    # print df.columns
    print df['obs_id']
